//! Conformer encoder on MLX arrays.
//!
//! Layout notes (safetensors weights are PyTorch-layout and transposed at load):
//!   - MLX Conv2d is channels-last: input (N,H,W,C), weight (O,kH,kW,I).
//!     PyTorch weights are OIHW → transposed once in `pt_conv2d_to_mlx`.
//!   - MLX Conv1d: input (N,L,C), weight (O,kW,I); see `pt_conv1d_to_mlx`.

use anyhow::{Context, Result};

use super::array::Array;
use super::ops;
use super::weights::MlxWeights;
use crate::config::ModelConfig;

// ---------------------------------------------------------------------------
// Weight transposition helpers (PyTorch layout → MLX layout)
// ---------------------------------------------------------------------------

/// PyTorch Conv2d weight: (O, I/g, kH, kW) → MLX: (O, kH, kW, I/g)
fn pt_conv2d_to_mlx(w: Array) -> Array {
    ops::transpose(&w, &[0, 2, 3, 1])
}

/// PyTorch Conv1d weight: (O, I/g, kW) → MLX: (O, kW, I/g)
fn pt_conv1d_to_mlx(w: Array) -> Array {
    ops::transpose(&w, &[0, 2, 1])
}

// ---------------------------------------------------------------------------
// ConvSubsampling
//   Three stride-2 Conv2d passes with ReLU, then a linear projection.
//   PyTorch input convention:  (N, C, H, W) = (1, 1, T, n_mels)
//   MLX input convention:      (N, H, W, C) = (1, T, n_mels, 1)
// ---------------------------------------------------------------------------
struct ConvSubsampling {
    c0_w: Array, // (256, 3, 3, 1)  — MLX layout after transpose
    c0_b: Array,
    c2_w: Array, // (256, 3, 3, 1)  — depthwise groups=256
    c2_b: Array,
    c3_w: Array, // (256, 1, 1, 256) — pointwise
    c3_b: Array,
    c5_w: Array,
    c5_b: Array,
    c6_w: Array,
    c6_b: Array,
    out_w: Array, // linear: (d_model, feat)
    out_b: Array,
    c2_groups: i32,
    c5_groups: i32,
}

impl ConvSubsampling {
    fn load(weights: &MlxWeights, prefix: &str) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{prefix}{n}"))?.shallow_clone())
        };
        let c2_w = pt_conv2d_to_mlx(get("conv.2.weight")?);
        let c5_w = pt_conv2d_to_mlx(get("conv.5.weight")?);
        // Depthwise groups == output channels of that depthwise conv.
        let c2_groups = c2_w.dim(0);
        let c5_groups = c5_w.dim(0);
        Ok(Self {
            c0_w: pt_conv2d_to_mlx(get("conv.0.weight")?),
            c0_b: get("conv.0.bias")?,
            c2_w,
            c2_b: get("conv.2.bias")?,
            c3_w: pt_conv2d_to_mlx(get("conv.3.weight")?),
            c3_b: get("conv.3.bias")?,
            c5_w,
            c5_b: get("conv.5.bias")?,
            c6_w: pt_conv2d_to_mlx(get("conv.6.weight")?),
            c6_b: get("conv.6.bias")?,
            out_w: get("out.weight")?,
            out_b: get("out.bias")?,
            c2_groups,
            c5_groups,
        })
    }

    /// x: (1, n_mels, T) with `valid` real frames → (output, full T', valid T').
    ///
    /// The centered STFT pad is a real time step of zeros. Strided convs can
    /// turn that into an extra nonzero frame, so the valid length is tracked
    /// with the PyTorch formula and everything past it is zeroed before the
    /// next stride, matching `MaskedConvSequential`.
    fn forward(&self, x: &Array, valid: i32) -> (Array, i32, i32) {
        // (1, n_mels, T) → (1, T, n_mels) → (1, T, n_mels, 1)
        let x = ops::transpose(x, &[0, 2, 1]);
        let x = ops::expand_dims(&x, &[3]);
        let mut valid = valid.min(x.dim(1));
        let x = zero_tail(&x, 1, valid);

        // Conv0: (1, T, n_mels, 1) → (1, T/2, n_mels/2, 256)
        let x = ops::conv2d(&x, &self.c0_w, 2, 2, 1, 1, 1);
        let x = add_bias_nhwc(&x, &self.c0_b);
        let x = ops::relu(&x);
        valid = conv_out_len(valid, self.c0_w.dim(1), 2, 1).min(x.dim(1));
        let x = zero_tail(&x, 1, valid);

        // Conv2 (depthwise) + Conv3 (pointwise)
        let x = ops::conv2d(&x, &self.c2_w, 2, 2, 1, 1, self.c2_groups);
        let x = add_bias_nhwc(&x, &self.c2_b);
        let x = ops::conv2d(&x, &self.c3_w, 1, 1, 0, 0, 1);
        let x = add_bias_nhwc(&x, &self.c3_b);
        let x = ops::relu(&x);
        valid = conv_out_len(valid, self.c2_w.dim(1), 2, 1).min(x.dim(1));
        let x = zero_tail(&x, 1, valid);

        // Conv5 (depthwise) + Conv6 (pointwise)
        let x = ops::conv2d(&x, &self.c5_w, 2, 2, 1, 1, self.c5_groups);
        let x = add_bias_nhwc(&x, &self.c5_b);
        let x = ops::conv2d(&x, &self.c6_w, 1, 1, 0, 0, 1);
        let x = add_bias_nhwc(&x, &self.c6_b);
        let x = ops::relu(&x);
        valid = conv_out_len(valid, self.c5_w.dim(1), 2, 1).min(x.dim(1));
        let x = zero_tail(&x, 1, valid);

        // x: (1, T', n_mels/8, 256) in NHWC — transpose to match PyTorch's
        // NCHW flatten order: (1, T', 256, n_mels/8) → (1, T', 256*n_mels/8)
        let x = ops::transpose(&x, &[0, 1, 3, 2]);
        let t_prime = x.dim(1);
        let feat = x.dim(2) * x.dim(3);
        let x = ops::reshape(&x, &[1, t_prime, feat]);

        // Linear projection: (1, T', feat) → (1, T', d_model).
        // Pad steps are zero going in, so they come out as the bias; the
        // attention mask (not another zeroing) drops them, as in the model.
        let out = ops::linear(&x, &self.out_w, &self.out_b);
        (out, t_prime, valid.min(t_prime))
    }
}

/// Add a (C,) bias to an NHWC array by broadcasting over N, H, W.
fn add_bias_nhwc(x: &Array, bias: &Array) -> Array {
    // bias: (C,) → (1, 1, 1, C) via reshape
    let c = bias.dim(0);
    let b = ops::reshape(bias, &[1, 1, 1, c]);
    ops::add(x, &b)
}

/// Output length of one conv axis. Same formula as PyTorch:
/// `(len + 2*pad - kernel) / stride + 1`.
fn conv_out_len(len: i32, kernel: i32, stride: i32, pad: i32) -> i32 {
    if len <= 0 {
        return 0;
    }
    (len + 2 * pad - kernel) / stride + 1
}

/// Zero time steps at `axis` from `valid` onward. A no-op when every step is real.
fn zero_tail(x: &Array, axis: i32, valid: i32) -> Array {
    let t = x.dim(axis);
    if valid >= t {
        return x.shallow_clone();
    }
    if valid <= 0 {
        return ops::zeros(&x.shape());
    }
    let n = x.ndim();
    let starts = vec![0i32; n];
    let mut stops: Vec<i32> = (0..n as i32).map(|d| x.dim(d)).collect();
    stops[axis as usize] = valid;
    let head = ops::slice(x, &starts, &stops);
    let mut tail_shape = x.shape();
    tail_shape[axis as usize] = t - valid;
    let tail = ops::zeros(&tail_shape);
    ops::cat(&[&head, &tail], axis)
}

/// Self-attention mask for a sequence whose first `valid` steps are real.
/// `additive` is added to scores; `keep` zeros the padded rows after softmax.
struct SequenceMask {
    additive: Array,
    keep: Array,
}

fn sequence_mask(time: i32, valid: i32) -> Option<SequenceMask> {
    if valid >= time {
        return None;
    }
    let (additive, keep) = pad_mask_values(time.max(0) as usize, valid.clamp(0, time) as usize);
    let shape = [1, 1, time.max(0), time.max(0)];
    Some(SequenceMask {
        additive: Array::from_data_f32(&additive, &shape),
        keep: Array::from_data_f32(&keep, &shape),
    })
}

/// Flat (time, time) additive scores and post-softmax keep flags.
/// A position is padding when its query or key index is `>= valid`.
fn pad_mask_values(time: usize, valid: usize) -> (Vec<f32>, Vec<f32>) {
    let mut additive = vec![0f32; time * time];
    let mut keep = vec![1f32; time * time];
    for q in 0..time {
        for k in 0..time {
            if q >= valid || k >= valid {
                additive[q * time + k] = -1.0e9;
                keep[q * time + k] = 0.0;
            }
        }
    }
    (additive, keep)
}

// ---------------------------------------------------------------------------
// Relative Positional Encoding (standard Conformer sinusoidal formula)
// ---------------------------------------------------------------------------
fn rel_positional_encoding(length: usize, d_model: usize) -> Array {
    let n_pos = 2 * length - 1;
    let mut pe = vec![0.0f32; n_pos * d_model];
    for i in 0..n_pos {
        let pos = (length as i32 - 1 - i as i32) as f32;
        for k in (0..d_model).step_by(2) {
            let div = ((k as f32) * -(10000.0f32.ln()) / d_model as f32).exp();
            pe[i * d_model + k] = (pos * div).sin();
            if k + 1 < d_model {
                pe[i * d_model + k + 1] = (pos * div).cos();
            }
        }
    }
    Array::from_data_f32(&pe, &[1, n_pos as i32, d_model as i32])
}

// ---------------------------------------------------------------------------
// ConformerFeedForward  (Linear → SiLU → Linear)
// ---------------------------------------------------------------------------
struct FeedForward {
    l1_w: Array,
    l1_b: Array,
    l2_w: Array,
    l2_b: Array,
}

impl FeedForward {
    fn load(weights: &MlxWeights, prefix: &str) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{prefix}{n}"))?.shallow_clone())
        };
        Ok(Self {
            l1_w: get("linear1.weight")?,
            l1_b: get("linear1.bias")?,
            l2_w: get("linear2.weight")?,
            l2_b: get("linear2.bias")?,
        })
    }

    fn forward(&self, x: &Array) -> Array {
        let h = ops::silu(&ops::linear(x, &self.l1_w, &self.l1_b));
        ops::linear(&h, &self.l2_w, &self.l2_b)
    }
}

// ---------------------------------------------------------------------------
// ConformerConvolution
//   pointwise_conv1 → GLU → depthwise_conv → BatchNorm → SiLU → pointwise_conv2
//
//   MLX Conv1d: input (N,L,C), weight (O,kW,I) — NHWC analog for 1D.
//   PyTorch Conv1d weights (O,I/g,kW) must be transposed → (O,kW,I/g).
// ---------------------------------------------------------------------------
struct ConformerConv {
    pw1_w: Array, // (2*d_model, 1, d_model) after MLX transpose
    pw1_b: Array,
    dw_w: Array, // (d_model, kW, 1) after MLX transpose
    dw_b: Array,
    // BatchNorm parameters (eval mode: normalize by running stats then affine)
    bn_w: Array,
    bn_b: Array,
    bn_rm: Array,
    bn_rv: Array,
    pw2_w: Array, // (d_model, 1, d_model) after MLX transpose
    pw2_b: Array,
    d_model: i32,
}

impl ConformerConv {
    fn load(weights: &MlxWeights, prefix: &str, d_model: i32) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{prefix}{n}"))?.shallow_clone())
        };
        Ok(Self {
            pw1_w: pt_conv1d_to_mlx(get("pointwise_conv1.weight")?),
            pw1_b: get("pointwise_conv1.bias")?,
            dw_w: pt_conv1d_to_mlx(get("depthwise_conv.weight")?),
            dw_b: get("depthwise_conv.bias")?,
            bn_w: get("batch_norm.weight")?,
            bn_b: get("batch_norm.bias")?,
            bn_rm: get("batch_norm.running_mean")?,
            bn_rv: get("batch_norm.running_var")?,
            pw2_w: pt_conv1d_to_mlx(get("pointwise_conv2.weight")?),
            pw2_b: get("pointwise_conv2.bias")?,
            d_model,
        })
    }

    fn forward(&self, x: &Array, valid: i32) -> Array {
        // x: (B, T, d_model) — already NLC (channels-last) for MLX conv1d
        // Pointwise conv1: (B, T, 2*d_model)
        let x = ops::conv1d(x, &self.pw1_w, 1, 0, 1);
        let b1 = ops::reshape(&self.pw1_b, &[1, 1, self.pw1_b.dim(0)]);
        let x = ops::add(&x, &b1);

        // GLU: split along last dim, gate with sigmoid
        let (a, gate) = split_last(&x, self.d_model);
        let x = ops::mul(&a, &ops::sigmoid(&gate));
        // Padded steps must not leak into valid ones through the depthwise kernel.
        let x = zero_tail(&x, 1, valid);

        // Depthwise conv1d: kernel size inferred from weight shape (O, kW, 1)
        let kw = self.dw_w.dim(1);
        let pad = (kw - 1) / 2;
        let x = ops::conv1d(&x, &self.dw_w, 1, pad, self.d_model);
        let b2 = ops::reshape(&self.dw_b, &[1, 1, self.dw_b.dim(0)]);
        let x = ops::add(&x, &b2);

        // BatchNorm eval mode: (x - running_mean) / sqrt(running_var + eps) * w + b
        // x: (B, T, C) — apply stats along last dim
        let x = batch_norm_nlc(&x, &self.bn_w, &self.bn_b, &self.bn_rm, &self.bn_rv);

        // SiLU activation
        let x = ops::silu(&x);

        // Pointwise conv2: (B, T, d_model)
        let x = ops::conv1d(&x, &self.pw2_w, 1, 0, 1);
        let b3 = ops::reshape(&self.pw2_b, &[1, 1, self.pw2_b.dim(0)]);
        ops::add(&x, &b3)
    }
}

/// Split last dimension at `split_at`, returning (left, right).
fn split_last(x: &Array, split_at: i32) -> (Array, Array) {
    let ndim = x.ndim() as i32;
    let last = ndim - 1;
    let total = x.dim(last);

    // Build slice bounds for left half: all dims full, last dim [0, split_at)
    let n = ndim as usize;
    let mut starts = vec![0i32; n];
    let mut stops: Vec<i32> = (0..ndim).map(|d| x.dim(d)).collect();

    stops[last as usize] = split_at;
    let left = ops::slice(x, &starts, &stops);

    starts[last as usize] = split_at;
    stops[last as usize] = total;
    let right = ops::slice(x, &starts, &stops);

    (left, right)
}

/// BatchNorm in eval mode for NLC layout (x: B,T,C).
fn batch_norm_nlc(
    x: &Array,
    weight: &Array,
    bias: &Array,
    running_mean: &Array,
    running_var: &Array,
) -> Array {
    let eps = 1e-5f32;
    let c = weight.dim(0);

    // Broadcast 1-D stats to (1, 1, C)
    let rm = ops::reshape(running_mean, &[1, 1, c]);
    let rv = ops::reshape(running_var, &[1, 1, c]);
    let w = ops::reshape(weight, &[1, 1, c]);
    let b = ops::reshape(bias, &[1, 1, c]);

    // (x - mean) * rsqrt(var + eps) * w + b
    // Using rsqrt (1/sqrt) keeps the entire computation on GPU — no CPU round-trips.
    let x_centered = ops::sub(x, &rm);
    let var_eps = ops::add(&rv, &Array::from_data_f32(&[eps], &[1, 1, 1]));
    let inv_std = ops::rsqrt(&var_eps);
    let x_norm = ops::mul(&x_centered, &inv_std);

    ops::add(&ops::mul(&x_norm, &w), &b)
}

// ---------------------------------------------------------------------------
// RelPositionMultiHeadAttention
// ---------------------------------------------------------------------------
struct RelPosAttn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    pos_w: Array,
    out_w: Array,
    out_b: Array,
    pos_bias_u: Array, // (n_heads, d_k)
    pos_bias_v: Array,
    n_heads: i32,
    d_k: i32,
    scale: f32,
}

impl RelPosAttn {
    fn load(weights: &MlxWeights, prefix: &str, n_heads: i32, d_model: i32) -> Result<Self> {
        let d_k = d_model / n_heads;
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{prefix}{n}"))?.shallow_clone())
        };
        Ok(Self {
            q_w: get("linear_q.weight")?,
            q_b: get("linear_q.bias")?,
            k_w: get("linear_k.weight")?,
            k_b: get("linear_k.bias")?,
            v_w: get("linear_v.weight")?,
            v_b: get("linear_v.bias")?,
            pos_w: get("linear_pos.weight")?,
            out_w: get("linear_out.weight")?,
            out_b: get("linear_out.bias")?,
            pos_bias_u: get("pos_bias_u")?,
            pos_bias_v: get("pos_bias_v")?,
            n_heads,
            d_k,
            scale: (d_k as f32).powf(-0.5),
        })
    }

    /// Relative shift: x (B, H, T, 2T-1) → (B, H, T, T)
    ///
    /// Matches the model's `RelPositionMultiHeadAttention.rel_shift`: pad one
    /// zero column on the left of the last dimension, view as (2T, T), drop
    /// the first row, view back as (T, 2T-1), then narrow the last dimension
    /// to T. The result is out[i][j] = x[i][T-1-i+j].
    fn rel_shift(&self, x: &Array, t: i32) -> Array {
        rel_shift(x, t)
    }

    fn forward(&self, x: &Array, pos_emb: &Array, mask: Option<&SequenceMask>) -> Array {
        let b = x.dim(0);
        let t = x.dim(1);

        // Project Q, K, V: (B, T, d_model) → (B, H, T, d_k)
        let reshape_qkv = |z: &Array| -> Array {
            let r = ops::reshape(z, &[b, t, self.n_heads, self.d_k]);
            ops::transpose(&r, &[0, 2, 1, 3])
        };

        let q = reshape_qkv(&ops::linear(x, &self.q_w, &self.q_b));
        let k = reshape_qkv(&ops::linear(x, &self.k_w, &self.k_b));
        let v = reshape_qkv(&ops::linear(x, &self.v_w, &self.v_b));

        // Positional projection: (1, 2T-1, d_model) → (1, H, 2T-1, d_k)
        let n_pos = pos_emb.dim(1);
        // pos_w has no bias — zero bias
        let pos_bias_zero = ops::zeros(&[1]);
        let p = ops::linear(pos_emb, &self.pos_w, &pos_bias_zero);
        let p = ops::reshape(&p, &[1, n_pos, self.n_heads, self.d_k]);
        let p = ops::transpose(&p, &[0, 2, 1, 3]);

        // Add content/position biases
        let u = ops::reshape(&self.pos_bias_u, &[1, self.n_heads, 1, self.d_k]);
        let v_bias = ops::reshape(&self.pos_bias_v, &[1, self.n_heads, 1, self.d_k]);
        let q_u = ops::add(&q, &u);
        let q_v = ops::add(&q, &v_bias);

        // Attention scores
        let p_t = ops::transpose_last2(&p);
        let k_t = ops::transpose_last2(&k);
        let matrix_ac = ops::matmul(&q_u, &k_t);
        let matrix_bd = ops::matmul(&q_v, &p_t);
        let matrix_bd = self.rel_shift(&matrix_bd, t);

        let scores = ops::scale(&ops::add(&matrix_ac, &matrix_bd), self.scale);
        let attn = match mask {
            Some(mask) => {
                let scores = ops::add(&scores, &mask.additive);
                let attn = ops::softmax(&scores, -1);
                ops::mul(&attn, &mask.keep)
            }
            None => ops::softmax(&scores, -1),
        };
        let out = ops::matmul(&attn, &v);

        // (B, H, T, d_k) → (B, T, d_model)
        let out = ops::transpose(&out, &[0, 2, 1, 3]);
        let out = ops::reshape(&out, &[b, t, self.n_heads * self.d_k]);
        ops::linear(&out, &self.out_w, &self.out_b)
    }
}

/// Pad one zero column on the left of the last dimension.
fn pad_left_zero(x: &Array) -> Array {
    let ndim = x.ndim() as i32;
    let mut zero_shape: Vec<i32> = x.shape();
    // Attention scores are always ≥1-D here; a scalar would be a caller error.
    *zero_shape
        .last_mut()
        .expect("pad_left_zero needs a non-scalar array") = 1;
    let zeros = ops::zeros(&zero_shape);
    ops::cat(&[&zeros, x], ndim - 1)
}

/// Relative shift: out[i][j] = x[i][T-1-i+j].
///
/// Steps (matching `RelPositionMultiHeadAttention.rel_shift` in the model
/// source): pad-left → view (B,H,2T,T) → drop first row →
/// view (B,H,T,2T-1) → narrow last dim to T.
fn rel_shift(x: &Array, t: i32) -> Array {
    let b = x.dim(0);
    let h = x.dim(1);
    let n = x.ndim() as i32;

    // (B, H, T, 2T-1) → (B, H, T, 2T)
    let x = pad_left_zero(x);
    // (B, H, T, 2T) → (B, H, 2T, T)
    let x = ops::reshape(&x, &[b, h, -1, t]);
    // Drop the first row along axis 2: (B, H, 2T-1, T)
    let mut starts = vec![0i32; n as usize];
    let stops: Vec<i32> = (0..n).map(|d| x.dim(d)).collect();
    starts[2] = 1;
    let x = ops::slice(&x, &starts, &stops);
    // (B, H, 2T-1, T) → (B, H, T, 2T-1)
    let pos_len = 2 * t - 1;
    let x = ops::reshape(&x, &[b, h, t, pos_len]);
    // Narrow the last dimension to T: (B, H, T, T)
    let starts = vec![0i32; n as usize];
    let mut stops: Vec<i32> = (0..n).map(|d| x.dim(d)).collect();
    stops[3] = t;
    ops::slice(&x, &starts, &stops)
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------
struct ConformerLayer {
    norm_ff1: (Array, Array),
    ff1: FeedForward,
    norm_self_att: (Array, Array),
    self_attn: RelPosAttn,
    norm_conv: (Array, Array),
    conv: ConformerConv,
    norm_ff2: (Array, Array),
    ff2: FeedForward,
    norm_out: (Array, Array),
}

impl ConformerLayer {
    fn load(weights: &MlxWeights, prefix: &str, n_heads: i32, d_model: i32) -> Result<Self> {
        let norm = |n: &str| -> Result<(Array, Array)> {
            let key = format!("{prefix}{n}");
            let w = weights.get(&format!("{key}.weight"))?.shallow_clone();
            let b = weights.get(&format!("{key}.bias"))?.shallow_clone();
            Ok((w, b))
        };
        Ok(Self {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: FeedForward::load(weights, &format!("{prefix}feed_forward1."))?,
            norm_self_att: norm("norm_self_att")?,
            self_attn: RelPosAttn::load(weights, &format!("{prefix}self_attn."), n_heads, d_model)?,
            norm_conv: norm("norm_conv")?,
            conv: ConformerConv::load(weights, &format!("{prefix}conv."), d_model)?,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: FeedForward::load(weights, &format!("{prefix}feed_forward2."))?,
            norm_out: norm("norm_out")?,
        })
    }

    fn forward(
        &self,
        x: &Array,
        pos_emb: &Array,
        valid: i32,
        mask: Option<&SequenceMask>,
    ) -> Array {
        let (nw1, nb1) = &self.norm_ff1;
        let ff1_out = self.ff1.forward(&ops::layer_norm(x, nw1, nb1, 1e-5));
        let x = ops::add(x, &ops::scale(&ff1_out, 0.5));

        let (nw2, nb2) = &self.norm_self_att;
        let attn_out = self
            .self_attn
            .forward(&ops::layer_norm(&x, nw2, nb2, 1e-5), pos_emb, mask);
        let x = ops::add(&x, &attn_out);

        let (nw3, nb3) = &self.norm_conv;
        let conv_out = self
            .conv
            .forward(&ops::layer_norm(&x, nw3, nb3, 1e-5), valid);
        let x = ops::add(&x, &conv_out);

        let (nw4, nb4) = &self.norm_ff2;
        let ff2_out = self.ff2.forward(&ops::layer_norm(&x, nw4, nb4, 1e-5));
        let x = ops::add(&x, &ops::scale(&ff2_out, 0.5));

        let (nw5, nb5) = &self.norm_out;
        ops::layer_norm(&x, nw5, nb5, 1e-5)
    }
}

// ---------------------------------------------------------------------------
// ConformerEncoder (public)
// ---------------------------------------------------------------------------
pub struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    layers: Vec<ConformerLayer>,
    enc_dec_proj_w: Option<Array>,
    enc_dec_proj_b: Option<Array>,
    d_model: i32,
}

impl ConformerEncoder {
    pub fn load(weights: &MlxWeights, cfg: &ModelConfig) -> Result<Self> {
        let enc = &cfg.encoder;
        let d_model = enc.d_model as i32;
        let n_heads = enc.n_heads as i32;

        let pre_encode = ConvSubsampling::load(weights, "encoder.pre_encode.")?;

        let mut layers = Vec::with_capacity(enc.n_layers);
        for i in 0..enc.n_layers {
            let prefix = format!("encoder.layers.{i}.");
            let layer = ConformerLayer::load(weights, &prefix, n_heads, d_model)
                .with_context(|| format!("Loading ConformerLayer {i}"))?;
            layers.push(layer);
        }

        // All-or-nothing: a checkpoint with only one of weight/bias would
        // otherwise silently skip the 1280→1024 projection and fail much
        // later with an opaque matmul shape error.
        let (enc_dec_proj_w, enc_dec_proj_b) = match (
            weights.get("encoder_decoder_proj.weight"),
            weights.get("encoder_decoder_proj.bias"),
        ) {
            (Ok(w), Ok(b)) => (Some(w.shallow_clone()), Some(b.shallow_clone())),
            (Err(_), Err(_)) => (None, None),
            (w, _) => {
                let missing = if w.is_err() { "weight" } else { "bias" };
                anyhow::bail!(
                    "encoder_decoder_proj is half-present ({missing} missing) — refusing to guess"
                );
            }
        };

        Ok(Self {
            pre_encode,
            layers,
            enc_dec_proj_w,
            enc_dec_proj_b,
            d_model,
        })
    }

    /// x: (1, n_mels, T), `valid_frames` real frames → (hidden states, valid encoder steps).
    pub fn forward(&self, x: &Array, valid_frames: i32) -> (Array, i32) {
        let (x, t_prime, valid) = self.pre_encode.forward(x, valid_frames);
        if t_prime <= 0 || valid <= 0 {
            return (x, 0);
        }
        let pos_emb = rel_positional_encoding(t_prime as usize, self.d_model as usize);
        let mask = sequence_mask(t_prime, valid);

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb, valid, mask.as_ref());
        }

        if let (Some(w), Some(b)) = (&self.enc_dec_proj_w, &self.enc_dec_proj_b) {
            x = ops::linear(&x, w, b);
        }

        (x, valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::ffi;

    /// Read element (i, j, k, l) of a 4-D array as a scalar f32.
    fn item4(a: &Array, i: i32, j: i32, k: i32, l: i32) -> f32 {
        let starts = [i, j, k, l];
        let stops = [i + 1, j + 1, k + 1, l + 1];
        let cell = ops::slice(a, &starts, &stops);
        let mut val = 0.0f32;
        let st = unsafe { ffi::mlx_array_item_float32(&mut val, cell.ptr) };
        assert_eq!(st, 0, "mlx_array_item_float32 failed");
        val
    }

    /// rel_shift must satisfy out[i][j] == x[i][T-1-i+j] (the reference
    /// rel_shift + `[:, :, :, :T]` narrow from modeling_cohere_asr.py).
    /// Uses the CPU device so the test needs no Metal GPU, only the linked
    /// mlx-c library.
    #[test]
    fn rel_shift_matches_reference_indexing() {
        let _guard = crate::mlx::stream::test_lock();
        crate::mlx::stream::init_mlx(false);
        for t in [2i32, 3, 4, 7] {
            let pos_len = 2 * t - 1;
            let data: Vec<f32> = (0..(t * pos_len)).map(|v| v as f32).collect();
            let x = Array::from_data_f32(&data, &[1, 1, t, pos_len]);
            let out = rel_shift(&x, t);
            assert_eq!(out.shape(), vec![1, 1, t, t]);
            for i in 0..t {
                for j in 0..t {
                    let got = item4(&out, 0, 0, i, j);
                    let want = data[(i * pos_len + (t - 1 - i + j)) as usize];
                    assert_eq!(got, want, "t={t} out[{i}][{j}]");
                }
            }
        }
    }

    #[test]
    fn subsampling_length_drops_the_stft_pad_frame() {
        // Three stride-2, kernel-3, pad-1 convs. A valid length that is a
        // multiple of 8 grows an extra encoder step when the pad frame is
        // included, which is the case the mask exists for.
        let mut full = 9i32;
        let mut valid = 8i32;
        for _ in 0..3 {
            full = conv_out_len(full, 3, 2, 1);
            valid = conv_out_len(valid, 3, 2, 1);
        }
        assert_eq!(valid, 1);
        assert_eq!(full, 2);
        assert!(valid < full);
    }

    #[test]
    fn pad_mask_blocks_padding_queries_and_keys() {
        let (additive, keep) = pad_mask_values(3, 2);
        // Real/real stays open.
        assert_eq!(additive[0], 0.0);
        assert_eq!(keep[0], 1.0);
        // Key 2 is padding for every query, including real query 0.
        assert_eq!(additive[2], -1.0e9);
        assert_eq!(keep[2], 0.0);
        // Query 2 is padding for every key.
        assert_eq!(additive[2 * 3], -1.0e9);
        assert_eq!(keep[2 * 3 + 1], 0.0);
    }
}
