//! Transformer decoder (FastConformer transducer head) on MLX arrays.
//!
//! Self-attention K/V cache grows by one token per decoding step.
//! Cross-attention K/V are pre-computed once from encoder output and reused.

use anyhow::Result;

use super::array::Array;
use super::ops;
use super::weights::MlxWeights;
use crate::config::ModelConfig;

// ---------------------------------------------------------------------------
// Decoder attention block
// ---------------------------------------------------------------------------
struct DecoderAttn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    out_w: Array,
    out_b: Array,
    n_heads: i32,
    head_dim: i32,
    hidden: i32,
}

impl DecoderAttn {
    fn load(weights: &MlxWeights, prefix: &str, n_heads: i32, hidden: i32) -> Result<Self> {
        let head_dim = hidden / n_heads;
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{prefix}{n}"))?.shallow_clone())
        };
        Ok(Self {
            q_w: get("query_net.weight")?,
            q_b: get("query_net.bias")?,
            k_w: get("key_net.weight")?,
            k_b: get("key_net.bias")?,
            v_w: get("value_net.weight")?,
            v_b: get("value_net.bias")?,
            out_w: get("out_projection.weight")?,
            out_b: get("out_projection.bias")?,
            n_heads,
            head_dim,
            hidden,
        })
    }

    /// Project Q, K, V from (B, T, hidden) → (B, H, T, head_dim).
    fn project_q(&self, hidden_states: &Array) -> Array {
        let b = hidden_states.dim(0);
        let t = hidden_states.dim(1);
        let q = ops::linear(hidden_states, &self.q_w, &self.q_b);
        let q = ops::reshape(&q, &[b, t, self.n_heads, self.head_dim]);
        ops::transpose(&q, &[0, 2, 1, 3])
    }

    fn project_kv(&self, source: &Array) -> (Array, Array) {
        let b = source.dim(0);
        let s = source.dim(1);
        let k = ops::linear(source, &self.k_w, &self.k_b);
        let k = ops::reshape(&k, &[b, s, self.n_heads, self.head_dim]);
        let k = ops::transpose(&k, &[0, 2, 1, 3]);
        let v = ops::linear(source, &self.v_w, &self.v_b);
        let v = ops::reshape(&v, &[b, s, self.n_heads, self.head_dim]);
        let v = ops::transpose(&v, &[0, 2, 1, 3]);
        (k, v)
    }

    /// Full attention: Q from hidden_states, K/V from source (may differ).
    /// Returns (B, T, hidden).
    fn forward(&self, hidden_states: &Array, k: &Array, v: &Array, mask: Option<&Array>) -> Array {
        let b = hidden_states.dim(0);
        let t = hidden_states.dim(1);
        let q = self.project_q(hidden_states);
        let scale = (self.head_dim as f32).powf(-0.5);
        let out = ops::scaled_dot_product_attention(&q, k, v, scale, mask);
        // (B, H, T, head_dim) → (B, T, hidden)
        let out = ops::transpose(&out, &[0, 2, 1, 3]);
        let out = ops::reshape(&out, &[b, t, self.hidden]);
        ops::linear(&out, &self.out_w, &self.out_b)
    }
}

// ---------------------------------------------------------------------------
// Decoder FFN  (dense_in → ReLU → dense_out)
// ---------------------------------------------------------------------------
struct DecoderFFN {
    dense_in_w: Array,
    dense_in_b: Array,
    dense_out_w: Array,
    dense_out_b: Array,
}

impl DecoderFFN {
    fn load(weights: &MlxWeights, prefix: &str) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{prefix}{n}"))?.shallow_clone())
        };
        Ok(Self {
            dense_in_w: get("dense_in.weight")?,
            dense_in_b: get("dense_in.bias")?,
            dense_out_w: get("dense_out.weight")?,
            dense_out_b: get("dense_out.bias")?,
        })
    }

    fn forward(&self, x: &Array) -> Array {
        let h = ops::relu(&ops::linear(x, &self.dense_in_w, &self.dense_in_b));
        ops::linear(&h, &self.dense_out_w, &self.dense_out_b)
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoderLayer
// ---------------------------------------------------------------------------
pub struct DecoderLayer {
    norm1: (Array, Array),
    self_attn: DecoderAttn,
    norm2: (Array, Array),
    cross_attn: DecoderAttn,
    norm3: (Array, Array),
    ffn: DecoderFFN,
}

impl DecoderLayer {
    fn load(weights: &MlxWeights, prefix: &str, n_heads: i32, hidden: i32) -> Result<Self> {
        let norm = |n: &str| -> Result<(Array, Array)> {
            let key = format!("{prefix}{n}");
            let w = weights.get(&format!("{key}.weight"))?.shallow_clone();
            let b = weights.get(&format!("{key}.bias"))?.shallow_clone();
            Ok((w, b))
        };
        Ok(Self {
            norm1: norm("layer_norm_1")?,
            self_attn: DecoderAttn::load(
                weights,
                &format!("{prefix}first_sub_layer."),
                n_heads,
                hidden,
            )?,
            norm2: norm("layer_norm_2")?,
            cross_attn: DecoderAttn::load(
                weights,
                &format!("{prefix}second_sub_layer."),
                n_heads,
                hidden,
            )?,
            norm3: norm("layer_norm_3")?,
            ffn: DecoderFFN::load(weights, &format!("{prefix}third_sub_layer."))?,
        })
    }

    /// Single-token forward with KV cache.
    ///
    /// hidden: (1, 1, hidden)
    /// self_k_cache / self_v_cache: Option<(1, H, T_prev, head_dim)>
    /// cross_k / cross_v: (1, H, T_enc, head_dim) — pre-computed once
    ///
    /// Returns (out, new_self_k, new_self_v).
    pub fn forward_cached(
        &self,
        hidden: &Array,
        self_k_cache: Option<&Array>,
        self_v_cache: Option<&Array>,
        cross_k: &Array,
        cross_v: &Array,
        cross_mask: Option<&Array>,
    ) -> (Array, Array, Array) {
        // --- Self-attention ---
        let (nw1, nb1) = &self.norm1;
        let normed = ops::layer_norm(hidden, nw1, nb1, 1e-5);
        let (k_new, v_new) = self.self_attn.project_kv(&normed);

        // The cache always holds K and V together (see `step`); a split
        // cache would silently attend over the wrong prefix, so fail loudly.
        let (k_full, v_full) = match (self_k_cache, self_v_cache) {
            (Some(kc), Some(vc)) => (
                ops::cat(&[kc, &k_new], 2), // concat along T dim
                ops::cat(&[vc, &v_new], 2),
            ),
            (None, None) => (k_new, v_new),
            _ => panic!("decoder KV cache holds K without V (or vice versa)"),
        };

        let self_out = self.self_attn.forward(&normed, &k_full, &v_full, None);
        let hidden = ops::add(hidden, &self_out);

        // --- Cross-attention ---
        let (nw2, nb2) = &self.norm2;
        let normed2 = ops::layer_norm(&hidden, nw2, nb2, 1e-5);
        let cross_out = self
            .cross_attn
            .forward(&normed2, cross_k, cross_v, cross_mask);
        let hidden = ops::add(&hidden, &cross_out);

        // --- FFN ---
        let (nw3, nb3) = &self.norm3;
        let normed3 = ops::layer_norm(&hidden, nw3, nb3, 1e-5);
        let ffn_out = self.ffn.forward(&normed3);
        let hidden = ops::add(&hidden, &ffn_out);

        (hidden, k_full, v_full)
    }
}

// ---------------------------------------------------------------------------
// Fixed Positional Encoding (stored in model weights)
// ---------------------------------------------------------------------------
struct FixedPosEnc {
    pos_enc: Array, // (max_seq_len, d_model)
}

impl FixedPosEnc {
    fn load(weights: &MlxWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            pos_enc: weights
                .get(&format!("{prefix}position_embedding.pos_enc"))?
                .shallow_clone(),
        })
    }

    /// Gather rows at given position indices.
    fn forward(&self, position_ids: &[i32]) -> Array {
        let indices = Array::from_slice_i32(position_ids);
        ops::take(&self.pos_enc, &indices, 0)
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoder (public)
// ---------------------------------------------------------------------------
/// One layer's self-attention `(K, V)` cache, `None` before the first token.
pub type DecoderKvLayer = (Option<Array>, Option<Array>);
/// Per-layer self-attention KV cache: one entry per decoder layer.
pub type DecoderKvCache = Vec<DecoderKvLayer>;

pub struct TransformerDecoder {
    token_emb: Array, // (vocab, hidden)
    pos_enc: FixedPosEnc,
    emb_norm_w: Array,
    emb_norm_b: Array,
    pub layers: Vec<DecoderLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
    head_w: Array,
    head_b: Array,
    hidden: i32,
}

impl TransformerDecoder {
    pub fn load(weights: &MlxWeights, cfg: &ModelConfig) -> Result<Self> {
        let dec = &cfg.transf_decoder.config_dict;
        let hidden = dec.hidden_size as i32;
        let n_heads = dec.num_attention_heads as i32;

        let emb_prefix = "transf_decoder._embedding.";
        let dec_prefix = "transf_decoder._decoder.";

        let token_emb = weights
            .get(&format!("{emb_prefix}token_embedding.weight"))?
            .shallow_clone();
        let pos_enc = FixedPosEnc::load(weights, emb_prefix)?;
        let emb_norm_w = weights
            .get(&format!("{emb_prefix}layer_norm.weight"))?
            .shallow_clone();
        let emb_norm_b = weights
            .get(&format!("{emb_prefix}layer_norm.bias"))?
            .shallow_clone();

        let mut layers = Vec::with_capacity(dec.num_layers);
        for i in 0..dec.num_layers {
            let prefix = format!("{dec_prefix}layers.{i}.");
            layers.push(DecoderLayer::load(weights, &prefix, n_heads, hidden)?);
        }

        let final_ln_w = weights
            .get(&format!("{dec_prefix}final_layer_norm.weight"))?
            .shallow_clone();
        let final_ln_b = weights
            .get(&format!("{dec_prefix}final_layer_norm.bias"))?
            .shallow_clone();
        let head_w = weights
            .get("log_softmax.mlp.layer0.weight")?
            .shallow_clone();
        let head_b = weights.get("log_softmax.mlp.layer0.bias")?.shallow_clone();

        Ok(Self {
            token_emb,
            pos_enc,
            emb_norm_w,
            emb_norm_b,
            layers,
            final_ln_w,
            final_ln_b,
            head_w,
            head_b,
            hidden,
        })
    }

    /// Number of rows in the fixed positional-encoding table — the largest
    /// absolute position (prompt + generated tokens) that can be embedded.
    pub fn max_positions(&self) -> i32 {
        self.pos_enc.pos_enc.dim(0)
    }

    /// Additive key mask of shape `(1, 1, 1, time)`: `0` on real encoder
    /// steps, `-1e9` on padding. Broadcasts over the decoder's query length.
    pub fn key_padding_mask(time: i32, valid: i32) -> Array {
        let n = time.max(0) as usize;
        let valid = valid.clamp(0, time) as usize;
        let mut scores = vec![0f32; n];
        for score in scores.iter_mut().skip(valid) {
            *score = -1.0e9;
        }
        Array::from_data_f32(&scores, &[1, 1, 1, time.max(0)])
    }

    /// Pre-compute cross-attention K/V for all decoder layers from encoder output.
    pub fn precompute_cross_kv(&self, encoder_hs: &Array) -> Vec<(Array, Array)> {
        self.layers
            .iter()
            .map(|layer| layer.cross_attn.project_kv(encoder_hs))
            .collect()
    }

    /// One greedy-decoding step.
    ///
    /// Returns (next_token_id, updated self_kv_cache).
    /// Argmax is computed on GPU — only a single i32 is transferred to CPU
    /// instead of the full-vocabulary logits vector per step.
    pub fn step(
        &self,
        token_id: i32,
        position: i32,
        self_kv_cache: &[DecoderKvLayer],
        cross_kv: &[(Array, Array)],
        cross_mask: Option<&Array>,
    ) -> (i32, DecoderKvCache) {
        // Token embedding lookup
        let idx = Array::from_slice_i32(&[token_id]);
        let emb = ops::take(&self.token_emb, &idx, 0); // (1, hidden)
        let emb = ops::reshape(&emb, &[1, 1, self.hidden]);

        // Positional encoding
        let pe = self.pos_enc.forward(&[position]); // (1, hidden)
        let pe = ops::reshape(&pe, &[1, 1, self.hidden]);

        let x = ops::add(&emb, &pe);
        let x = ops::layer_norm(&x, &self.emb_norm_w, &self.emb_norm_b, 1e-5);

        let mut new_kv: DecoderKvCache = Vec::with_capacity(self.layers.len());
        let mut hidden = x;

        for (i, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = &self_kv_cache[i];
            let (ck, cv) = &cross_kv[i];

            let (new_hidden, k_full, v_full) =
                layer.forward_cached(&hidden, kc.as_ref(), vc.as_ref(), ck, cv, cross_mask);
            hidden = new_hidden;
            new_kv.push((Some(k_full), Some(v_full)));
        }

        // Final layer norm + classification head
        let hidden = ops::layer_norm(&hidden, &self.final_ln_w, &self.final_ln_b, 1e-5);
        // Squeeze T dim: (1, 1, hidden) → (1, hidden)
        let hidden = ops::squeeze(&hidden, &[1]);
        let logits = ops::linear(&hidden, &self.head_w, &self.head_b); // (1, vocab)
        let logits = ops::squeeze(&logits, &[0]); // (vocab,)

        // Argmax on GPU — transfers a single i32 instead of 16,384 floats
        let next_token = logits.argmax_flat() as i32;

        (next_token, new_kv)
    }
}
