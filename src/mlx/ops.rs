//! Safe wrappers around MLX C FFI operations.
//!
//! Every function follows the same pattern:
//!   1. Create an empty output array with `Array::empty()`
//!   2. Call the FFI function with `&mut res.ptr`
//!   3. Check the returned status code (panics with the op name on failure)
//!   4. Return the result
//!
//! Shapes use `i32`, not `i64` — MLX's C API uses 32-bit dimension sizes.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_int;

use super::array::Array;
use super::ffi::{self, mlx_dtype};
use super::stream::default_stream;

thread_local! {
    /// Message from the most recent mlx-c error, captured by our error
    /// handler. Read (and cleared) by `ck` when a call reports failure.
    static LAST_MLX_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(crate) unsafe extern "C" fn mlx_error_handler(
    msg: *const std::os::raw::c_char,
    _data: *mut std::os::raw::c_void,
) {
    if msg.is_null() {
        return;
    }
    let text = std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned();
    LAST_MLX_ERROR.with(|cell| *cell.borrow_mut() = Some(text));
}

fn take_last_mlx_error() -> Option<String> {
    LAST_MLX_ERROR.with(|cell| cell.borrow_mut().take())
}

/// Crate-visible check for a captured mlx-c error (used by `array`/`stream`).
pub(crate) fn check_status(status: c_int, op: &'static str) {
    ck(status, op);
}

/// Panic with context when an mlx-c call reports failure.
///
/// mlx-c surfaces detail through the error handler installed by
/// [`super::stream`] (the default handler would `exit(-1)` instead); the
/// status code alone carries no message, so the captured text is the
/// actionable part. A loud panic here beats silently propagating a
/// null/partial array that would fail much later (or worse, decode to wrong
/// text).
fn ck(status: c_int, op: &'static str) {
    if status != 0 {
        let detail = take_last_mlx_error()
            .unwrap_or_else(|| "no detail captured (is the mlx-c error handler installed?)".into());
        panic!("mlx-c call failed: {op}: {detail}");
    }
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/// Matrix multiplication: (…, M, K) × (…, K, N) → (…, M, N).
pub fn matmul(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_matmul(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    ck(st, "mlx_matmul");
    res
}

/// Linear layer: x @ w.T + b.
/// x: (B, T, in), w: (out, in), b: (out,) → (B, T, out)
pub fn linear(x: &Array, w: &Array, b: &Array) -> Array {
    let wt = transpose(w, &[1, 0]);
    let y = matmul(x, &wt);
    add(&y, b)
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

pub fn reshape(a: &Array, shape: &[i32]) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_reshape(
            &mut res.ptr,
            a.ptr,
            shape.as_ptr(),
            shape.len(),
            default_stream(),
        )
    };
    ck(st, "mlx_reshape");
    res
}

pub fn transpose(a: &Array, axes: &[i32]) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_transpose_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            axes.len(),
            default_stream(),
        )
    };
    ck(st, "mlx_transpose_axes");
    res
}

/// Swap the last two dimensions — shorthand used everywhere in attention.
pub fn transpose_last2(a: &Array) -> Array {
    let ndim = a.ndim() as i32;
    assert!(ndim >= 2, "transpose_last2 needs ndim >= 2, got {ndim}");
    let axes: Vec<i32> = (0..ndim - 2).chain([ndim - 1, ndim - 2]).collect();
    transpose(a, &axes)
}

pub fn expand_dims(a: &Array, axes: &[i32]) -> Array {
    // The mlx-c API has mlx_expand_dims for a single axis and
    // mlx_expand_dims_axes for multiple axes.
    if axes.len() == 1 {
        let mut res = Array::empty();
        let st = unsafe { ffi::mlx_expand_dims(&mut res.ptr, a.ptr, axes[0], default_stream()) };
        ck(st, "mlx_expand_dims");
        res
    } else {
        let mut res = Array::empty();
        let st = unsafe {
            ffi::mlx_expand_dims_axes(
                &mut res.ptr,
                a.ptr,
                axes.as_ptr(),
                axes.len(),
                default_stream(),
            )
        };
        ck(st, "mlx_expand_dims_axes");
        res
    }
}

pub fn squeeze(a: &Array, axes: &[i32]) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_squeeze_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            axes.len(),
            default_stream(),
        )
    };
    ck(st, "mlx_squeeze_axes");
    res
}

// ---------------------------------------------------------------------------
// Concatenation
// ---------------------------------------------------------------------------

pub fn cat(arrays: &[&Array], axis: i32) -> Array {
    // RAII guard so the mlx_vector_array is freed even if a later `ck`
    // panics (a leaked vector would also leak its array references).
    struct VectorArray(ffi::mlx_vector_array);
    impl Drop for VectorArray {
        fn drop(&mut self) {
            unsafe { ffi::mlx_vector_array_free(self.0) };
        }
    }
    let vec = VectorArray(unsafe { ffi::mlx_vector_array_new() });
    assert!(!vec.0.is_null(), "mlx_vector_array_new returned null");
    for a in arrays {
        ck(
            unsafe { ffi::mlx_vector_array_append_value(vec.0, a.ptr) },
            "mlx_vector_array_append_value",
        );
    }
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_concatenate_axis(&mut res.ptr, vec.0, axis, default_stream()) };
    ck(st, "mlx_concatenate_axis");
    res
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

pub fn add(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_add(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    ck(st, "mlx_add");
    res
}

pub fn sub(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_subtract(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    ck(st, "mlx_subtract");
    res
}

pub fn mul(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_multiply(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    ck(st, "mlx_multiply");
    res
}

pub fn rsqrt(a: &Array) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_rsqrt(&mut res.ptr, a.ptr, default_stream()) };
    ck(st, "mlx_rsqrt");
    res
}

/// Scale by a scalar — creates a temporary scalar Array.
pub fn scale(a: &Array, s: f32) -> Array {
    let scalar = unsafe { Array::from_ptr(ffi::mlx_array_new_float(s)) };
    mul(a, &scalar)
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// ReLU activation: max(0, x).
pub fn relu(a: &Array) -> Array {
    let zero = unsafe { Array::from_ptr(ffi::mlx_array_new_float(0.0)) };
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_maximum(&mut res.ptr, a.ptr, zero.ptr, default_stream()) };
    ck(st, "mlx_maximum");
    res
}

/// SiLU (Swish) activation: x * sigmoid(x).
pub fn silu(a: &Array) -> Array {
    let sig = sigmoid(a);
    mul(a, &sig)
}

/// Sigmoid activation via mlx_sigmoid.
pub fn sigmoid(a: &Array) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_sigmoid(&mut res.ptr, a.ptr, default_stream()) };
    ck(st, "mlx_sigmoid");
    res
}

pub fn softmax(a: &Array, axis: i32) -> Array {
    let axes = [axis];
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_softmax_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            1,
            true, // precise = true for numerical stability
            default_stream(),
        )
    };
    ck(st, "mlx_softmax_axes");
    res
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

pub fn argmax(a: &Array, axis: i32, keepdims: bool) -> Array {
    let mut res = Array::empty();
    let st = unsafe { ffi::mlx_argmax_axis(&mut res.ptr, a.ptr, axis, keepdims, default_stream()) };
    ck(st, "mlx_argmax_axis");
    res
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// Fused layer norm using the fast kernel in mlx.core.fast.
pub fn layer_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_fast_layer_norm(
            &mut res.ptr,
            x.ptr,
            weight.ptr,
            bias.ptr,
            eps,
            default_stream(),
        )
    };
    ck(st, "mlx_fast_layer_norm");
    res
}

/// Scaled dot-product attention using MLX's fused fast kernel.
/// q/k/v: (B, H, T, d_k).  mask: optional additive mask.
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
) -> Array {
    // mlx-c's fused attention only accepts the mask modes "", "causal" and
    // "array"; an additive float mask must be passed as mode "array".
    let mask_mode = if mask.is_some() {
        CString::new("array").unwrap()
    } else {
        CString::new("").unwrap()
    };
    let mask_ptr = mask.map_or(std::ptr::null_mut(), |m| m.ptr);
    let sinks_ptr: ffi::mlx_array = std::ptr::null_mut();
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_fast_scaled_dot_product_attention(
            &mut res.ptr,
            q.ptr,
            k.ptr,
            v.ptr,
            scale,
            mask_mode.as_ptr(),
            mask_ptr,
            sinks_ptr,
            default_stream(),
        )
    };
    ck(st, "mlx_fast_scaled_dot_product_attention");
    res
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// Gather rows from `arr` at integer `indices` along the given axis.
/// Equivalent to arr[indices] — used for token/positional embeddings.
pub fn take(arr: &Array, indices: &Array, axis: i32) -> Array {
    let mut res = Array::empty();
    let st =
        unsafe { ffi::mlx_take_axis(&mut res.ptr, arr.ptr, indices.ptr, axis, default_stream()) };
    ck(st, "mlx_take_axis");
    res
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

pub fn zeros(shape: &[i32]) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_zeros(
            &mut res.ptr,
            shape.as_ptr(),
            shape.len(),
            mlx_dtype::MLX_FLOAT32,
            default_stream(),
        )
    };
    ck(st, "mlx_zeros");
    res
}

// ---------------------------------------------------------------------------
// Slicing (used directly by encoder.rs)
// ---------------------------------------------------------------------------

/// Slice an array along all dimensions with given start/stop indices and
/// unit strides.
pub fn slice(x: &Array, starts: &[i32], stops: &[i32]) -> Array {
    let n = starts.len();
    let strides = vec![1i32; n];
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_slice(
            &mut res.ptr,
            x.ptr,
            starts.as_ptr(),
            n,
            stops.as_ptr(),
            n,
            strides.as_ptr(),
            n,
            default_stream(),
        )
    };
    ck(st, "mlx_slice");
    res
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

/// 2-D convolution.
/// input: (N, H, W, C_in) — MLX channels-last
/// weight: (C_out, kH, kW, C_in/groups) — transposed from PyTorch OIHW
/// Returns (N, H', W', C_out)
pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    groups: i32,
) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_conv2d(
            &mut res.ptr,
            input.ptr,
            weight.ptr,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            1, // dilation_h
            1, // dilation_w
            groups,
            default_stream(),
        )
    };
    ck(st, "mlx_conv2d");
    res
}

/// 1-D convolution.
/// input: (N, L, C_in) — MLX channels-last
/// weight: (C_out, kW, C_in/groups) — transposed from PyTorch OIW
/// Returns (N, L', C_out)
pub fn conv1d(input: &Array, weight: &Array, stride: i32, padding: i32, groups: i32) -> Array {
    let mut res = Array::empty();
    let st = unsafe {
        ffi::mlx_conv1d(
            &mut res.ptr,
            input.ptr,
            weight.ptr,
            stride,
            padding,
            1, // dilation
            groups,
            default_stream(),
        )
    };
    ck(st, "mlx_conv1d");
    res
}
