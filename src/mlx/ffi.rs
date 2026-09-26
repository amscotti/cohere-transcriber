//! Raw FFI declarations for the mlx-c library.
//!
//! These declarations match the mlx-c headers in the vendored `mlx-c/` tree.
//!
//! All op functions follow the same pattern:
//!   - Output is written through a `*mut mlx_array` pointer (first arg)
//!   - The last arg is always the `mlx_stream` to execute on
//!   - Most functions return `c_int` (0 on success)
//!   - Caller owns the returned array and must free it with `mlx_array_free`
//!
//! Opaque types: the C headers define types like `struct mlx_array_ { void* ctx; }`
//! which are single-pointer-sized structs passed by value. We represent them as
//! `*mut c_void` which is ABI-compatible on all standard calling conventions.
//!
//! Note: `#[link(name = "mlxc")]` is NOT used here because `build.rs` already
//! handles linking via `cargo:rustc-link-lib=static=mlxc`.
//!
//! Safety: all functions are unsafe. Use the safe wrappers in `ops.rs`.

#![allow(non_camel_case_types)]
// NOTE: no `allow(dead_code)` here — every declaration below must have a
// caller in `array.rs`, `ops.rs`, or `stream.rs`. The build is warning-free,
// so deleting an op wrapper without its declaration (or vice versa) fails
// loudly instead of rotting silently.

use std::os::raw::{c_char, c_float, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

pub type mlx_array = *mut c_void;
pub type mlx_stream = *mut c_void;
pub type mlx_device = *mut c_void;
pub type mlx_vector_array = *mut c_void;

/// Error callback: receives a human-readable message. Must not unwind into
/// mlx-c — store the message and return normally; the failing call then
/// reports a non-zero status which `ops::ck` turns into a Rust panic.
pub type mlx_error_handler_func =
    Option<unsafe extern "C" fn(msg: *const c_char, data: *mut c_void)>;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

// Only the dtypes this binary ever passes to mlx-c. Discriminants must
// match `mlx_dtype` in mlx-c/mlx/c/array.h.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mlx_dtype {
    MLX_INT32 = 7,
    MLX_FLOAT32 = 10,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mlx_device_type {
    MLX_CPU = 0,
    MLX_GPU = 1,
}

extern "C" {
    // -----------------------------------------------------------------------
    // Array lifecycle
    // -----------------------------------------------------------------------

    pub fn mlx_array_new() -> mlx_array;
    pub fn mlx_array_free(arr: mlx_array) -> c_int;
    pub fn mlx_array_set(arr: *mut mlx_array, src: mlx_array) -> c_int;

    // -----------------------------------------------------------------------
    // Array creation
    // -----------------------------------------------------------------------

    pub fn mlx_array_new_data(
        data: *const c_void,
        shape: *const c_int,
        ndim: c_int,
        dtype: mlx_dtype,
    ) -> mlx_array;

    pub fn mlx_array_new_float(val: c_float) -> mlx_array;

    // -----------------------------------------------------------------------
    // Array metadata
    // -----------------------------------------------------------------------

    pub fn mlx_array_ndim(arr: mlx_array) -> usize;
    pub fn mlx_array_shape(arr: mlx_array) -> *const c_int;

    // -----------------------------------------------------------------------
    // Array evaluation and data access
    // -----------------------------------------------------------------------

    pub fn mlx_array_eval(arr: mlx_array) -> c_int;
    pub fn mlx_array_item_uint32(res: *mut u32, arr: mlx_array) -> c_int;
    /// Test-only scalar reader used by encoder unit tests.
    #[cfg(test)]
    pub fn mlx_array_item_float32(res: *mut c_float, arr: mlx_array) -> c_int;

    // -----------------------------------------------------------------------
    // Error handling (mlx/c/error.h)
    // -----------------------------------------------------------------------

    /// Install a handler invoked when an mlx-c call fails. The default
    /// handler prints to stdout and calls `exit(-1)`, so installing our own
    /// is what makes the status codes (and `ops::ck`) meaningful at all.
    pub fn mlx_set_error_handler(
        handler: mlx_error_handler_func,
        data: *mut c_void,
        dtor: Option<unsafe extern "C" fn(data: *mut c_void)>,
    );

    // -----------------------------------------------------------------------
    // Memory management (mlx/c/memory.h)
    // -----------------------------------------------------------------------

    /// Free cached (no-longer-used) Metal buffers. Bounded workloads with
    /// growing allocations (e.g. per-step KV-cache concatenation) otherwise
    /// keep every intermediate size alive in the allocator cache.
    pub fn mlx_clear_cache() -> c_int;

    // -----------------------------------------------------------------------
    // Device
    // -----------------------------------------------------------------------

    pub fn mlx_device_new_type(dtype: mlx_device_type, index: c_int) -> mlx_device;
    pub fn mlx_device_free(dev: mlx_device) -> c_int;
    pub fn mlx_set_default_device(dev: mlx_device) -> c_int;

    // -----------------------------------------------------------------------
    // Stream
    // -----------------------------------------------------------------------

    pub fn mlx_get_default_stream(res: *mut mlx_stream, dev: mlx_device) -> c_int;
    pub fn mlx_synchronize(stream: mlx_stream) -> c_int;

    // -----------------------------------------------------------------------
    // Core ops
    // -----------------------------------------------------------------------

    // Arithmetic
    pub fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_subtract(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_multiply(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;

    // Matrix multiplication
    pub fn mlx_matmul(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;

    // Shape manipulation
    pub fn mlx_reshape(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const c_int,
        shape_num: usize,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_transpose_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        axes_num: usize,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_expand_dims(res: *mut mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) -> c_int;

    pub fn mlx_expand_dims_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        axes_num: usize,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_squeeze_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        axes_num: usize,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_slice(
        res: *mut mlx_array,
        a: mlx_array,
        start: *const c_int,
        start_num: usize,
        stop: *const c_int,
        stop_num: usize,
        strides: *const c_int,
        strides_num: usize,
        s: mlx_stream,
    ) -> c_int;

    // Concatenation
    pub fn mlx_concatenate_axis(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Creation ops
    pub fn mlx_zeros(
        res: *mut mlx_array,
        shape: *const c_int,
        shape_num: usize,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;

    // Indexing
    pub fn mlx_take_axis(
        res: *mut mlx_array,
        a: mlx_array,
        indices: mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Reduction
    pub fn mlx_argmax_axis(
        res: *mut mlx_array,
        a: mlx_array,
        axis: c_int,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    // Math functions
    pub fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_maximum(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_rsqrt(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;

    // Activation-related
    pub fn mlx_softmax_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        axes_num: usize,
        precise: bool,
        s: mlx_stream,
    ) -> c_int;

    // Convolution
    pub fn mlx_conv1d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_conv2d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride_0: c_int,
        stride_1: c_int,
        padding_0: c_int,
        padding_1: c_int,
        dilation_0: c_int,
        dilation_1: c_int,
        groups: c_int,
        s: mlx_stream,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Fast ML ops
    // -----------------------------------------------------------------------

    pub fn mlx_fast_layer_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        bias: mlx_array,
        eps: c_float,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_fast_scaled_dot_product_attention(
        res: *mut mlx_array,
        queries: mlx_array,
        keys: mlx_array,
        values: mlx_array,
        scale: c_float,
        mask_mode: *const c_char,
        mask_arr: mlx_array,
        sinks: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Vector
    // -----------------------------------------------------------------------

    pub fn mlx_vector_array_new() -> mlx_vector_array;
    pub fn mlx_vector_array_free(vec: mlx_vector_array) -> c_int;
    pub fn mlx_vector_array_append_value(vec: mlx_vector_array, val: mlx_array) -> c_int;
}
