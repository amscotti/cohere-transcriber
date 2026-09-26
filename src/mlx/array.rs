//! Safe RAII wrapper around `mlx_array`.
//!
//! `Array` owns an MLX array handle and frees it on drop.
//! All heavy computation is lazy — MLX only materialises results when
//! `eval()` is called (or data is read).

use std::ffi::c_void;

use super::ffi::{self, mlx_array, mlx_dtype};

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

/// A reference-counted MLX array. Sharing is explicit via
/// [`Array::shallow_clone`] — `Clone` is deliberately not implemented so
/// copies never happen by accident.
pub struct Array {
    pub(crate) ptr: mlx_array,
}

// Safety: MLX arrays are ref-counted C objects; they are safe to send between
// threads after the stream they were created on has been synchronized. `Sync`
// is deliberately NOT implemented: `eval()` mutates the array's (non-atomic)
// internal status, so shared `&Array` access from several threads would race.
unsafe impl Send for Array {}

impl Array {
    /// Wrap an existing raw pointer (takes ownership).
    pub(crate) unsafe fn from_ptr(ptr: mlx_array) -> Self {
        assert!(!ptr.is_null(), "mlx_array pointer is null");
        Self { ptr }
    }

    /// Create an uninitialised (empty) placeholder — used as the output slot
    /// for FFI calls before they write a real value.
    ///
    /// `mlx_array_new` returns the one-pointer mlx-c struct *by value* with a
    /// null context; at the FFI boundary that is bit-identical to a null
    /// pointer, so a null `ptr` here is the expected representation of "no
    /// array yet", not an error. Every op fills it via `&mut res.ptr`; `Drop`
    /// (`mlx_array_free`) is a no-op on a null context.
    pub(crate) fn empty() -> Self {
        let ptr = unsafe { ffi::mlx_array_new() };
        Self { ptr }
    }

    // -----------------------------------------------------------------------
    // Creation helpers
    // -----------------------------------------------------------------------

    /// Create a float32 array from a flat buffer with an explicit shape.
    pub fn from_data_f32(data: &[f32], shape: &[i32]) -> Self {
        // Compute the product as usize so a huge shape cannot wrap a signed
        // i32 product and slip past the length check.
        let product: usize = shape.iter().map(|&d| d.max(0) as usize).product();
        assert_eq!(
            data.len(),
            product,
            "data length does not match shape product"
        );
        let ptr = unsafe {
            ffi::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len() as i32,
                mlx_dtype::MLX_FLOAT32,
            )
        };
        unsafe { Self::from_ptr(ptr) }
    }

    /// Create a 1-D int32 index array from a Rust slice.
    pub fn from_slice_i32(data: &[i32]) -> Self {
        let shape = [data.len() as i32];
        let ptr = unsafe {
            ffi::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                1,
                mlx_dtype::MLX_INT32,
            )
        };
        unsafe { Self::from_ptr(ptr) }
    }

    // -----------------------------------------------------------------------
    // Shape / metadata
    // -----------------------------------------------------------------------

    pub fn ndim(&self) -> usize {
        unsafe { ffi::mlx_array_ndim(self.ptr) }
    }

    pub fn dim(&self, axis: i32) -> i32 {
        let n = self.ndim();
        let shape_ptr = unsafe { ffi::mlx_array_shape(self.ptr) };
        assert!(!shape_ptr.is_null(), "mlx_array_shape returned null");
        let idx = if axis >= 0 {
            axis as usize
        } else {
            (n as i32 + axis) as usize
        };
        assert!(idx < n, "axis {} out of range for ndim {}", axis, n);
        unsafe { *shape_ptr.add(idx) }
    }

    pub fn shape(&self) -> Vec<i32> {
        let n = self.ndim();
        let shape_ptr = unsafe { ffi::mlx_array_shape(self.ptr) };
        if shape_ptr.is_null() || n == 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(shape_ptr, n).to_vec() }
    }

    // -----------------------------------------------------------------------
    // Evaluation and data access
    // -----------------------------------------------------------------------

    /// Force materialisation of any pending lazy computation.
    pub fn eval(&self) {
        let st = unsafe { ffi::mlx_array_eval(self.ptr) };
        super::ops::check_status(st, "mlx_array_eval");
    }

    /// Read the index of the maximum element (argmax over flattened array).
    /// Convenience for greedy decoding.
    pub fn argmax_flat(&self) -> i64 {
        let am = super::ops::argmax(self, -1, false);
        // MLX's argmax produces a uint32 result — read it through the typed
        // accessor instead of type-punning via the int32 data pointer, and
        // check the status so an unevaluatable array is a loud error.
        let mut out: u32 = 0;
        let st = unsafe { ffi::mlx_array_item_uint32(&mut out, am.ptr) };
        assert_eq!(
            st, 0,
            "mlx_array_item_uint32 failed (argmax not evaluated?)"
        );
        out as i64
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { ffi::mlx_array_free(self.ptr) };
    }
}

// Arrays cannot be cheaply cloned without incrementing the ref count via the
// C API.  Provide an explicit method instead of implementing Clone to avoid
// accidental copies.
impl Array {
    /// Shallow copy — uses `mlx_array_set` to share the underlying storage
    /// with reference counting (O(1)). Assigning into the empty placeholder
    /// allocates the handle; nothing leaks.
    pub fn shallow_clone(&self) -> Self {
        let mut new = Self::empty();
        let st = unsafe { ffi::mlx_array_set(&mut new.ptr, self.ptr) };
        assert_eq!(st, 0, "mlx_array_set failed");
        new
    }
}
