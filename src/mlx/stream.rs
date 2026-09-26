//! MLX stream and device initialisation.
//!
//! MLX uses an execution stream model.  All array operations are enqueued on
//! a stream (CPU or GPU).  We keep a single global GPU stream for the lifetime
//! of the process, initialised once via `init_mlx`.

use std::sync::OnceLock;

use super::ffi::{self, mlx_device_type, mlx_stream};

struct MlxContext {
    stream: mlx_stream,
}

// Safety: the stream handle is an owned mlx-c object used from any thread
// only after init has fully published it via OnceLock.
unsafe impl Send for MlxContext {}
unsafe impl Sync for MlxContext {}

static CONTEXT: OnceLock<MlxContext> = OnceLock::new();

/// Initialise the MLX runtime.  Must be called once before any array ops.
///
/// `use_gpu = true` selects the Metal GPU (default).
/// `use_gpu = false` forces CPU execution (useful for debugging).
pub fn init_mlx(use_gpu: bool) {
    // Install the error handler before anything can fail: weights loading
    // (`mlx_array_new_data`) and device creation both run before the first
    // `default_stream()` call, and without this they would hit mlx-c's
    // default handler, which `exit(-1)`s instead of reporting.
    install_error_handler();
    CONTEXT.get_or_init(|| {
        let device_type = if use_gpu {
            mlx_device_type::MLX_GPU
        } else {
            mlx_device_type::MLX_CPU
        };

        let device = unsafe { ffi::mlx_device_new_type(device_type, 0) };
        assert!(
            !device.is_null(),
            "mlx_device_new_type returned null (Metal initialisation failed?)"
        );
        assert_eq!(
            unsafe { ffi::mlx_set_default_device(device) },
            0,
            "mlx_set_default_device failed"
        );

        let mut stream: mlx_stream = std::ptr::null_mut();
        assert_eq!(
            unsafe { ffi::mlx_get_default_stream(&mut stream, device) },
            0,
            "mlx_get_default_stream failed"
        );
        assert!(
            !stream.is_null(),
            "mlx_get_default_stream returned null stream"
        );

        // Both calls above copy the (small, value-type) device into MLX-owned
        // state, so our handle is redundant past this point — free it rather
        // than storing it unread for the life of the process.
        assert_eq!(
            unsafe { ffi::mlx_device_free(device) },
            0,
            "mlx_device_free failed"
        );

        MlxContext { stream }
    });
}

/// Return the global default stream.  Panics if `init_mlx` has not been called.
pub fn default_stream() -> mlx_stream {
    install_error_handler();
    CONTEXT
        .get()
        .expect("MLX not initialised — call init_mlx() before using arrays")
        .stream
}

/// Block until all pending operations on the default stream have completed.
pub fn synchronize() {
    if let Some(ctx) = CONTEXT.get() {
        let st = unsafe { ffi::mlx_synchronize(ctx.stream) };
        super::ops::check_status(st, "mlx_synchronize");
    }
}

/// Free MLX's cached (already unused) Metal buffers.
///
/// Workloads that grow allocations step by step — the decoder's per-token KV
/// cache concatenation, above all — leave every intermediate buffer size in
/// MLX's allocator cache. On machines with lots of RAM the default cache
/// limit is correspondingly large, so those buffers are never recycled.
/// Calling this between audio chunks bounds that growth.
pub fn clear_cache() {
    let st = unsafe { ffi::mlx_clear_cache() };
    if st != 0 {
        tracing::debug!("mlx_clear_cache reported status {st}");
    }
}

/// Hold this for the whole body of any test that calls into MLX.
/// `init_mlx` applies once per process, and the runtime is not safe to use
/// from two tests at the same time.
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

static INSTALL_ERROR_HANDLER: std::sync::Once = std::sync::Once::new();

/// Replace mlx-c's default error handler (printf + `exit(-1)`) with one that
/// stores the message for [`super::ops`] to report. Must run before the
/// first op call; `default_stream()` is on every op path, so hooking it here
/// covers everything.
fn install_error_handler() {
    INSTALL_ERROR_HANDLER.call_once(|| unsafe {
        ffi::mlx_set_error_handler(
            Some(super::ops::mlx_error_handler),
            std::ptr::null_mut(),
            None,
        )
    });
}
