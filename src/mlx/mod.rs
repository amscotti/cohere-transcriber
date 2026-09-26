//! MLX backend — macOS/Apple Silicon only.
//!
//! Requires the `mlx` cargo feature (on by default), the vendored `mlx-c/`
//! tree, and the Metal frameworks (see build.rs and README.md).

pub mod array;
pub mod decoder;
pub mod encoder;
pub mod ffi;
pub mod inference;
pub mod ops;
pub mod stream;
pub mod weights;
