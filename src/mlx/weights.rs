//! MLX weight loader — reads SafeTensors and materialises weights as MLX arrays.
//!
//! Floating-point tensors are decoded to F32 and uploaded to the Metal device
//! via `Array::from_data_f32`. The file is read tensor-by-tensor (header
//! first, then each tensor's byte range) so the raw bytes of the ~4 GB
//! checkpoint are never held in memory alongside the F32 weights — the peak
//! is one tensor's temporaries plus the weights themselves, not
//! `file + weights`.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::array::Array;

/// Convert a BFloat16 bit pattern to f32.
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Convert a Float16 bit pattern to f32.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) as u32) << 31;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mant = (half & 0x3FF) as u32;
    if exp == 0 && mant == 0 {
        return f32::from_bits(sign);
    }
    if exp == 31 {
        return if mant == 0 {
            f32::from_bits(sign | 0x7F800000)
        } else {
            f32::NAN
        };
    }
    if exp == 0 {
        // Subnormal: mant * 2^-24 exactly (the normal formula would add an
        // implicit leading 1).
        let value = (mant as f32) * 2.0f32.powi(-24);
        return if sign != 0 { -value } else { value };
    }
    f32::from_bits(sign | ((exp + (127 - 15)) << 23) | (mant << 13))
}

/// Parsed safetensors header entry (dtype string, shape, byte range).
#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

pub struct MlxWeights {
    tensors: HashMap<String, Array>,
    /// Small f32 tensors kept on the CPU side (mel filterbank / window) so
    /// the audio frontend can use the checkpoint's own preprocessing
    /// buffers without an FFI read-back.
    cpu_f32: HashMap<String, Vec<f32>>,
}

impl MlxWeights {
    /// Load `model.safetensors`, decoding every floating tensor to F32 and
    /// uploading it to the default MLX stream (GPU).
    ///
    /// Tensors are streamed one at a time: the safetensors header is parsed
    /// first, then each tensor's byte range is read and released before the
    /// next. Peak memory is therefore the F32 weights (~2x the BF16 file)
    /// plus a single tensor's temporaries, instead of file + weights.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading MLX weights from {:?}", path);
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Cannot read safetensors at {path:?}"))?;

        // --- Parse the safetensors header ---
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .context("Failed to read safetensors header length")?;
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        anyhow::ensure!(
            header_len > 0 && header_len <= 100_000_000,
            "Implausible safetensors header length {header_len}"
        );
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .context("Failed to read safetensors header")?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&header_bytes).context("Failed to parse safetensors header")?;

        let data_base = 8u64 + header_len as u64;
        let file_len = std::fs::metadata(path)
            .with_context(|| format!("Cannot stat {path:?}"))?
            .len();
        anyhow::ensure!(
            data_base <= file_len,
            "safetensors header extends past the end of {path:?}"
        );

        // --- Stream every tensor ---
        let mut tensors = HashMap::new();
        let mut cpu_f32: HashMap<String, Vec<f32>> = HashMap::new();
        let mut skipped = 0usize;
        let mut loaded = 0usize;

        for (name, info_value) in &header {
            if name == "__metadata__" {
                continue;
            }
            let info: RawTensorInfo = serde_json::from_value(info_value.clone())
                .with_context(|| format!("Invalid safetensors entry '{name}'"))?;
            let (start, end) = info.data_offsets;
            anyhow::ensure!(
                end >= start && data_base + end <= file_len,
                "Tensor '{name}' has an out-of-range byte range {start}..{end}"
            );
            let nbytes = (end - start) as usize;

            // Element size by dtype; anything not float is metadata (skipped).
            let elem_size = match info.dtype.as_str() {
                "BF16" | "F16" => 2usize,
                "F32" => 4usize,
                _ => 0usize,
            };
            if elem_size == 0 {
                // Integer/scalar metadata (e.g. BatchNorm num_batches_tracked)
                // is not a model weight. Counted and summarised once instead
                // of warning per tensor.
                skipped += 1;
                continue;
            }
            let shape: Vec<i32> = info.shape.iter().map(|&d| d as i32).collect();
            let product: usize = shape.iter().map(|&d| d.max(0) as usize).product();
            anyhow::ensure!(
                nbytes == product * elem_size,
                "Tensor '{name}' has {} bytes but shape {:?} x {} implies {}",
                nbytes,
                info.shape,
                elem_size,
                product * elem_size
            );

            let decode_f32 = |raw: &[u8]| -> Vec<f32> {
                match info.dtype.as_str() {
                    "BF16" => raw
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
                        .collect(),
                    "F16" => raw
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                        .collect(),
                    _ => raw
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                }
            };

            // Read exactly this tensor's byte range, then convert and upload.
            // The raw slice is dropped before the next tensor is read.
            file.seek(SeekFrom::Start(data_base + start))
                .with_context(|| format!("Seek failed for tensor '{name}'"))?;
            let mut raw = vec![0u8; nbytes];
            file.read_exact(&mut raw)
                .with_context(|| format!("Read failed for tensor '{name}'"))?;

            let f32_vals: Vec<f32> = decode_f32(&raw);
            debug_assert_eq!(f32_vals.len(), product);

            if name == "preprocessor.featurizer.fb" || name == "preprocessor.featurizer.window" {
                // Keep the mel filterbank/window on the CPU for audio.rs.
                cpu_f32.insert(name.clone(), f32_vals.clone());
            }
            tensors.insert(name.to_string(), Array::from_data_f32(&f32_vals, &shape));
            loaded += 1;
            if loaded.is_multiple_of(256) {
                tracing::info!("Loaded {loaded} tensors...");
            }
        }

        if skipped > 0 {
            tracing::debug!("Skipped {skipped} non-float metadata tensors (BatchNorm counters)");
        }
        tracing::info!("Loaded {loaded} MLX tensors");
        Ok(Self { tensors, cpu_f32 })
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.tensors
            .get(name)
            .with_context(|| format!("Missing weight: '{name}'"))
    }

    /// A small float tensor kept on the CPU side (by exact name), if present.
    pub fn tensor_f32(&self, name: &str) -> Option<&[f32]> {
        self.cpu_f32.get(name).map(|v| v.as_slice())
    }
}

/// One tensor in a tiny safetensors file written by tests.
#[cfg(test)]
pub(crate) struct RawTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

#[cfg(test)]
pub(crate) fn f32_tensor(name: &str, shape: &[u64], fill: f32) -> RawTensor {
    let n: u64 = shape.iter().copied().product();
    let mut bytes = Vec::with_capacity(n as usize * 4);
    for _ in 0..n {
        bytes.extend_from_slice(&fill.to_le_bytes());
    }
    RawTensor {
        name: name.to_string(),
        dtype: "F32".to_string(),
        shape: shape.to_vec(),
        bytes,
    }
}

/// Write a safetensors file. Offsets are relative to the data section, which
/// starts immediately after the JSON header.
#[cfg(test)]
pub(crate) fn write_safetensors(path: &Path, tensors: &[RawTensor]) -> std::io::Result<()> {
    use std::io::Write;

    let mut offset = 0u64;
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let start = offset;
        let end = start + tensor.bytes.len() as u64;
        offset = end;
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header).expect("safetensors header");
    let mut file = std::fs::File::create(path)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    for tensor in tensors {
        file.write_all(&tensor.bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_subnormals_convert_exactly() {
        // Smallest subnormal: mant=1 → 2^-24.
        assert_eq!(f16_to_f32(1), 2.0f32.powi(-24));
        // Largest subnormal: mant=0x3FF → 1023 * 2^-24.
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2.0f32.powi(-24));
        // Zero / negative zero.
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        // One: exp=15, mant=0.
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // Inf / NaN.
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn bf16_rounds_trip_through_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ct-weights-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn load_rejects_a_missing_or_truncated_file() {
        let err = match MlxWeights::load("/no/such/model.safetensors") {
            Err(err) => err,
            Ok(_) => panic!("missing file should fail"),
        };
        assert!(format!("{err:#}").contains("Cannot read safetensors"));

        let dir = temp_dir("bad");
        let path = dir.join("model.safetensors");
        std::fs::write(&path, 100_000_001u64.to_le_bytes()).unwrap();
        let err = match MlxWeights::load(&path) {
            Err(err) => err,
            Ok(_) => panic!("huge header should fail"),
        };
        assert!(format!("{err:#}").contains("Implausible safetensors header"));

        std::fs::write(&path, 20u64.to_le_bytes()).unwrap();
        assert!(MlxWeights::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_reads_float_dtypes_and_skips_integer_metadata() {
        let _guard = crate::mlx::stream::test_lock();
        crate::mlx::stream::init_mlx(false);

        let dir = temp_dir("ok");
        let path = dir.join("model.safetensors");
        let tensors = vec![
            f32_tensor("plain", &[2, 3], 0.5),
            f32_tensor("preprocessor.featurizer.fb", &[2, 2], 0.25),
            f32_tensor("preprocessor.featurizer.window", &[4], 1.0),
            RawTensor {
                name: "half".into(),
                dtype: "F16".into(),
                shape: vec![2],
                // 1.0 and -1.0 in float16.
                bytes: [0x3C00u16, 0xBC00]
                    .into_iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            },
            RawTensor {
                name: "brain".into(),
                dtype: "BF16".into(),
                shape: vec![1],
                bytes: 0x3F80u16.to_le_bytes().to_vec(),
            },
            RawTensor {
                name: "batch_norm.num_batches_tracked".into(),
                dtype: "I64".into(),
                shape: vec![1],
                bytes: 7u64.to_le_bytes().to_vec(),
            },
        ];
        // A declared size that does not match the payload is rejected.
        let bad = dir.join("bad-size.safetensors");
        let mut mismatch = f32_tensor("mismatch", &[4], 1.0);
        mismatch.bytes.truncate(4);
        write_safetensors(&bad, &[mismatch]).unwrap();
        let err = match MlxWeights::load(&bad) {
            Err(err) => err,
            Ok(_) => panic!("mismatched tensor size should fail"),
        };
        assert!(format!("{err:#}").contains("bytes"));

        write_safetensors(&path, &tensors).unwrap();
        let weights = MlxWeights::load(&path).unwrap();
        assert_eq!(weights.get("plain").unwrap().shape(), vec![2, 3]);
        assert!(weights.get("batch_norm.num_batches_tracked").is_err());
        assert_eq!(
            weights.tensor_f32("preprocessor.featurizer.fb").unwrap(),
            &[0.25, 0.25, 0.25, 0.25]
        );
        assert_eq!(
            weights
                .tensor_f32("preprocessor.featurizer.window")
                .unwrap()
                .len(),
            4
        );
        let half = weights.get("half").unwrap();
        half.eval();
        assert_eq!(half.shape(), vec![2]);
        let brain = weights.get("brain").unwrap();
        brain.eval();
        assert_eq!(brain.dim(0), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
