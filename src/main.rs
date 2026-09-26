//! cohere-transcribe — single-binary Cohere Transcribe ASR for Apple Silicon.
//!
//! Downloads the model from Hugging Face on first run (no Python needed),
//! then transcribes audio files (WAV/FLAC/MP3/AAC/OGG) to text.
//!
//! Engine: a Rust implementation of `CohereLabs/cohere-transcribe-03-2026`
//! (FastConformer encoder + transformer decoder) running on Apple MLX
//! (Metal GPU).
//!
//! License: Apache-2.0 — see LICENSE, NOTICE, and THIRD_PARTY_NOTICES.md.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

mod audio;
mod config;
mod download;
mod mlx;
mod tokenizer;

const MODEL_ID: &str = "CohereLabs/cohere-transcribe-03-2026";
/// Embedded id->piece vocabulary extracted from the model's SentencePiece
/// tokenizer (checked into the tree; avoids any Python runtime dep).
const VOCAB_JSON: &str = include_str!("../vocab.json");

/// Fixed prompt length produced by [`tokenizer::SpecialTokens::build_prompt`]
/// (9 control tokens) — used with --max-tokens to bound decoder positions.
const PROMPT_TOKENS: usize = 9;

fn cache_home() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".cache")
}

fn default_model_dir() -> PathBuf {
    cache_home().join("cohere-transcriber/models/cohere-transcribe-03-2026")
}

/// Files that must exist for a model directory to be considered complete.
/// `vocab.json` is deliberately NOT required: the binary embeds a copy and
/// falls back to it (see `load_tokenizer_with_embedded_fallback`), so an
/// older model dir without that file keeps working offline.
const REQUIRED_MODEL_FILES: [&str; 3] =
    ["config.json", "model.safetensors", "tokenizer_config.json"];

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    for required in REQUIRED_MODEL_FILES {
        anyhow::ensure!(
            model_dir.join(required).exists(),
            "Missing required file '{}' in {:?}",
            required,
            model_dir
        );
    }
    Ok(())
}

#[derive(Parser, Debug)]
#[command(
    name = "cohere-transcribe",
    version,
    about = "Cohere Transcribe ASR — single native binary (Apple Silicon)",
    long_about = "Transcribe audio files using the Cohere Transcribe model.\n\n\
                  The model is downloaded automatically from Hugging Face on first run\n\
                  (requires HF_TOKEN env var or an existing `huggingface-cli login`).\n\
                  No Python or ffmpeg required."
)]
struct Args {
    /// Model directory (created + auto-downloaded on first run if missing)
    #[arg(short, long)]
    model_dir: Option<PathBuf>,

    /// Audio file(s) to transcribe
    #[arg(
        required_unless_present = "download_only",
        conflicts_with = "download_only"
    )]
    audio_files: Vec<PathBuf>,

    /// Language code (en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko)
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Disable punctuation in the output
    #[arg(long)]
    no_punctuation: bool,

    /// Maximum number of tokens to generate per audio segment (bounded by
    /// the model's 1024-row position table: prompt + tokens must fit)
    #[arg(
        long,
        default_value_t = 448,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    max_tokens: usize,

    /// Download the model into --model-dir (re-downloading even if present)
    /// and exit
    #[arg(long)]
    download_only: bool,

    /// Log verbosity (-v info, -vv debug)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> Result<()> {
    run(Args::parse())
}

fn run(args: Args) -> Result<()> {
    let log_level = match args.verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    // Ignore an already-installed subscriber so tests can call `run` twice.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .try_init();

    let model_dir = args.model_dir.clone().unwrap_or_else(default_model_dir);
    // Allow pointing at a mirror repo (e.g. an ungated community mirror).
    let model_id = std::env::var("COHERE_MODEL_ID").unwrap_or_else(|_| MODEL_ID.to_string());

    // Fail fast on bad audio paths before touching the network or the GPU:
    // a typo'd filename should not first trigger a 3.8 GB model download.
    if !args.download_only {
        for audio_path in &args.audio_files {
            anyhow::ensure!(
                audio_path.is_file(),
                "Audio file not found: {:?}",
                audio_path.display()
            );
        }
    }

    ensure_model(&model_id, &model_dir, args.download_only)?;
    // ensure_model only downloads when files are missing (or when
    // --download-only forces a refresh); the directory is validated here so
    // both paths fail loudly instead of mid-inference.
    validate_model_dir(&model_dir)?;

    if args.download_only {
        return Ok(());
    }

    tracing::info!("Loading model config...");
    let cfg = config::ModelConfig::load(&model_dir)?;
    cfg.validate()?;

    tracing::info!("Loading tokenizer...");
    let tokenizer = load_tokenizer_with_embedded_fallback(&model_dir)?;

    anyhow::ensure!(
        cfg.supported_languages.contains(&args.language),
        "Language '{}' is not supported. Supported: {:?}",
        args.language,
        cfg.supported_languages
    );

    // The decoder prompt is a fixed 9 control tokens; together with
    // --max-tokens it must fit the model's positional-encoding table.
    if let Some(max_seq) = cfg.transf_decoder.config_dict.max_sequence_length {
        anyhow::ensure!(
            args.max_tokens.saturating_add(PROMPT_TOKENS) <= max_seq,
            "--max-tokens {} + {} prompt tokens exceeds the model's max sequence length {max_seq}",
            args.max_tokens,
            PROMPT_TOKENS
        );
    }

    tracing::info!("Backend: MLX (Apple Metal)");
    mlx::stream::init_mlx(true);

    tracing::info!("Loading model weights (this may take a minute)...");
    let weights = mlx::weights::MlxWeights::load(model_dir.join("model.safetensors"))?;

    // Prefer the checkpoint's own mel filterbank and analysis window
    // (`preprocessor.featurizer.fb` / `.window`) over the computed ones.
    let mut mel_cfg = audio::MelConfig::from_model_config(&cfg);
    if let Some(fb) = weights.tensor_f32("preprocessor.featurizer.fb") {
        tracing::debug!(
            "Using the checkpoint's mel filterbank ({} weights)",
            fb.len()
        );
        mel_cfg.filterbank = Some(fb.to_vec());
    }
    if let Some(w) = weights.tensor_f32("preprocessor.featurizer.window") {
        tracing::debug!(
            "Using the checkpoint's analysis window ({} samples)",
            w.len()
        );
        mel_cfg.window = Some(w.to_vec());
    }

    tracing::info!("Building encoder...");
    let encoder = mlx::encoder::ConformerEncoder::load(&weights, &cfg)
        .context("Failed to load ConformerEncoder")?;

    tracing::info!("Building decoder...");
    let decoder = mlx::decoder::TransformerDecoder::load(&weights, &cfg)
        .context("Failed to load TransformerDecoder")?;

    let ctx = TranscribeContext {
        mel_cfg: &mel_cfg,
        encoder: &encoder,
        decoder: &decoder,
        tokenizer: &tokenizer,
        language: &args.language,
        punctuation: !args.no_punctuation,
        max_tokens: args.max_tokens,
        min_energy_window_samples: cfg.min_energy_window_samples,
    };

    for audio_path in &args.audio_files {
        if args.audio_files.len() > 1 {
            eprintln!("[{}]", audio_path.display());
        }
        let transcript = process_audio(
            audio_path,
            &ctx,
            cfg.max_audio_clip_s,
            cfg.overlap_chunk_second,
        )?;
        if transcript.trim().is_empty() {
            eprintln!(
                "Warning: no speech recognised in {} (empty transcript)",
                audio_path.display()
            );
        }
        println!("{}", transcript);
    }
    Ok(())
}

/// Load the tokenizer, falling back to the copy of `vocab.json` embedded in
/// the binary when the model directory has no usable `vocab.json` (missing
/// or unparseable — e.g. an interrupted write or a foreign orientation).
fn load_tokenizer_with_embedded_fallback(model_dir: &Path) -> Result<tokenizer::Tokenizer> {
    let vocab = match tokenizer::Vocab::load(model_dir) {
        Ok(vocab) => vocab,
        Err(e) => {
            tracing::warn!(
                "vocab.json unusable in {:?} ({e:#}); using the vocabulary embedded in the binary",
                model_dir
            );
            tokenizer::Vocab::from_json_str(VOCAB_JSON)
                .context("Failed to parse embedded vocab.json")?
        }
    };
    tokenizer::Tokenizer::from_vocab_and_config_dir(vocab, model_dir)
}

/// Ensure the model directory exists and is complete, downloading it from
/// Hugging Face (via hf-hub) on first use. With `force`, re-download even
/// when the directory already looks complete (--download-only refresh).
fn ensure_model(model_id: &str, model_dir: &Path, force: bool) -> Result<()> {
    let present = REQUIRED_MODEL_FILES
        .iter()
        .all(|f| model_dir.join(f).exists());
    if present && !force {
        tracing::info!("Model found at {:?}", model_dir);
        return Ok(());
    }
    if present && force {
        tracing::info!("Refreshing model at {:?} (--download-only)", model_dir);
    } else {
        tracing::info!(
            "Model not found at {:?} — downloading '{model_id}' from Hugging Face...",
            model_dir
        );
    }
    download::download_model(model_id, model_dir, force)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Audio processing (MLX backend)
// ---------------------------------------------------------------------------

/// Everything one transcription needs besides the audio itself: model
/// handles plus the request's language and decoding settings.
struct TranscribeContext<'a> {
    mel_cfg: &'a audio::MelConfig,
    encoder: &'a mlx::encoder::ConformerEncoder,
    decoder: &'a mlx::decoder::TransformerDecoder,
    tokenizer: &'a tokenizer::Tokenizer,
    language: &'a str,
    punctuation: bool,
    max_tokens: usize,
    /// Energy-window size (samples) used when searching for chunk boundaries.
    min_energy_window_samples: usize,
}

fn process_audio(
    audio_path: &Path,
    ctx: &TranscribeContext,
    max_clip_s: f64,
    overlap_s: f64,
) -> Result<String> {
    tracing::info!("Loading audio: {:?}", audio_path);
    let samples = audio::load_audio(audio_path, ctx.mel_cfg.sample_rate)
        .with_context(|| format!("Failed to load audio: {:?}", audio_path))?;

    tracing::info!(
        "Audio loaded: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / ctx.mel_cfg.sample_rate as f64
    );

    anyhow::ensure!(
        max_clip_s > 0.0,
        "max_audio_clip_s must be positive (got {max_clip_s})"
    );
    anyhow::ensure!(
        overlap_s >= 0.0 && overlap_s < max_clip_s,
        "overlap_chunk_second ({overlap_s}) must be in [0, max_audio_clip_s ({max_clip_s}))"
    );
    if samples.is_empty() {
        anyhow::bail!("Empty audio: {:?}", audio_path);
    }

    // Chunking follows the model's preprocessing pipeline: clips up to
    // (max_audio_clip_s - overlap_chunk_second) go through whole; longer
    // input is split at the quietest energy window near each nominal
    // boundary, producing NON-overlapping chunks (overlap_chunk_second is
    // the split-search context, not literal waveform overlap).
    let fast_path_s = (max_clip_s - overlap_s).max(0.0);
    let duration_s = samples.len() as f64 / ctx.mel_cfg.sample_rate as f64;

    let parts: Vec<String> = if duration_s <= fast_path_s {
        let text = transcribe_chunk(&samples, ctx)?;
        mlx::stream::clear_cache();
        vec![text]
    } else {
        let chunks = split_audio_chunks_energy(
            &samples,
            ctx.mel_cfg.sample_rate,
            max_clip_s,
            overlap_s,
            ctx.min_energy_window_samples,
        );
        let mut parts = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            parts.push(transcribe_chunk(chunk, ctx)?);
            // Free the per-chunk decode graph (KV caches and other cached
            // Metal buffers) before the next chunk.
            mlx::stream::clear_cache();
        }
        parts
    };

    Ok(join_chunk_texts(&parts, chunk_separator(ctx.language)))
}

/// Split a waveform into chunks at low-energy boundaries, mirroring the
/// model's `split_audio_chunks_energy`: walk `chunk_size` frames at a
/// time, then move the boundary to the quietest `min_energy_window`-sample
/// window inside the last `boundary_context` samples of the nominal chunk.
fn split_audio_chunks_energy(
    samples: &[f32],
    sample_rate: usize,
    max_clip_s: f64,
    overlap_s: f64,
    min_energy_window: usize,
) -> Vec<&[f32]> {
    let chunk_size = ((max_clip_s * sample_rate as f64).round() as usize).max(1);
    let boundary_context = ((overlap_s * sample_rate as f64).round() as usize).max(1);
    let min_energy_window = min_energy_window.max(1);
    let total = samples.len();

    if total <= chunk_size {
        return vec![samples];
    }

    let mut chunks: Vec<&[f32]> = Vec::new();
    let mut idx = 0usize;
    while idx < total {
        if idx + chunk_size >= total {
            chunks.push(&samples[idx..total]);
            break;
        }

        let search_start = idx.max(idx + chunk_size - boundary_context);
        let search_end = (idx + chunk_size).min(total);
        let split_point = if search_end <= search_start {
            idx + chunk_size
        } else {
            find_split_point_energy(samples, search_start, search_end, min_energy_window)
        };
        // Always make progress: at least one sample per chunk.
        let split_point = split_point.clamp(idx + 1, total);
        chunks.push(&samples[idx..split_point]);
        idx = split_point;
    }
    chunks
}

/// Index of the quietest `window`-sample window inside
/// [start_idx, end_idx) (rms energy per window).
fn find_split_point_energy(
    samples: &[f32],
    start_idx: usize,
    end_idx: usize,
    window: usize,
) -> usize {
    let segment = &samples[start_idx..end_idx];
    if segment.len() <= window {
        return (start_idx + end_idx) / 2;
    }
    let mut best_i = 0usize;
    let mut best_energy = f32::INFINITY;
    let upper = segment.len() - window;
    let mut i = 0usize;
    while i < upper {
        let e = segment[i..i + window]
            .iter()
            .map(|&s| s * s)
            .sum::<f32>()
            .sqrt();
        if e < best_energy {
            best_energy = e;
            best_i = i;
        }
        i += window;
    }
    start_idx + best_i
}

/// Join chunk texts like the model's `join_chunk_texts`: strip each part,
/// drop empties, join with the language-appropriate separator ("" for
/// scripts without spaces).
fn join_chunk_texts(parts: &[String], separator: &str) -> String {
    parts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}

/// `""` for the languages whose scripts carry no spaces, else `" "`.
fn chunk_separator(language: &str) -> &'static str {
    if language == "ja" || language == "zh" {
        ""
    } else {
        " "
    }
}

fn transcribe_chunk(samples: &[f32], ctx: &TranscribeContext) -> Result<String> {
    anyhow::ensure!(!samples.is_empty(), "Cannot transcribe empty audio chunk");
    tracing::debug!("Computing mel features for {} samples...", samples.len());
    let dithered = add_dither(samples, ctx.mel_cfg.dither as f32, samples.len() as u64);
    let mel = audio::compute_mel_features(&dithered, ctx.mel_cfg);
    // A clip shorter than one hop has no real frames (only the STFT pad).
    if mel.valid_frames == 0 {
        return Ok(String::new());
    }
    let (flat, shape) = audio::mel_to_tensor_data(&mel.rows);
    tracing::debug!(
        "Mel features shape: {:?} (valid frames {})",
        shape,
        mel.valid_frames
    );

    // audio.rs reports shapes as i64; the MLX C API uses i32.

    let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let mel_array = mlx::array::Array::from_data_f32(&flat, &shape_i32);

    mlx::inference::transcribe(
        mlx::inference::MelInput {
            features: &mel_array,
            valid_frames: mel.valid_frames as i32,
        },
        ctx.encoder,
        ctx.decoder,
        ctx.tokenizer,
        ctx.language,
        ctx.punctuation,
        ctx.max_tokens,
    )
}

/// Deterministic dither. The seed is the clip's sample count, which is what
/// the model's preprocessor uses; the noise itself is Gaussian at `dither`.
fn add_dither(samples: &[f32], dither: f32, seed: u64) -> Vec<f32> {
    // The model's preprocessor skips dither for any non-positive value.
    if dither <= 0.0 {
        return samples.to_vec();
    }
    // Simple LCG pseudo-random for reproducibility
    let mut rng = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = samples.to_vec();
    for s in out.iter_mut() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Full 32-bit draws → u, v uniform in [0, 1). Box-Muller requires
        // the full range: a half-range input skews the output (std ~1.3x
        // and a non-Gaussian shape).
        let u = (rng >> 32) as f32 / (u32::MAX as f32 + 1.0);
        // Box-Muller transform for Gaussian noise
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (rng >> 32) as f32 / (u32::MAX as f32 + 1.0);
        let noise = (-2.0 * u.max(1e-38).ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
        *s += dither * noise;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_split_single_chunk_under_threshold() {
        let samples = vec![0.5f32; 16000]; // 1 s
        let chunks = split_audio_chunks_energy(&samples, 16000, 35.0, 5.0, 1600);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 16000);
    }

    #[test]
    fn energy_split_covers_everything_and_progresses() {
        // 100 s of alternating loud/quiet blocks; chunks must tile the
        // input without overlap and each must be non-empty and <= chunk cap.
        let sr = 16000;
        let mut samples = Vec::with_capacity(100 * sr);
        for t in 0..100 * sr {
            let loud = (t / sr) % 2 == 0;
            samples.push(if loud { 0.8 } else { 0.0 });
        }
        let chunks = split_audio_chunks_energy(&samples, sr, 35.0, 5.0, 1600);
        assert!(
            chunks.len() >= 3,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        let chunk_cap = (35.0 * sr as f64).round() as usize;
        let mut covered = 0usize;
        for c in &chunks {
            assert!(!c.is_empty());
            assert!(c.len() <= chunk_cap);
            covered += c.len();
        }
        assert_eq!(covered, samples.len());
    }

    #[test]
    fn energy_split_prefers_quiet_boundaries() {
        // 3 nominal chunks; a silent stretch sits just before the first
        // nominal boundary — the split must land inside the search window.
        let sr = 16000;
        let mut samples = vec![0.9f32; 80 * sr];
        // Quiet 1 s window ending right at the 35 s boundary.
        for s in samples.iter_mut().take(35 * sr).skip(34 * sr) {
            *s = 0.0;
        }
        let chunks = split_audio_chunks_energy(&samples, sr, 35.0, 5.0, 1600);
        let boundary = chunks[0].len();
        // Inside [30 s, 35 s] and at/inside the quiet region.
        assert!(
            boundary >= 34 * sr && boundary <= 35 * sr,
            "boundary at {boundary}"
        );
        assert!(chunks.iter().map(|c| c.len()).sum::<usize>() == 80 * sr);
    }

    #[test]
    fn find_split_point_handles_short_segments() {
        // A segment at most one window long has no grid point to search:
        // the model's implementation returns the midpoint.
        let samples = vec![1.0f32; 10_000];
        assert_eq!(
            find_split_point_energy(&samples, 1_000, 2_000, 1_600),
            1_500
        );
        // A long constant segment returns the first grid point.
        assert_eq!(
            find_split_point_energy(&samples, 1_000, 5_000, 1_600),
            1_000
        );
        // A quiet window at the second grid slot wins.
        let mut q = samples.clone();
        for s in q.iter_mut().take(1_000 + 2 * 1_600).skip(1_000 + 1_600) {
            *s = 0.0;
        }
        assert_eq!(
            find_split_point_energy(&q, 1_000, 5_000, 1_600),
            1_000 + 1_600
        );
    }

    #[test]
    fn join_strips_and_uses_language_separator() {
        let parts = vec!["  hello ".to_string(), String::new(), "world ".to_string()];
        assert_eq!(join_chunk_texts(&parts, " "), "hello world");
        assert_eq!(join_chunk_texts(&parts, ""), "helloworld");
        assert_eq!(join_chunk_texts(&[], " "), "");
        assert_eq!(chunk_separator("ja"), "");
        assert_eq!(chunk_separator("zh"), "");
        assert_eq!(chunk_separator("en"), " ");
        assert_eq!(chunk_separator("fr"), " ");
    }

    #[test]
    fn dither_is_roughly_gaussian_at_the_configured_scale() {
        // The configured dither is the target standard deviation; the
        // tolerance rejects a half-range Box-Muller input (which would
        // inflate the std ~1.3x).
        let n = 200_000;
        let noise = add_dither(&vec![0.0; n], 1e-3, 12345);
        let mean = noise.iter().sum::<f32>() / n as f32;
        let var = noise.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();
        assert!(mean.abs() < 2e-4, "mean {mean} should be ~0");
        assert!(
            (std - 1e-3).abs() < 2e-4,
            "std {std} should be ~1e-3 (the dither amplitude)"
        );
        // Zero dither must be the identity.
        assert_eq!(add_dither(&[0.25, -0.25], 0.0, 7), vec![0.25, -0.25]);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ct-main-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_wav(path: &std::path::Path, samples: &[i16]) {
        let data_bytes = samples.len() * 2;
        let mut bytes = Vec::with_capacity(44 + data_bytes);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&(16000u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    const TINY_CONFIG: &str = r#"{
        "encoder": {"d_model": 16, "n_layers": 1, "n_heads": 4},
        "transf_decoder": {"config_dict": {
            "hidden_size": 16, "num_attention_heads": 4, "num_layers": 1,
            "max_sequence_length": 64
        }},
        "preprocessor": {"sample_rate": 16000, "features": 8, "n_fft": 64,
                         "window_size": 0.002, "window_stride": 0.001, "dither": 0.0},
        "max_audio_clip_s": 0.016,
        "overlap_chunk_second": 0.004,
        "sample_rate": 16000,
        "supported_languages": ["en"],
        "min_energy_window_samples": 16
    }"#;

    const TINY_TOKENS: &str = r#"{
        "added_tokens_decoder": {
            "1": {"content": "<|endoftext|>", "special": true},
            "2": {"content": "<|startoftranscript|>", "special": true},
            "3": {"content": "<|startofcontext|>", "special": true},
            "4": {"content": "<|emo:undefined|>", "special": true},
            "5": {"content": "<|pnc|>", "special": true},
            "6": {"content": "<|nopnc|>", "special": true},
            "7": {"content": "<|noitn|>", "special": true},
            "8": {"content": "<|notimestamp|>", "special": true},
            "9": {"content": "<|nodiarize|>", "special": true},
            "10": {"content": "<|en|>", "special": true}
        }
    }"#;

    fn args(
        dir: std::path::PathBuf,
        audio: Vec<std::path::PathBuf>,
        language: &str,
        max_tokens: usize,
        download_only: bool,
    ) -> Args {
        Args {
            model_dir: Some(dir),
            audio_files: audio,
            language: language.to_string(),
            no_punctuation: false,
            max_tokens,
            download_only,
            verbose: 0,
        }
    }

    #[test]
    fn missing_audio_fails_before_any_download() {
        let dir = temp_dir("missing-audio");
        let err = run(args(
            dir.join("no-model"),
            vec![dir.join("no-such.wav")],
            "en",
            4,
            false,
        ))
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("Audio file not found"),
            "{err:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn complete_model_dir_skips_download() {
        let dir = temp_dir("present");
        for name in ["config.json", "model.safetensors", "tokenizer_config.json"] {
            std::fs::write(dir.join(name), b"{}").unwrap();
        }
        // force=false is the normal path: existing files are left alone.
        ensure_model("CohereLabs/cohere-transcribe-03-2026", &dir, false).unwrap();
        let err = validate_model_dir(&dir.join("empty")).unwrap_err();
        assert!(format!("{err:#}").contains("Missing required file"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_unknown_language_and_an_oversized_token_budget() {
        let dir = temp_dir("bad-request");
        std::fs::write(dir.join("config.json"), TINY_CONFIG).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), TINY_TOKENS).unwrap();
        std::fs::write(dir.join("vocab.json"), r#"{"12":"hi"}"#).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"placeholder").unwrap();
        write_wav(&dir.join("clip.wav"), &[0, 1000, -1000, 500]);
        let err = run(args(
            dir.clone(),
            vec![dir.join("clip.wav")],
            "zz",
            4,
            false,
        ))
        .unwrap_err();
        assert!(format!("{err:#}").contains("not supported"), "{err:#}");
        let err = run(args(
            dir.clone(),
            vec![dir.join("clip.wav")],
            "en",
            100,
            false,
        ))
        .unwrap_err();
        assert!(format!("{err:#}").contains("max sequence"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tokenizer_falls_back_to_the_embedded_vocab() {
        let dir = temp_dir("vocab-fallback");
        std::fs::write(dir.join("tokenizer_config.json"), TINY_TOKENS).unwrap();
        std::fs::write(dir.join("vocab.json"), b"{not json").unwrap();
        let tok = load_tokenizer_with_embedded_fallback(&dir).unwrap();
        // The embedded table is the real model vocab, so a low id decodes.
        let _ = tok.decode(&[10]);

        std::fs::write(dir.join("vocab.json"), r#"{"12": "hi"}"#).unwrap();
        let tok = load_tokenizer_with_embedded_fallback(&dir).unwrap();
        assert_eq!(tok.decode(&[12]), "hi");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn tiny_checkpoint(path: &std::path::Path) {
        use crate::mlx::weights::{f32_tensor, write_safetensors, RawTensor};
        let mut tensors: Vec<RawTensor> = Vec::new();
        let add = |tensors: &mut Vec<RawTensor>, name: &str, shape: &[u64], fill: f32| {
            tensors.push(f32_tensor(name, shape, fill));
        };
        let norm = |tensors: &mut Vec<RawTensor>, name: &str, n: u64| {
            tensors.push(f32_tensor(&format!("{name}.weight"), &[n], 1.0));
            tensors.push(f32_tensor(&format!("{name}.bias"), &[n], 0.0));
        };
        let linear = |tensors: &mut Vec<RawTensor>, name: &str, out: u64, inn: u64| {
            tensors.push(f32_tensor(&format!("{name}.weight"), &[out, inn], 0.02));
            tensors.push(f32_tensor(&format!("{name}.bias"), &[out], 0.0));
        };

        let pre = "encoder.pre_encode";
        add(
            &mut tensors,
            &format!("{pre}.conv.0.weight"),
            &[4, 1, 3, 3],
            0.02,
        );
        add(&mut tensors, &format!("{pre}.conv.0.bias"), &[4], 0.0);
        add(
            &mut tensors,
            &format!("{pre}.conv.2.weight"),
            &[4, 1, 3, 3],
            0.02,
        );
        add(&mut tensors, &format!("{pre}.conv.2.bias"), &[4], 0.0);
        add(
            &mut tensors,
            &format!("{pre}.conv.3.weight"),
            &[4, 4, 1, 1],
            0.02,
        );
        add(&mut tensors, &format!("{pre}.conv.3.bias"), &[4], 0.0);
        add(
            &mut tensors,
            &format!("{pre}.conv.5.weight"),
            &[4, 1, 3, 3],
            0.02,
        );
        add(&mut tensors, &format!("{pre}.conv.5.bias"), &[4], 0.0);
        add(
            &mut tensors,
            &format!("{pre}.conv.6.weight"),
            &[4, 4, 1, 1],
            0.02,
        );
        add(&mut tensors, &format!("{pre}.conv.6.bias"), &[4], 0.0);
        // After three stride-2 convs, 8 mel bins collapse to 4 features.
        add(&mut tensors, &format!("{pre}.out.weight"), &[16, 4], 0.02);
        add(&mut tensors, &format!("{pre}.out.bias"), &[16], 0.0);

        let layer = "encoder.layers.0";
        norm(&mut tensors, &format!("{layer}.norm_feed_forward1"), 16);
        linear(
            &mut tensors,
            &format!("{layer}.feed_forward1.linear1"),
            32,
            16,
        );
        linear(
            &mut tensors,
            &format!("{layer}.feed_forward1.linear2"),
            16,
            32,
        );
        norm(&mut tensors, &format!("{layer}.norm_self_att"), 16);
        for proj in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            linear(&mut tensors, &format!("{layer}.self_attn.{proj}"), 16, 16);
        }
        add(
            &mut tensors,
            &format!("{layer}.self_attn.linear_pos.weight"),
            &[16, 16],
            0.02,
        );
        add(
            &mut tensors,
            &format!("{layer}.self_attn.pos_bias_u"),
            &[4, 4],
            0.0,
        );
        add(
            &mut tensors,
            &format!("{layer}.self_attn.pos_bias_v"),
            &[4, 4],
            0.0,
        );
        norm(&mut tensors, &format!("{layer}.norm_conv"), 16);
        add(
            &mut tensors,
            &format!("{layer}.conv.pointwise_conv1.weight"),
            &[32, 16, 1],
            0.02,
        );
        add(
            &mut tensors,
            &format!("{layer}.conv.pointwise_conv1.bias"),
            &[32],
            0.0,
        );
        add(
            &mut tensors,
            &format!("{layer}.conv.depthwise_conv.weight"),
            &[16, 1, 3],
            0.02,
        );
        add(
            &mut tensors,
            &format!("{layer}.conv.depthwise_conv.bias"),
            &[16],
            0.0,
        );
        norm(&mut tensors, &format!("{layer}.conv.batch_norm"), 16);
        add(
            &mut tensors,
            &format!("{layer}.conv.batch_norm.running_mean"),
            &[16],
            0.0,
        );
        add(
            &mut tensors,
            &format!("{layer}.conv.batch_norm.running_var"),
            &[16],
            1.0,
        );
        add(
            &mut tensors,
            &format!("{layer}.conv.pointwise_conv2.weight"),
            &[16, 16, 1],
            0.02,
        );
        add(
            &mut tensors,
            &format!("{layer}.conv.pointwise_conv2.bias"),
            &[16],
            0.0,
        );
        norm(&mut tensors, &format!("{layer}.norm_feed_forward2"), 16);
        linear(
            &mut tensors,
            &format!("{layer}.feed_forward2.linear1"),
            32,
            16,
        );
        linear(
            &mut tensors,
            &format!("{layer}.feed_forward2.linear2"),
            16,
            32,
        );
        norm(&mut tensors, &format!("{layer}.norm_out"), 16);

        add(
            &mut tensors,
            "transf_decoder._embedding.token_embedding.weight",
            &[32, 16],
            0.02,
        );
        add(
            &mut tensors,
            "transf_decoder._embedding.position_embedding.pos_enc",
            &[64, 16],
            0.01,
        );
        norm(&mut tensors, "transf_decoder._embedding.layer_norm", 16);
        let dec = "transf_decoder._decoder";
        norm(&mut tensors, &format!("{dec}.layers.0.layer_norm_1"), 16);
        norm(&mut tensors, &format!("{dec}.layers.0.layer_norm_2"), 16);
        norm(&mut tensors, &format!("{dec}.layers.0.layer_norm_3"), 16);
        for sub in ["first_sub_layer", "second_sub_layer"] {
            for proj in ["query_net", "key_net", "value_net", "out_projection"] {
                linear(
                    &mut tensors,
                    &format!("{dec}.layers.0.{sub}.{proj}"),
                    16,
                    16,
                );
            }
        }
        linear(
            &mut tensors,
            &format!("{dec}.layers.0.third_sub_layer.dense_in"),
            32,
            16,
        );
        linear(
            &mut tensors,
            &format!("{dec}.layers.0.third_sub_layer.dense_out"),
            16,
            32,
        );
        norm(&mut tensors, &format!("{dec}.final_layer_norm"), 16);
        linear(&mut tensors, "log_softmax.mlp.layer0", 32, 16);

        write_safetensors(path, &tensors).unwrap();
    }

    #[test]
    fn tiny_model_transcribes_a_wav_including_a_split() {
        let _guard = crate::mlx::stream::test_lock();
        // Pin the process-wide runtime before `run` asks for the GPU, so this
        // test and the other MLX tests share one device.
        crate::mlx::stream::init_mlx(false);

        let dir = temp_dir("tiny");
        std::fs::write(dir.join("config.json"), TINY_CONFIG).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), TINY_TOKENS).unwrap();
        let vocab = (0..32)
            .map(|id| format!(r#""{id}":"a""#))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(dir.join("vocab.json"), format!("{{{vocab}}}")).unwrap();
        tiny_checkpoint(&dir.join("model.safetensors"));
        // 0.05 s at 16 kHz. The tiny config's clip is 0.016 s, so this is split.
        let mut samples = Vec::with_capacity(800);
        for i in 0..800 {
            let x = (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16000.0).sin();
            samples.push((x * 8000.0) as i16);
        }
        write_wav(&dir.join("clip.wav"), &samples);
        run(args(
            dir.clone(),
            vec![dir.join("clip.wav")],
            "en",
            4,
            false,
        ))
        .expect("tiny model should transcribe");
        std::fs::remove_dir_all(&dir).ok();
    }
}
