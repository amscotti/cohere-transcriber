use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

// Model configuration schema (config.json). Each struct lists exactly the
// keys the inference path consumes; anything else in the file (dropout,
// windowing names, head dimensions, …) is ignored by serde.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub transf_decoder: TransfDecoderConfig,
    pub preprocessor: PreprocessorConfig,
    pub max_audio_clip_s: f64,
    pub overlap_chunk_second: f64,
    pub sample_rate: usize,
    pub supported_languages: Vec<String>,
    /// Energy-window size (samples) used when searching for chunk boundaries.
    /// The model's default is 1600 and the shipped config omits it.
    #[serde(default = "default_min_energy_window_samples")]
    pub min_energy_window_samples: usize,
}

fn default_min_energy_window_samples() -> usize {
    1600
}

#[derive(Debug, Deserialize)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

#[derive(Debug, Deserialize)]
pub struct TransfDecoderConfig {
    pub config_dict: DecoderConfigDict,
}

#[derive(Debug, Deserialize)]
pub struct DecoderConfigDict {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    /// Rows in the decoder's fixed positional-encoding table. Optional in
    /// the schema; when absent the runtime bound on --max-tokens is not
    /// pre-checked here (inference still enforces the table's real size).
    #[serde(default)]
    pub max_sequence_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub features: usize,
    pub n_fft: usize,
    pub window_size: f64,
    pub window_stride: f64,
    pub dither: f64,
}

impl ModelConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read config.json at {:?}", path))?;
        serde_json::from_str(&content).context("Failed to parse config.json")
    }

    /// Sanity-check the values the transcription path depends on. Besides
    /// the semantic checks, every value the arithmetic divides by (or
    /// subtracts from) is validated so a malformed config from `-m` fails
    /// here with a message instead of panicking inside the mel/MLX code.
    pub fn validate(&self) -> Result<()> {
        use anyhow::ensure;

        // --- semantics ---
        ensure!(self.sample_rate > 0, "config sample_rate must be positive");
        ensure!(
            self.max_audio_clip_s > 0.0,
            "config max_audio_clip_s must be positive"
        );
        ensure!(
            self.overlap_chunk_second >= 0.0 && self.overlap_chunk_second < self.max_audio_clip_s,
            "config overlap_chunk_second ({}) must be in [0, max_audio_clip_s ({}))",
            self.overlap_chunk_second,
            self.max_audio_clip_s
        );
        ensure!(
            !self.supported_languages.is_empty(),
            "config supported_languages must not be empty"
        );

        // --- preprocessor (drives the mel math) ---
        let pp = &self.preprocessor;
        ensure!(
            pp.sample_rate > 0,
            "preprocessor.sample_rate must be positive"
        );
        ensure!(
            self.sample_rate == pp.sample_rate,
            "config sample_rate ({}) and preprocessor.sample_rate ({}) disagree; \
             the pipeline resamples to the preprocessor rate",
            self.sample_rate,
            pp.sample_rate
        );
        ensure!(pp.features > 0, "preprocessor.features must be positive");
        ensure!(pp.n_fft > 0, "preprocessor.n_fft must be positive");
        ensure!(
            pp.window_size > 0.0,
            "preprocessor.window_size must be positive"
        );
        ensure!(
            pp.window_stride > 0.0,
            "preprocessor.window_stride must be positive"
        );
        ensure!(
            pp.dither.is_finite() && pp.dither >= 0.0,
            "preprocessor.dither ({}) must be finite and non-negative",
            pp.dither
        );
        ensure!(
            self.min_energy_window_samples > 0,
            "min_energy_window_samples must be positive"
        );
        let win_length = (pp.window_size * pp.sample_rate as f64).round() as usize;
        let hop_length = (pp.window_stride * pp.sample_rate as f64).round() as usize;
        ensure!(
            hop_length > 0,
            "preprocessor.window_stride ({}) rounds to 0 samples at {} Hz",
            pp.window_stride,
            pp.sample_rate
        );
        ensure!(
            win_length >= 2,
            "preprocessor.window_size ({}) rounds to fewer than 2 samples at {} Hz",
            pp.window_size,
            pp.sample_rate
        );
        ensure!(
            win_length <= pp.n_fft,
            "window size ({win_length} samples) must not exceed n_fft ({})",
            pp.n_fft
        );

        // --- attention head divisibility (d_model / n_heads truncates) ---
        let enc = &self.encoder;
        ensure!(enc.n_heads > 0, "encoder.n_heads must be positive");
        ensure!(
            enc.d_model.is_multiple_of(enc.n_heads),
            "encoder.d_model ({}) must be divisible by encoder.n_heads ({})",
            enc.d_model,
            enc.n_heads
        );
        let dec = &self.transf_decoder.config_dict;
        ensure!(
            dec.num_attention_heads > 0,
            "transf_decoder num_attention_heads must be positive"
        );
        ensure!(
            dec.hidden_size.is_multiple_of(dec.num_attention_heads),
            "transf_decoder hidden_size ({}) must be divisible by num_attention_heads ({})",
            dec.hidden_size,
            dec.num_attention_heads
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelConfig {
        ModelConfig {
            encoder: EncoderConfig {
                d_model: 512,
                n_layers: 1,
                n_heads: 8,
            },
            transf_decoder: TransfDecoderConfig {
                config_dict: DecoderConfigDict {
                    hidden_size: 512,
                    num_attention_heads: 8,
                    num_layers: 2,
                    max_sequence_length: Some(1024),
                },
            },
            preprocessor: PreprocessorConfig {
                sample_rate: 16000,
                features: 128,
                n_fft: 512,
                window_size: 0.025,
                window_stride: 0.01,
                dither: 1e-5,
            },
            max_audio_clip_s: 35.0,
            overlap_chunk_second: 2.0,
            sample_rate: 16000,
            supported_languages: vec!["en".to_string()],
            min_energy_window_samples: 1600,
        }
    }

    #[test]
    fn accepts_sane_config() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn rejects_overlap_covering_whole_clip() {
        let mut cfg = valid_config();
        cfg.overlap_chunk_second = cfg.max_audio_clip_s;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_window_stride() {
        let mut cfg = valid_config();
        cfg.preprocessor.window_stride = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_window_larger_than_n_fft() {
        let mut cfg = valid_config();
        cfg.preprocessor.window_size = 0.05; // 800 samples > n_fft 512
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_indivisible_heads() {
        let mut cfg = valid_config();
        cfg.encoder.n_heads = 7;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_sample_rate_mismatch() {
        let mut cfg = valid_config();
        cfg.sample_rate = 8000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_negative_dither() {
        let mut cfg = valid_config();
        cfg.preprocessor.dither = -1e-5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_energy_window() {
        let mut cfg = valid_config();
        cfg.min_energy_window_samples = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn defaults_min_energy_window_when_absent() {
        let json = r#"{
            "encoder": {"d_model": 512, "n_layers": 1, "n_heads": 8},
            "transf_decoder": {"config_dict": {"hidden_size": 512, "num_attention_heads": 8, "num_layers": 2}},
            "preprocessor": {"sample_rate": 16000, "features": 128, "n_fft": 512,
                             "window_size": 0.025, "window_stride": 0.01, "dither": 1e-5},
            "max_audio_clip_s": 35.0,
            "overlap_chunk_second": 5.0,
            "sample_rate": 16000,
            "supported_languages": ["en"]
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).expect("parse");
        assert_eq!(cfg.min_energy_window_samples, 1600);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn load_reads_config_json_from_the_model_dir() {
        let dir = std::env::temp_dir().join(format!(
            "ct-cfg-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let json = r#"{
            "encoder": {"d_model": 16, "n_layers": 1, "n_heads": 4},
            "transf_decoder": {"config_dict": {"hidden_size": 16, "num_attention_heads": 4, "num_layers": 1, "max_sequence_length": 64}},
            "preprocessor": {"sample_rate": 16000, "features": 8, "n_fft": 64,
                             "window_size": 0.002, "window_stride": 0.001, "dither": 0.0},
            "max_audio_clip_s": 0.05,
            "overlap_chunk_second": 0.01,
            "sample_rate": 16000,
            "supported_languages": ["en"]
        }"#;
        std::fs::write(dir.join("config.json"), json).unwrap();
        let cfg = ModelConfig::load(&dir).unwrap();
        assert_eq!(cfg.encoder.d_model, 16);
        assert_eq!(cfg.preprocessor.features, 8);
        assert!(cfg.validate().is_ok());
        assert!(ModelConfig::load(dir.join("missing")).is_err());
        std::fs::write(dir.join("config.json"), "not json").unwrap();
        assert!(ModelConfig::load(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
