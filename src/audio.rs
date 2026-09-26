use anyhow::{Context, Result};
use std::path::Path;

/// Mel filterbank parameters matching the model's preprocessor config.
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub preemph: f64,
    pub dither: f64,
    pub log_zero_guard: f64,
    /// The checkpoint ships its own mel filterbank
    /// (`preprocessor.featurizer.fb`, librosa Slaney scale). When present it
    /// is used verbatim; otherwise the Slaney formula below is the fallback.
    pub filterbank: Option<Vec<f32>>,
    /// The checkpoint's stored analysis window
    /// (`preprocessor.featurizer.window`, symmetric Hann). When present it
    /// is used verbatim.
    pub window: Option<Vec<f32>>,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 128,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            fmin: 0.0,
            fmax: 8000.0,
            preemph: 0.97,
            dither: 1e-5,
            log_zero_guard: 2.0f64.powi(-24),
            filterbank: None,
            window: None,
        }
    }
}

impl MelConfig {
    pub fn from_model_config(cfg: &crate::config::ModelConfig) -> Self {
        let pp = &cfg.preprocessor;
        let win_length = (pp.window_size * pp.sample_rate as f64).round() as usize;
        let hop_length = (pp.window_stride * pp.sample_rate as f64).round() as usize;
        Self {
            sample_rate: pp.sample_rate,
            n_mels: pp.features,
            n_fft: pp.n_fft,
            win_length,
            hop_length,
            fmin: 0.0,
            fmax: pp.sample_rate as f64 / 2.0,
            preemph: 0.97,
            dither: pp.dither,
            log_zero_guard: 2.0f64.powi(-24),
            filterbank: None,
            window: None,
        }
    }
}

/// Hint appended to decode errors so users know what works and how to
/// convert unsupported files.
const SUPPORTED_FORMATS_HINT: &str = "Supported: WAV, FLAC, MP3, M4A/MP4 (AAC), OGG Vorbis. \
     Convert other formats first, e.g. `yt-dlp -x --audio-format m4a <url>` or \
     `ffmpeg -i <input> -ar 16000 -ac 1 out.wav`.";

/// Message for a container/codec we can't decode, with an actionable hint.
/// Opus (the usual YouTube download codec) gets a specific mention because
/// it is by far the most common case users hit.
fn unsupported_codec_message(codec: symphonia::core::codecs::CodecType) -> String {
    if codec == symphonia::core::codecs::CODEC_TYPE_OPUS {
        format!(
            "Opus audio is not supported (common in WebM files downloaded from YouTube). \
             {SUPPORTED_FORMATS_HINT}"
        )
    } else {
        format!("Audio codec is not supported. {SUPPORTED_FORMATS_HINT}")
    }
}

/// Load audio from a file, resample to the target sample rate, and return mono f32 samples.
pub fn load_audio(path: impl AsRef<Path>, target_sr: usize) -> Result<Vec<f32>> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path.as_ref())
        .with_context(|| format!("Cannot open audio file {:?}", path.as_ref()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("Unsupported audio format/container. {SUPPORTED_FORMATS_HINT}"))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        // Audio tracks carry a sample rate; this skips video/subtitle tracks
        // in muxed containers as well as unknown codecs.
        .find(|t| {
            t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL
                && t.codec_params.sample_rate.is_some()
        })
        .with_context(|| format!("No usable audio track found. {SUPPORTED_FORMATS_HINT}"))?;

    let src_sr = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")? as usize;

    let mut track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| unsupported_codec_message(track.codec_params.codec))?;

    let mut samples: Vec<f32> = Vec::new();

    // `ResetRequired` (e.g. chained OGG) demands re-examining the track list
    // and re-creating the decoder, so the loop re-runs track discovery. Real
    // I/O errors are fatal — only `UnexpectedEof` (symphonia's end-of-stream
    // signal) ends decoding, and corrupt packets are skipped rather than
    // aborting the whole file.
    'decode: loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break 'decode; // normal end of stream
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                // Re-examine tracks and re-create the decoder for the new
                // segment. Refuse a sample-rate change we cannot handle.
                let new_track = format
                    .tracks()
                    .iter()
                    .find(|t| {
                        t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL
                            && t.codec_params.sample_rate.is_some()
                    })
                    .context("No audio track after stream reset")?;
                let new_sr = new_track
                    .codec_params
                    .sample_rate
                    .context("Unknown sample rate")? as usize;
                anyhow::ensure!(
                    new_sr == src_sr,
                    "Stream reset changed the sample rate ({src_sr} → {new_sr}); unsupported"
                );
                decoder = symphonia::default::get_codecs()
                    .make(&new_track.codec_params, &DecoderOptions::default())
                    .context("Failed to re-create audio decoder after stream reset")?;
                track_id = new_track.id;
                continue 'decode;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break 'decode;
            }
            // A single corrupt packet (common in real-world MP3/AAC streams)
            // skips that packet, not the whole file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                tracing::warn!("Skipping one corrupt audio packet");
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        // Take the channel count from the decoded buffer itself, not the
        // container metadata: the two can disagree, and `chan()` panics on
        // out-of-range access.
        let buf_f32: Vec<f32> = match &decoded {
            AudioBufferRef::F32(buf) => {
                downmix_frames(buf.frames(), buf.spec().channels.count(), |ch, frame| {
                    buf.chan(ch)[frame]
                })
            }
            AudioBufferRef::S16(buf) => {
                downmix_frames(buf.frames(), buf.spec().channels.count(), |ch, frame| {
                    buf.chan(ch)[frame] as f32 / 32768.0
                })
            }
            AudioBufferRef::S32(buf) => {
                downmix_frames(buf.frames(), buf.spec().channels.count(), |ch, frame| {
                    buf.chan(ch)[frame] as f32 / 2147483648.0
                })
            }
            AudioBufferRef::U8(buf) => {
                downmix_frames(buf.frames(), buf.spec().channels.count(), |ch, frame| {
                    (buf.chan(ch)[frame] as f32 - 128.0) / 128.0
                })
            }
            _ => {
                // Convert via interleaved float for other formats
                let mut tmp = decoded.make_equivalent::<f32>();
                decoded.convert(&mut tmp);
                downmix_frames(tmp.frames(), tmp.spec().channels.count(), |ch, frame| {
                    tmp.chan(ch)[frame]
                })
            }
        };

        samples.extend(buf_f32);
    }

    // Resample if needed
    if src_sr != target_sr {
        samples = resample(&samples, src_sr, target_sr)?;
    }

    Ok(samples)
}

/// Average multi-channel planar frames to mono.
fn downmix_frames(
    n_frames: usize,
    channels: usize,
    mut sample_at: impl FnMut(usize, usize) -> f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_frames);
    for frame in 0..n_frames {
        let mut sample = 0.0f32;
        for ch in 0..channels {
            sample += sample_at(ch, frame);
        }
        out.push(sample / channels as f32);
    }
    out
}

fn resample(input: &[f32], src_sr: usize, dst_sr: usize) -> Result<Vec<f32>> {
    resample_impl(input, src_sr, dst_sr).map(|(out, _generated)| out)
}

/// Resampling core. Returns `(samples, generated)` where `generated` is the
/// total number of output samples emitted before trimming — exposed so tests
/// can prove the flush does not over-generate.
fn resample_impl(input: &[f32], src_sr: usize, dst_sr: usize) -> Result<(Vec<f32>, usize)> {
    use rubato::{FftFixedIn, Resampler};

    if input.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let ratio = dst_sr as f64 / src_sr as f64;
    let chunk_size = 4096;

    let mut resampler = FftFixedIn::<f32>::new(src_sr, dst_sr, chunk_size, 2, 1)
        .context("Failed to create resampler")?;

    // The FFT resampler has an inherent latency: the first `output_delay()`
    // output samples correspond to filter warm-up, not signal, and the tail
    // of the input only leaves the resampler after additional zero input.
    // Without compensating for both, every resampled file is time-shifted
    // and loses its tail.
    let delay = resampler.output_delay();

    let mut output = Vec::new();
    let mut pos = 0;

    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let mut chunk = input[pos..end].to_vec();
        if chunk.len() < chunk_size {
            // Zero-pad the final partial chunk so it can be processed; the
            // padding only affects samples beyond the requested length.
            chunk.resize(chunk_size, 0.0);
        }

        let out = resampler
            .process(&[chunk], None)
            .context("Resampling failed")?;
        output.extend_from_slice(&out[0]);
        pos += chunk_size;
    }

    // Flush the buffered tail by feeding zero chunks until enough output has
    // been produced to cover the warm-up delay plus the expected signal
    // length. (`process_partial(None, ..)` is NOT a flush for FftFixedIn: it
    // feeds a full zero chunk per call and would keep generating output;
    // feeding a bounded number of explicit zero chunks terminates.)
    let expected = (input.len() as f64 * ratio).ceil() as usize;
    let zero_chunk = vec![0.0f32; chunk_size];
    let mut flush_calls = 0usize;
    while output.len() < delay + expected {
        let out = resampler
            .process(std::slice::from_ref(&zero_chunk), None)
            .context("Resampling flush failed")?;
        if out[0].is_empty() {
            break;
        }
        output.extend_from_slice(&out[0]);
        flush_calls += 1;
        if flush_calls > 64 {
            tracing::warn!(
                "resampler flush did not converge after {flush_calls} calls; \
                 truncating to the expected length"
            );
            break;
        }
    }

    let generated = output.len();

    // Drop the warm-up samples, then fix the length to the expected count
    // (librosa's fix=True length: ceil(n * ratio)).
    if output.len() > delay {
        output.drain(..delay);
    } else {
        output.clear();
    }
    if output.len() < expected {
        tracing::debug!(
            "resampler produced {} of {} expected samples; padding with zeros",
            output.len(),
            expected
        );
        output.resize(expected, 0.0);
    } else {
        output.truncate(expected);
    }

    Ok((output, generated))
}

/// Compute mel filterbank matrix: (n_mels, n_fft/2+1).
/// Follows librosa.filters.mel with `htk=False` (Slaney mel scale) and
/// `norm="slaney"` — the scale the model's preprocessor uses and the one
/// baked into the checkpoint's `preprocessor.featurizer.fb` buffer. The HTK
/// formula `2595*log10(1+f/700)` differs by up to ~120% per weight at low
/// frequencies and must not be substituted.
pub fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Slaney-style piecewise Hz↔mel mapping (as in librosa.core.convert):
    // linear below 1 kHz, log above.
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| -> f64 {
        let mut mel = f / f_sp;
        if f >= min_log_hz {
            mel = min_log_mel + (f / min_log_hz).ln() / logstep;
        }
        mel
    };
    let mel_to_hz = |m: f64| -> f64 {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 center frequencies in mel space
    let mel_pts: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

    // FFT bin center frequencies
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut fb = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let f_lower = hz_pts[m];
        let f_center = hz_pts[m + 1];
        let f_upper = hz_pts[m + 2];
        // Slaney normalization factor
        let norm = 2.0 / (f_upper - f_lower);

        for k in 0..n_freqs {
            let f = fft_freqs[k];
            let w = if f >= f_lower && f <= f_center {
                (f - f_lower) / (f_center - f_lower)
            } else if f > f_center && f <= f_upper {
                (f_upper - f) / (f_upper - f_center)
            } else {
                0.0
            };
            fb[m * n_freqs + k] = (w * norm) as f32;
        }
    }

    fb
}

/// Log-mel features for one clip.
///
/// `rows` has shape (n_mels, frames). `valid_frames` is how many of those
/// frames are real: a centered STFT always emits one trailing pad frame,
/// which is zero and must be masked in the encoder.
pub struct MelFeatures {
    pub rows: Vec<Vec<f32>>,
    pub valid_frames: usize,
}

/// Compute mel log-filterbank features from raw mono audio at `cfg.sample_rate`.
pub fn compute_mel_features(audio: &[f32], cfg: &MelConfig) -> MelFeatures {
    let n = audio.len();
    if n == 0 {
        return MelFeatures {
            rows: vec![Vec::new(); cfg.n_mels],
            valid_frames: 0,
        };
    }

    // 1. Pre-emphasis: y[t] = x[t] - 0.97 * x[t-1]
    let mut preemphasized = Vec::with_capacity(n);
    preemphasized.push(audio[0]);
    for i in 1..n {
        preemphasized.push(audio[i] - (cfg.preemph as f32) * audio[i - 1]);
    }

    // 2. Analysis window: the checkpoint's stored (symmetric Hann) window
    //    when available, else the formula.
    let win_length = cfg.win_length;
    let hann: Vec<f32> = match &cfg.window {
        Some(w) if w.len() == win_length => w.clone(),
        Some(w) => {
            tracing::warn!(
                "checkpoint window has {} samples, expected {win_length}; using computed Hann",
                w.len()
            );
            hann_window(win_length)
        }
        None => hann_window(win_length),
    };

    // 3. STFT (see `stft_power_frames` for the exact torch.stft-compatible
    //    framing) followed by mel projection.
    let power_spectrum = stft_power_frames(&preemphasized, cfg, &hann);
    let n_frames = power_spectrum.len();
    let n_freqs = cfg.n_fft / 2 + 1;

    // 4. Mel filterbank: the checkpoint's own buffer
    //    (`preprocessor.featurizer.fb`) when present, else the Slaney
    //    formula.
    let mel_fb = match &cfg.filterbank {
        Some(fb) if fb.len() == cfg.n_mels * n_freqs => fb.clone(),
        Some(fb) => {
            tracing::warn!(
                "checkpoint filterbank has {} weights, expected {}; using computed Slaney bank",
                fb.len(),
                cfg.n_mels * n_freqs
            );
            mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        }
        None => mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax),
    };

    let log_guard = cfg.log_zero_guard as f32;
    let mut mel_features: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_mels);

    for m in 0..cfg.n_mels {
        let mut mel_row = vec![0.0f32; n_frames];
        for k in 0..n_freqs {
            let fb_val = mel_fb[m * n_freqs + k];
            if fb_val == 0.0 {
                continue;
            }
            for (t, frame) in power_spectrum.iter().enumerate() {
                mel_row[t] += fb_val * frame[k];
            }
        }

        // 5. Log
        for v in mel_row.iter_mut() {
            *v = (*v + log_guard).ln();
        }

        mel_features.push(mel_row);
    }

    // 6. Per-feature normalization over the *valid* frames, matching the
    //    model's preprocessor:
    //      seq_len  = floor(n_samples / hop)
    //      mean/std over the first seq_len frames only, std with the
    //      sample estimator (N-1, NaN→0), then std += 1e-5
    //      frames >= seq_len are zeroed (pad_value) after normalization.
    //    torch.stft(center=True) yields 1 + floor(n/hop) frames, so the
    //    trailing frame is always outside the valid region.
    let eps = 1e-5f32;
    let valid_frames = (n / cfg.hop_length).min(n_frames);
    for row in mel_features.iter_mut() {
        if row.is_empty() {
            continue;
        }
        if valid_frames == 0 {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let v = valid_frames;
        let mean: f32 = row[..v].iter().sum::<f32>() / v as f32;
        let var: f32 = if v > 1 {
            row[..v].iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (v - 1) as f32
        } else {
            0.0 // single-frame case yields std 0 before the epsilon
        };
        // std = sqrt(var) (sample estimator with NaN→0), then +1e-5 before
        // dividing; this normalises every feature to unit variance.
        let std = var.sqrt() + eps;
        for (t, val) in row.iter_mut().enumerate() {
            *val = if t < v { (*val - mean) / std } else { 0.0 };
        }
    }

    MelFeatures {
        rows: mel_features,
        valid_frames,
    }
}

/// Symmetric (periodic=false) Hann window of length `win_length`.
fn hann_window(win_length: usize) -> Vec<f32> {
    (0..win_length)
        .map(|i| {
            let pi = std::f32::consts::PI;
            0.5 * (1.0 - (2.0 * pi * i as f32 / (win_length - 1) as f32).cos())
        })
        .collect()
}

/// Center-padded STFT power frames of a signal, framed exactly like
/// `torch.stft(center=True, pad_mode="constant", win_length, n_fft, hop)`:
/// the signal is zero-padded by n_fft/2 on both sides, and the (shorter)
/// analysis window is centered inside each n_fft buffer — buffer slot j
/// holds padded[start + j] * window_padded[j], i.e. window sample i
/// multiplies padded[start + offset + i] with offset = (n_fft-win)/2.
/// Returns `1 + floor(n/hop)` rows of `n_fft/2+1` power bins.
fn stft_power_frames(audio: &[f32], cfg: &MelConfig, hann: &[f32]) -> Vec<Vec<f32>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = audio.len();
    let pad = cfg.n_fft / 2;
    let mut padded = vec![0.0f32; n + 2 * pad];
    padded[pad..pad + n].copy_from_slice(audio);

    let padded_n = padded.len();
    let n_frames = 1 + padded_n.saturating_sub(cfg.n_fft) / cfg.hop_length;
    let n_freqs = cfg.n_fft / 2 + 1;
    let offset = (cfg.n_fft - cfg.win_length) / 2;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(cfg.n_fft);
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); cfg.n_fft];

    let mut power_spectrum: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
    for start in (0..padded_n).step_by(cfg.hop_length).take(n_frames) {
        for c in buf.iter_mut() {
            *c = Complex::default();
        }
        for (i, w) in hann.iter().enumerate() {
            let sig_pos = start + offset + i;
            if sig_pos < padded_n {
                buf[offset + i].re = padded[sig_pos] * w;
            }
        }
        fft.process(&mut buf);

        let mut frame_power = vec![0.0f32; n_freqs];
        for (k, p) in frame_power.iter_mut().enumerate() {
            let re = buf[k].re;
            let im = buf[k].im;
            *p = re * re + im * im;
        }
        power_spectrum.push(frame_power);
    }
    power_spectrum
}

/// Convert mel features Vec<Vec<f32>> of shape (n_mels, T) to a flat f32 Vec.
pub fn mel_to_tensor_data(mel: &[Vec<f32>]) -> (Vec<f32>, Vec<i64>) {
    let n_mels = mel.len();
    let n_frames = if n_mels > 0 { mel[0].len() } else { 0 };
    let mut flat = Vec::with_capacity(n_mels * n_frames);
    for row in mel {
        flat.extend_from_slice(row);
    }
    // Shape: (1, n_mels, n_frames) for batch_size=1
    (flat, vec![1, n_mels as i64, n_frames as i64])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_audio_yields_empty_features_not_panic() {
        let cfg = MelConfig::default();
        let mel = compute_mel_features(&[], &cfg);
        assert_eq!(mel.valid_frames, 0);
        assert_eq!(mel.rows.len(), cfg.n_mels);
        assert!(mel.rows.iter().all(|row| row.is_empty()));
    }

    #[test]
    fn filterbank_has_expected_shape_and_nonnegative_weights() {
        let cfg = MelConfig::default();
        let fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
        assert_eq!(fb.len(), cfg.n_mels * (cfg.n_fft / 2 + 1));
        assert!(fb.iter().all(|&w| w >= 0.0));
        assert!(fb.iter().any(|&w| w > 0.0));
    }

    #[test]
    fn filterbank_uses_slaney_scale_like_the_checkpoint() {
        // Discriminates Slaney from the HTK mel scale: for the first band
        // (edges 0 Hz and 46.77 Hz, center 23.38 Hz), bin 1 (31.25 Hz) is on
        // the falling edge under Slaney but *outside* the band under HTK.
        // The checkpoint's `preprocessor.featurizer.fb` has ≈0.0275 here
        // (bf16: 0.02832), exactly the Slaney value.
        let cfg = MelConfig::default();
        let fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
        let w = fb[1] as f64; // row 0, bin 1
        let expected = {
            // Slaney band geometry for band 0 with 128 bands over [0, 8000] Hz
            let f_sp = 200.0 / 3.0;
            let logstep = (6.4f64).ln() / 27.0;
            let hz_to_mel = |f: f64| {
                if f >= 1000.0 {
                    15.0 + (f / 1000.0).ln() / logstep
                } else {
                    f / f_sp
                }
            };
            let mel_max = hz_to_mel(8000.0);
            // Both band-edge mels are < 15, so mel_to_hz is the linear branch.
            let center = (mel_max / 129.0) * f_sp;
            let upper = (2.0 * mel_max / 129.0) * f_sp;
            let f = 16000.0 / 512.0; // bin 1
            let tri = (upper - f) / (upper - center);
            tri * 2.0 / upper
        };
        assert!(
            (w - expected).abs() < 1e-3,
            "row0/bin1 weight {w} should match Slaney {expected}"
        );
        // Cross-check against the checkpoint's own value for this weight
        // (`preprocessor.featurizer.fb` is 0.02838 in f32, 0.02832 in bf16).
        assert!(
            (w - 0.02832).abs() < 5e-4,
            "row0/bin1 {w} should match the checkpoint ≈0.02832"
        );
        // Under HTK this bin would be exactly 0 (band ends at 27.9 Hz).
        assert!(
            w > 0.015,
            "row0/bin1 must be inside the band (Slaney), got {w}"
        );
    }

    #[test]
    fn stft_window_is_centered_like_torch() {
        // An impulse at sample p lands in frame t with window index
        // p - (hop*t - n_fft/2 + (n_fft-win)/2); its DC power equals
        // hann[idx]^2, which pins the window's centered placement.
        let cfg = MelConfig::default();
        let hann = hann_window(cfg.win_length);
        let p = 1000usize;
        let mut audio = vec![0.0f32; cfg.sample_rate];
        audio[p] = 1.0;
        let frames = stft_power_frames(&audio, &cfg, &hann);
        let t = 7usize;
        let idx = p as i64
            - (cfg.hop_length as i64 * t as i64 - (cfg.n_fft / 2) as i64
                + ((cfg.n_fft - cfg.win_length) / 2) as i64);
        assert!(idx >= 0 && (idx as usize) < cfg.win_length);
        let want = hann[idx as usize].powi(2);
        let got = frames[t][0];
        assert!(
            (got - want).abs() < 1e-6,
            "frame {t} impulse power {got} should be hann[{idx}]^2 = {want}"
        );
        // A window applied without the centering offset would place the
        // impulse 56 samples later; assert the measured power is not that.
        let uncentered = hann[(idx + 56) as usize].powi(2);
        assert!(
            (got - uncentered).abs() > 1e-4,
            "impulse landed at the uncentered position"
        );
    }

    #[test]
    fn normalisation_masks_trailing_frame_and_uses_valid_stats() {
        let cfg = MelConfig::default();
        let audio: Vec<f32> = (0..cfg.sample_rate)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / cfg.sample_rate as f32).sin())
            .collect();
        let mel = compute_mel_features(&audio, &cfg);
        // torch.stft(center=True) produces 1 + n/hop frames; the model's
        // preprocessor considers only floor(n/hop) valid and zeroes the rest.
        let n_frames = mel.rows[0].len();
        assert_eq!(n_frames, 1 + cfg.sample_rate / cfg.hop_length);
        let valid = cfg.sample_rate / cfg.hop_length;
        assert_eq!(mel.valid_frames, valid);
        for row in &mel.rows {
            assert_eq!(
                row[valid], 0.0,
                "trailing frame must be masked to pad_value"
            );
        }
        // Valid region of varying rows stays ~zero-mean and unit-variance,
        // matching the model's per-feature normalization (std = sqrt(var) + eps).
        let mut varied = 0;
        for row in &mel.rows {
            let slice = &row[..valid];
            let mean: f64 = slice.iter().map(|&v| v as f64).sum::<f64>() / valid as f64;
            let var: f64 = slice
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / valid as f64;
            if var > 1e-6 {
                varied += 1;
                assert!(mean.abs() < 1e-3, "valid-region mean {mean} should be ~0");
                assert!(
                    (var - 1.0).abs() < 0.1,
                    "valid-region variance {var} should be ~1 (std normalization)"
                );
            }
        }
        assert!(varied > 0);
    }

    #[test]
    fn resample_returns_expected_length_and_preserves_tone() {
        // 0.5 s of 440 Hz at 44.1 kHz → 16 kHz.
        let src_sr = 44100;
        let n = src_sr / 2;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / src_sr as f32).sin())
            .collect();
        let out = resample(&input, src_sr, 16000).expect("resample");
        let expected = (n as f64 * 16000.0 / src_sr as f64).ceil() as usize;
        assert_eq!(out.len(), expected);
        // The middle of the output should still be a ~440 Hz tone: check
        // that it has significant energy and is not mostly zeros.
        let mid = &out[out.len() / 3..2 * out.len() / 3];
        let peak = mid.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(peak > 0.5, "resampled tone lost amplitude (peak {peak})");
    }

    #[test]
    fn resample_uses_ceil_length_like_librosa() {
        // 44101 samples at 44.1 kHz → 16000.36… output samples: librosa's
        // fix=True emits ceil (16001), not round (16000). The extra sample
        // matters because `valid_frames = floor(n_samples / hop)` flips when
        // the resampled length crosses a multiple of the hop.
        let n = 44101usize;
        let out = resample(&vec![0.0f32; n], 44100, 16000).expect("resample");
        let ceil = (n as f64 * 16000.0 / 44100.0).ceil() as usize;
        let round = (n as f64 * 16000.0 / 44100.0).round() as usize;
        assert_ne!(ceil, round, "test input should distinguish ceil from round");
        assert_eq!(out.len(), ceil);
    }

    #[test]
    fn resample_flush_does_not_overgenerate() {
        // Flushing must finish after a small, bounded number of extra chunks;
        // an unbounded zero-feed would keep generating output.
        let src_sr = 48000;
        let n = src_sr * 5; // 5 s
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / src_sr as f32).sin())
            .collect();
        let (out, generated) = resample_impl(&input, src_sr, 16000).expect("resample");
        let expected = (n as f64 * 16000.0 / src_sr as f64).ceil() as usize;
        assert_eq!(out.len(), expected);
        // Allow the warm-up delay plus a couple of max-size chunks of tail;
        // the bound rejects runaway flushing.
        let max_chunk_out = 4096; // generous upper bound for one process() call
        assert!(
            generated <= expected + 2 * max_chunk_out + 4096,
            "flush over-generated: {generated} samples for {expected} expected"
        );
    }

    #[test]
    fn mel_features_are_per_row_normalised() {
        // A sine wave has genuine spectral variance, so per-row normalisation
        // must yield ~zero-mean rows. (Pure silence is a degenerate constant
        // input where float summation noise dominates; it is covered by the
        // empty-audio test above instead.)
        let cfg = MelConfig::default();
        let audio: Vec<f32> = (0..cfg.sample_rate)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / cfg.sample_rate as f32).sin())
            .collect();
        let mel = compute_mel_features(&audio, &cfg);
        assert_eq!(mel.rows.len(), cfg.n_mels);
        assert!(!mel.rows[0].is_empty());
        // Bands with no signal energy are constant (log-guard floor); only
        // rows carrying variance must come out ~zero-mean.
        let mut varied_rows = 0;
        for row in &mel.rows {
            let n = row.len() as f64;
            let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            if var > 1e-6 {
                varied_rows += 1;
                assert!(
                    mean.abs() < 1e-3,
                    "normalised row should have ~zero mean, got {mean}"
                );
            }
        }
        assert!(
            varied_rows > 0,
            "sine input should produce variance in at least one mel band"
        );
    }

    #[test]
    fn tensor_data_shape_matches_features() {
        let mel = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (flat, shape) = mel_to_tensor_data(&mel);
        assert_eq!(shape, vec![1, 3, 2]);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn unsupported_codec_message_mentions_opus_and_conversion() {
        let opus = unsupported_codec_message(symphonia::core::codecs::CODEC_TYPE_OPUS);
        assert!(opus.contains("Opus"), "{opus}");
        assert!(opus.contains("Supported:"), "{opus}");
        assert!(opus.contains("yt-dlp"), "{opus}");

        let other = unsupported_codec_message(symphonia::core::codecs::CODEC_TYPE_NULL);
        assert!(other.contains("Audio codec is not supported"), "{other}");
        assert!(other.contains("ffmpeg"), "{other}");
    }

    #[test]
    fn unsupported_file_error_includes_conversion_hint() {
        // A file that cannot be probed must produce an actionable error, not
        // just "unsupported".
        let path = std::env::temp_dir().join(format!(
            "cohere-transcriber-test-{}.webm",
            std::process::id()
        ));
        std::fs::write(&path, b"this is not a media file").unwrap();
        let err = load_audio(&path, 16000).expect_err("garbage input must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("Supported:"), "hint missing from error: {msg}");
        assert!(
            msg.contains("yt-dlp") || msg.contains("ffmpeg"),
            "conversion hint missing from error: {msg}"
        );
        std::fs::remove_file(&path).ok();
    }

    fn write_pcm16_wav(path: &std::path::Path, interleaved: &[i16], channels: u16, rate: u32) {
        let data_bytes = interleaved.len() * 2;
        let block = u32::from(channels) * 2;
        let mut bytes = Vec::with_capacity(44 + data_bytes);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * block).to_le_bytes());
        bytes.extend_from_slice(&(block as u16).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for sample in interleaved {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn loads_pcm_wav_scales_samples_and_downmixes_to_mono() {
        let path = std::env::temp_dir().join(format!("ct-wav-{}-mono.wav", std::process::id()));
        // 16384 / 32768 = 0.5.
        write_pcm16_wav(&path, &[0, 16384, -16384], 1, 16000);
        let samples = load_audio(&path, 16000).unwrap();
        assert_eq!(samples.len(), 3);
        assert!(samples[0].abs() < 1e-4, "{}", samples[0]);
        assert!((samples[1] - 0.5).abs() < 1e-3, "{}", samples[1]);
        assert!((samples[2] + 0.5).abs() < 1e-3, "{}", samples[2]);

        let stereo = std::env::temp_dir().join(format!("ct-wav-{}-stereo.wav", std::process::id()));
        // Interleaved L/R. The average of 16384 and -16384 is silence.
        write_pcm16_wav(&stereo, &[16384, -16384, 0, 0], 2, 16000);
        let mixed = load_audio(&stereo, 16000).unwrap();
        assert_eq!(mixed.len(), 2);
        assert!(mixed[0].abs() < 1e-3, "{}", mixed[0]);
        assert!(mixed[1].abs() < 1e-4, "{}", mixed[1]);
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&stereo).ok();
    }

    #[test]
    fn loads_and_resamples_a_higher_rate_wav() {
        let path = std::env::temp_dir().join(format!("ct-wav-{}-44k.wav", std::process::id()));
        let n = 4410;
        let tone: Vec<i16> = (0..n)
            .map(|i| {
                let x = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
                (x * 16000.0) as i16
            })
            .collect();
        write_pcm16_wav(&path, &tone, 1, 44100);
        let samples = load_audio(&path, 16000).unwrap();
        let expected = (n as f64 * 16000.0 / 44100.0).ceil() as usize;
        assert_eq!(samples.len(), expected);
        let peak = samples.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(peak > 0.1, "resampled wav lost the tone (peak {peak})");
        std::fs::remove_file(&path).ok();
    }
}
