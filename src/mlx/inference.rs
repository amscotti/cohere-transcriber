//! Greedy decoding loop: encode → step the decoder → detokenize.

use anyhow::Result;

use super::array::Array;
use super::decoder::{DecoderKvCache, TransformerDecoder};
use super::encoder::ConformerEncoder;
use crate::tokenizer::Tokenizer;

/// One chunk of mel features and how many of its time steps are real.
/// The steps at `valid_frames..` are the centered-STFT pad.
pub struct MelInput<'a> {
    pub features: &'a Array,
    pub valid_frames: i32,
}

/// Run a full transcription: encode → greedy decode → detokenize.
pub fn transcribe(
    mel: MelInput<'_>,
    encoder: &ConformerEncoder,
    decoder: &TransformerDecoder,
    tokenizer: &Tokenizer,
    language: &str,
    punctuation: bool,
    max_new_tokens: usize,
) -> Result<String> {
    // 1. Encode. `valid_len` is the number of encoder steps that correspond
    // to real audio; anything past it is the centered-STFT pad frame after
    // subsampling and must not be cross-attended.
    tracing::debug!("Running MLX encoder...");
    let (encoder_hs, valid_len) = encoder.forward(mel.features, mel.valid_frames);
    tracing::debug!(
        "Encoder output shape: {:?} (valid steps {valid_len})",
        encoder_hs.shape()
    );
    if valid_len <= 0 {
        return Ok(String::new());
    }

    // Force evaluation before cross-attention pre-computation. `eval()` only
    // enqueues the graph, so block until the encoder output is materialised —
    // the per-step KV cache that follows must observe complete values.
    encoder_hs.eval();
    super::stream::synchronize();

    let enc_len = encoder_hs.dim(1);
    let cross_mask =
        (valid_len < enc_len).then(|| TransformerDecoder::key_padding_mask(enc_len, valid_len));

    // 2. Pre-compute cross-attention K/V — reused every decoder step
    let cross_kv = decoder.precompute_cross_kv(&encoder_hs);

    // 3. Build prompt token IDs
    let prompt = tokenizer.special.build_prompt(language, punctuation)?;
    let n_prompt = prompt.len();
    tracing::debug!("Prompt IDs: {:?}", prompt);

    // Position ids index a fixed [max_sequence_length, hidden] table; the
    // prompt plus everything generated must stay inside it. MLX's gather
    // does not bounds-check, so enforce it here rather than read garbage.
    let max_positions = decoder.max_positions() as usize;
    anyhow::ensure!(
        max_new_tokens.saturating_add(n_prompt) <= max_positions,
        "--max-tokens {max_new_tokens} + {n_prompt} prompt tokens exceeds the model's \
         position table ({max_positions} rows); use a smaller value"
    );

    // 4. Initialize self-attention KV cache (empty)
    let mut self_kv_cache: DecoderKvCache =
        (0..decoder.layers.len()).map(|_| (None, None)).collect();

    // 5. Prime decoder with prompt tokens
    let mut next_token = 0i32;
    for (i, &token_id) in prompt.iter().enumerate() {
        let (token, new_kv) = decoder.step(
            token_id as i32,
            i as i32,
            &self_kv_cache,
            &cross_kv,
            cross_mask.as_ref(),
        );
        self_kv_cache = new_kv;
        next_token = token;
    }

    // 6. Greedy decode until EOS or max_new_tokens.
    //    Argmax is computed on GPU inside decoder.step() — only a single i32
    //    is transferred per step instead of the full-vocabulary logits vector.
    //    Like HF `generate` with eos_token_id, only EOS ends generation; a
    //    mid-stream `<|nospeech|>` is simply skipped during detokenization
    //    like any other special token.
    let eos_id = tokenizer.special.eos as i32;
    let mut generated: Vec<i64> = Vec::new();
    let mut position = n_prompt as i32;

    while generated.len() < max_new_tokens {
        if next_token == eos_id {
            break;
        }
        generated.push(next_token as i64);
        if generated.len() == max_new_tokens {
            tracing::warn!(
                "stopped at the --max-tokens cap ({max_new_tokens}) before EOS; \
                 this segment's transcript may be truncated"
            );
            break;
        }
        let (token, new_kv) = decoder.step(
            next_token,
            position,
            &self_kv_cache,
            &cross_kv,
            cross_mask.as_ref(),
        );
        self_kv_cache = new_kv;
        next_token = token;
        position += 1;
    }

    tracing::debug!("Generated token IDs: {:?}", generated);

    // 7. Decode tokens to text (stripped, matching the model's transcripts).
    Ok(tokenizer.decode(&generated).trim().to_string())
}
