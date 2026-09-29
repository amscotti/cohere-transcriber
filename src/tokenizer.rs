use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Special token IDs derived from the model's tokenizer config: exactly the
/// IDs the decoder prompt needs.
pub struct SpecialTokens {
    pub eos: i64,
    pub startoftranscript: i64,
    pub pnc: i64,
    pub nopnc: i64,
    pub startofcontext: i64,
    pub noitn: i64,
    pub notimestamp: i64,
    pub nodiarize: i64,
    pub emo_undefined: i64,
    pub lang_ids: HashMap<String, i64>,
    pub special_ids: HashSet<i64>,
}

impl SpecialTokens {
    pub fn from_tokenizer_config(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer_config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read tokenizer_config.json at {path:?}"))?;

        #[derive(Deserialize)]
        struct TokenEntry {
            content: String,
            special: Option<bool>,
        }
        #[derive(Deserialize)]
        struct TokenizerConfig {
            added_tokens_decoder: HashMap<String, TokenEntry>,
        }

        let cfg: TokenizerConfig =
            serde_json::from_str(&content).context("Failed to parse tokenizer_config.json")?;

        let mut token_to_id: HashMap<String, i64> = HashMap::new();
        let mut special_ids = HashSet::new();

        for (id_str, entry) in &cfg.added_tokens_decoder {
            let id: i64 = id_str.parse().context("Invalid token ID")?;
            token_to_id.insert(entry.content.clone(), id);
            if entry.special.unwrap_or(false) {
                special_ids.insert(id);
            }
        }

        let get = |name: &str| -> Result<i64> {
            token_to_id
                .get(name)
                .copied()
                .with_context(|| format!("Special token '{name}' not found"))
        };

        // Language tokens look like `<|en|>`: collect every added token of
        // exactly that shape (two lowercase letters) so new model languages
        // work without a code change. The exact length matters: control
        // tokens such as `<|pnc|>` share the `<|…|>` brackets but are longer.
        let mut lang_ids = HashMap::new();
        for (content, id) in &token_to_id {
            if content.len() == 6 && content.starts_with("<|") && content.ends_with("|>") {
                let code = &content[2..content.len() - 2];
                if code.len() == 2 && code.chars().all(|c| c.is_ascii_lowercase()) {
                    lang_ids.insert(code.to_string(), *id);
                    special_ids.insert(*id);
                }
            }
        }

        Ok(Self {
            // Every ID below feeds the decoder prompt, so a missing entry
            // is a hard error rather than a silent wrong ID.
            eos: get("<|endoftext|>")?,
            startoftranscript: get("<|startoftranscript|>")?,
            pnc: get("<|pnc|>")?,
            nopnc: get("<|nopnc|>")?,
            startofcontext: get("<|startofcontext|>")?,
            noitn: get("<|noitn|>")?,
            notimestamp: get("<|notimestamp|>")?,
            nodiarize: get("<|nodiarize|>")?,
            emo_undefined: get("<|emo:undefined|>")?,
            lang_ids,
            special_ids,
        })
    }

    /// Build the decoder prompt token IDs for the given language and punctuation setting.
    ///
    /// Prompt format:
    ///   <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
    ///   <|{lang}|> <|{lang}|> <|pnc|or|nopnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    pub fn build_prompt(&self, language: &str, punctuation: bool) -> Result<Vec<i64>> {
        let lang_id = self
            .lang_ids
            .get(language)
            .copied()
            .with_context(|| format!("Unsupported language: '{language}'"))?;
        let pnc_id = if punctuation { self.pnc } else { self.nopnc };
        Ok(vec![
            self.startofcontext,
            self.startoftranscript,
            self.emo_undefined,
            lang_id,
            lang_id,
            pnc_id,
            self.noitn,
            self.notimestamp,
            self.nodiarize,
        ])
    }
}

/// Vocabulary mapping token ID → piece string.
///
/// The table normally lives at `<model_dir>/vocab.json` (written there on
/// download from the copy embedded in this binary). Use [`Vocab::load`] for
/// the file or [`Vocab::from_json_str`] for an in-memory JSON document
/// (e.g. the embedded copy).
pub struct Vocab {
    pub id_to_piece: HashMap<i64, String>,
}

impl Vocab {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("vocab.json");
        let content = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "Cannot read vocab.json at {path:?}. \
                 Re-run with a complete model directory (it is written automatically on download)."
            )
        })?;
        Self::from_json_str(&content).context("Failed to parse vocab.json")
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let raw: HashMap<String, String> =
            serde_json::from_str(content).context("Failed to parse vocab.json")?;
        let id_to_piece = raw
            .into_iter()
            .map(|(k, v)| -> Result<(i64, String)> { Ok((k.parse()?, v)) })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { id_to_piece })
    }
}

pub struct Tokenizer {
    vocab: Vocab,
    pub special: SpecialTokens,
    /// Set once an unknown token id has been seen during decoding, so the
    /// warning (which usually indicates a stale vocab.json) fires once
    /// instead of per token.
    warned_unknown_id: std::sync::atomic::AtomicBool,
}

/// Append buffered `<0xXX>` bytes to `out` as UTF-8 (lossy on invalid
/// sequences), then clear the buffer.
fn flush_pending_bytes(out: &mut String, pending: &mut Vec<u8>) {
    if pending.is_empty() {
        return;
    }
    out.push_str(&String::from_utf8_lossy(pending));
    pending.clear();
}

impl Tokenizer {
    /// Build from an already-loaded [`Vocab`] plus the tokenizer config on disk.
    /// Used for the embedded-vocabulary fallback when the model directory has
    /// no usable `vocab.json`.
    pub fn from_vocab_and_config_dir(vocab: Vocab, model_dir: impl AsRef<Path>) -> Result<Self> {
        let special = SpecialTokens::from_tokenizer_config(model_dir.as_ref())?;
        Ok(Self {
            vocab,
            special,
            warned_unknown_id: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[cfg(test)]
    pub(crate) fn from_parts_for_test(vocab: Vocab, special: SpecialTokens) -> Self {
        Self {
            vocab,
            special,
            warned_unknown_id: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Decode a list of token IDs to a string, skipping all special tokens.
    /// Handles SentencePiece ▁ (U+2581) word-boundary markers and <0xXX> byte tokens.
    ///
    /// Byte tokens are accumulated and decoded as UTF-8 (with lossy fallback)
    /// so multibyte sequences (e.g. CJK) round-trip correctly.
    pub fn decode(&self, ids: &[i64]) -> String {
        let skip = &self.special.special_ids;

        let mut result = String::new();
        let mut pending_bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if skip.contains(&id) {
                continue;
            }
            let piece = match self.vocab.id_to_piece.get(&id) {
                Some(p) => p.as_str(),
                None => {
                    // An id outside the table usually means a stale/mismatched
                    // vocab.json; say so once instead of silently dropping.
                    if !self
                        .warned_unknown_id
                        .swap(true, std::sync::atomic::Ordering::Relaxed)
                    {
                        tracing::warn!(
                            "token id {id} is not in vocab.json and will be skipped \
                             (stale vocabulary?)"
                        );
                    } else {
                        tracing::debug!("skipping unknown token id {id}");
                    }
                    continue;
                }
            };

            if piece.starts_with("<0x") && piece.ends_with('>') {
                // Raw byte token (e.g. <0xE3>) — buffer for UTF-8 decoding.
                if let Ok(byte) = u8::from_str_radix(&piece[3..piece.len() - 1], 16) {
                    pending_bytes.push(byte);
                }
                continue;
            }

            // A non-byte piece ends any pending byte run.
            flush_pending_bytes(&mut result, &mut pending_bytes);

            if piece.starts_with('\u{2581}') {
                // ▁ marks a word boundary — replace with space
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(&piece['\u{2581}'.len_utf8()..]);
            } else if piece == "<unk>" || piece == "<s>" || piece == "</s>" {
                // Skip sentence-piece meta tokens
            } else {
                result.push_str(piece);
            }
        }
        flush_pending_bytes(&mut result, &mut pending_bytes);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer(pairs: Vec<(i64, &str)>, special_ids: Vec<i64>) -> Tokenizer {
        let vocab = Vocab {
            id_to_piece: pairs
                .into_iter()
                .map(|(id, piece)| (id, piece.to_string()))
                .collect(),
        };
        let special = SpecialTokens {
            eos: 3,
            startoftranscript: 4,
            pnc: 5,
            nopnc: 6,
            startofcontext: 7,
            noitn: 9,
            notimestamp: 11,
            nodiarize: 13,
            emo_undefined: 16,
            lang_ids: HashMap::new(),
            special_ids: special_ids.into_iter().collect(),
        };
        Tokenizer::from_parts_for_test(vocab, special)
    }

    #[test]
    fn decodes_word_boundaries() {
        // Only ▁-initial pieces start a new word; a bare piece continues it.
        let tok = test_tokenizer(
            vec![(10, "\u{2581}hello"), (11, "\u{2581}world"), (12, "world")],
            vec![],
        );
        assert_eq!(tok.decode(&[10, 11]), "hello world");
        assert_eq!(tok.decode(&[10, 12]), "helloworld");
    }

    #[test]
    fn skips_special_tokens() {
        let tok = test_tokenizer(vec![(10, "\u{2581}hi"), (99, "<|endoftext|>")], vec![99]);
        assert_eq!(tok.decode(&[10, 99]), "hi");
    }

    #[test]
    fn decodes_multibyte_utf8_from_byte_tokens() {
        // "あ" (U+3042) is UTF-8 E3 81 82.
        let tok = test_tokenizer(
            vec![
                (20, "<0xE3>"),
                (21, "<0x81>"),
                (22, "<0x82>"),
                (23, "\u{2581}hello"),
            ],
            vec![],
        );
        assert_eq!(tok.decode(&[20, 21, 22]), "あ");
        // Byte run followed by a word piece flushes before the space logic.
        assert_eq!(tok.decode(&[20, 21, 22, 23]), "あ hello");
    }

    #[test]
    fn decodes_ascii_byte_token() {
        let tok = test_tokenizer(vec![(30, "<0x41>")], vec![]);
        assert_eq!(tok.decode(&[30]), "A");
    }

    #[test]
    fn invalid_byte_sequence_is_lossy_not_mojibake() {
        // 0xFF alone is invalid UTF-8; lossy decoding yields U+FFFD, and must
        // not yield 'ÿ' (U+00FF, the old `byte as char` behaviour).
        let tok = test_tokenizer(vec![(31, "<0xFF>")], vec![]);
        assert_eq!(tok.decode(&[31]), "�");
    }

    #[test]
    fn vocab_from_json_str_rejects_bad_ids() {
        assert!(Vocab::from_json_str(r#"{"abc": "hi"}"#).is_err());
        let vocab = Vocab::from_json_str(r#"{"10": "▁hi"}"#).unwrap();
        assert_eq!(vocab.id_to_piece.get(&10).unwrap(), "▁hi");
    }

    #[test]
    fn skips_unknown_ids_and_sentencepiece_meta_tokens() {
        let tok = test_tokenizer(
            vec![(10, "\u{2581}hi"), (11, "<unk>"), (12, "</s>")],
            vec![],
        );
        assert_eq!(tok.decode(&[10, 11, 12, 99]), "hi");
        // The second unknown id still drops out, after the one-time warning.
        assert_eq!(tok.decode(&[99, 10]), "hi");
    }

    #[test]
    fn invalid_hex_byte_token_is_dropped() {
        let tok = test_tokenizer(vec![(1, "<0xZZ>")], vec![]);
        assert_eq!(tok.decode(&[1]), "");
    }

    fn token_config() -> String {
        r#"{
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
                "10": {"content": "<|en|>", "special": true},
                "11": {"content": "<|fr|>", "special": true}
            }
        }"#
        .to_string()
    }

    #[test]
    fn tokenizer_config_builds_the_prompt_and_language_ids() {
        let dir = std::env::temp_dir().join(format!(
            "ct-tok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), token_config()).unwrap();
        let special = SpecialTokens::from_tokenizer_config(&dir).unwrap();
        assert_eq!(special.eos, 1);
        assert_eq!(special.lang_ids.get("en"), Some(&10));
        assert_eq!(special.lang_ids.get("fr"), Some(&11));
        // Control tokens share the brackets but are not two-letter languages.
        assert!(!special.lang_ids.values().any(|id| *id == 5));
        assert_eq!(special.build_prompt("en", true).unwrap().len(), 9);
        assert_eq!(special.build_prompt("en", true).unwrap()[5], 5);
        assert_eq!(special.build_prompt("fr", false).unwrap()[5], 6);
        assert!(special.build_prompt("de", true).is_err());

        std::fs::write(dir.join("vocab.json"), r#"{"10": "▁hi"}"#).unwrap();
        let vocab = Vocab::load(&dir).unwrap();
        assert_eq!(vocab.id_to_piece.get(&10).unwrap(), "▁hi");
        let missing = dir.join("absent");
        std::fs::create_dir_all(&missing).unwrap();
        assert!(Vocab::load(&missing).is_err());

        // A config missing a prompt token is a hard error.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {"10": {"content": "<|en|>", "special": true}}}"#,
        )
        .unwrap();
        assert!(SpecialTokens::from_tokenizer_config(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
