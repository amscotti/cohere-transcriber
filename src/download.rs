//! Model download from Hugging Face via `hf-hub` — no Python required.
//!
//! Token resolution order: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, then the
//! `token` file written by `huggingface-cli login` (looked up under
//! `$HF_HOME`, `$XDG_CACHE_HOME/huggingface`, or `~/.cache/huggingface`).
//! The blob cache honours `HF_HOME`/`HF_ENDPOINT` via
//! `ApiBuilder::from_env`.

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::path::{Component, Path, PathBuf};

use crate::VOCAB_JSON;

/// The only files the binary reads at inference time (plus a few small
/// metadata files kept for completeness/compat). Everything else in the
/// repo — README, plots, demo audio, eval artifacts — is skipped, which
/// also keeps junk (and nested-directory surprises) out of the model dir.
/// A repo-provided `vocab.json` wins over the embedded copy (the embedded
/// table is only written when the repo doesn't ship one).
const ALLOWED_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.model",
    "tokenizer.json",
    "vocab.json",
    "preprocessor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
];

fn hf_cache_home() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HF_HOME") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir).join("huggingface"));
        }
    }
    let home = std::env::var("HOME").ok()?;
    Some(Path::new(&home).join(".cache/huggingface"))
}

fn resolve_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.trim().is_empty() {
            return Some(t.trim().to_string());
        }
    }
    // Covers `HF_TOKEN` prefixed variants some setups export.
    if let Ok(t) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        if !t.trim().is_empty() {
            return Some(t.trim().to_string());
        }
    }
    let token_file = hf_cache_home()?.join("token");
    std::fs::read_to_string(token_file)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// A repo-relative name is safe to place under the destination only if every
/// component is a normal one (no `..`, no absolute paths, no root) and it
/// contains no drive-letter/colon or backslash forms.
fn is_safe_relative_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\\')
        && !name.contains(':')
        && std::path::Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// The actionable hint for gated-model failures.
fn gated_hint(model_id: &str) -> String {
    format!(
        "This model is gated on Hugging Face — make sure you have accepted the \
         license at https://huggingface.co/{model_id} and set HF_TOKEN (or run \
         `huggingface-cli login` once)."
    )
}

pub fn download_model(model_id: &str, dest: &Path, refresh: bool) -> Result<()> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("Cannot create model dir {}", dest.display()))?;

    // from_env() honours HF_HOME (cache location) and HF_ENDPOINT (mirrors);
    // the explicit token (which also covers the token *file*) wins over any
    // env token from_env may have picked up.
    let mut builder = ApiBuilder::from_env();
    if let Some(token) = resolve_token() {
        builder = builder.with_token(Some(token));
    }
    let api: Api = builder
        .build()
        .context("Failed to build Hugging Face API client")?;

    let revision = std::env::var("COHERE_MODEL_REVISION").unwrap_or_else(|_| "main".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision,
    ));

    let info = repo.info().with_context(|| {
        format!(
            "Failed to list files for '{model_id}'. {}",
            gated_hint(model_id)
        )
    })?;

    for file in &info.siblings {
        let name = &file.rfilename;
        if !ALLOWED_FILES.contains(&name.as_str()) {
            tracing::debug!("Skipping repo file not needed at runtime: {name}");
            continue;
        }
        if !is_safe_relative_name(name) {
            anyhow::bail!("Refusing unsafe file name from repo metadata: {name:?}");
        }
        eprintln!("  downloading {name}...");
        // `get` serves an already-cached blob without touching the network;
        // `download` re-fetches (used when the caller explicitly asked for a
        // refresh, e.g. --download-only).
        let cached = if refresh {
            repo.download(name)
        } else {
            repo.get(name)
        }
        .with_context(|| format!("Failed to download {name}. {}", gated_hint(model_id)))?;
        let out = dest.join(name);
        copy_atomic(&cached, &out).with_context(|| format!("Failed to write {}", out.display()))?;
    }

    // vocab.json is not published on HF — it is embedded in this binary
    // (no Python/sentencepiece needed at runtime). Only write it when the
    // repo didn't provide one, so a mirror shipping a newer table wins.
    // Written atomically so an interrupted write cannot leave a corrupt table
    // that then gets picked up by the loader.
    if !dest.join("vocab.json").exists() {
        let out = dest.join("vocab.json");
        let mut part = out.clone().into_os_string();
        part.push(format!(".{}.part", std::process::id()));
        let part = PathBuf::from(part);
        std::fs::write(&part, VOCAB_JSON)
            .with_context(|| format!("Failed to write {}", part.display()))?;
        std::fs::rename(&part, &out).with_context(|| "Failed to move vocab.json into place")?;
    }

    eprintln!("Model ready at {}", dest.display());
    Ok(())
}

/// Copy `src` to `dst` atomically: copy to a `.part` sibling, verify the
/// copied length, then rename over the destination. An interrupted copy
/// therefore cannot leave a truncated file that still passes the
/// existence-only validation.
fn copy_atomic(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Cannot create {}", parent.display()))?;
    }
    // Refuse to copy a file onto itself (e.g. the destination is a symlink
    // into the hf-hub cache we are reading from): fs::copy truncates the
    // destination first, which would destroy the cached blob.
    let src_canon = std::fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf());
    let dst_canon = std::fs::canonicalize(dst).unwrap_or_else(|_| dst.to_path_buf());
    if src_canon == dst_canon {
        tracing::debug!(
            "{} already is {}; skipping copy",
            src.display(),
            dst.display()
        );
        return Ok(());
    }

    let mut part = dst.as_os_str().to_os_string();
    // Per-process suffix: two concurrent first runs must not fight over the
    // same temporary file (one would fail its size check or rename).
    part.push(format!(".{}.part", std::process::id()));
    let part = PathBuf::from(part);
    std::fs::copy(src, &part)
        .with_context(|| format!("Cannot copy {} to {}", src.display(), part.display()))?;
    let expected = std::fs::metadata(src)
        .with_context(|| format!("Cannot stat {}", src.display()))?
        .len();
    let got = std::fs::metadata(&part)
        .with_context(|| format!("Cannot stat {}", part.display()))?
        .len();
    anyhow::ensure!(
        got == expected,
        "Copied {} is {} bytes but the source is {expected}",
        part.display(),
        got
    );
    std::fs::rename(&part, dst).with_context(|| {
        format!(
            "Cannot move {} into place (was the file removed mid-copy?)",
            part.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_names_reject_traversal_and_absolute_paths() {
        assert!(is_safe_relative_name("model.safetensors"));
        assert!(is_safe_relative_name("sub/dir/file.bin"));
        assert!(!is_safe_relative_name(""));
        assert!(!is_safe_relative_name("../escape"));
        assert!(!is_safe_relative_name("a/../../escape"));
        assert!(!is_safe_relative_name("/etc/passwd"));
        assert!(!is_safe_relative_name("."));
        assert!(!is_safe_relative_name(".."));
        assert!(!is_safe_relative_name("C:\\win"));
        assert!(!is_safe_relative_name("C:/win"));
        assert!(!is_safe_relative_name("a:b"));
    }

    #[test]
    fn atomic_copy_writes_full_file() {
        let dir = std::env::temp_dir().join(format!("ct-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.bin");
        let dst = dir.join("dst.bin");
        std::fs::write(&src, vec![7u8; 100_000]).unwrap();
        copy_atomic(&src, &dst).unwrap();
        assert_eq!(std::fs::metadata(&dst).unwrap().len(), 100_000);
        // No per-process .part leftover.
        let leftover = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains(".part"));
        assert!(!leftover, "unexpected .part file left behind");
        // Self-copy is a no-op that must not truncate the source.
        copy_atomic(&src, &src).unwrap();
        assert_eq!(std::fs::metadata(&src).unwrap().len(), 100_000);
        std::fs::remove_dir_all(&dir).ok();
    }
}
