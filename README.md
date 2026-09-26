# cohere-transcriber

Transcribe speech to text on an Apple Silicon Mac. One native binary runs
[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
on the GPU. It downloads the model on first use. No Python and no ffmpeg.

```bash
cohere-transcribe recording.wav
```

The transcript is printed to stdout. Progress and warnings go to stderr.

## Requirements

- Apple Silicon Mac, macOS 14 or newer
- About 16 GB of memory. The weights are held in memory at about 8 GB, and
  loading peaks a little higher.
- To build: Rust 1.88 or newer, CMake, and the Xcode command line tools

## Download

Publishing a GitHub release builds
`cohere-transcribe-<tag>-aarch64-apple-darwin.zip` for Apple Silicon and
attaches it to that release. Unpack it and leave `cohere-transcribe` and
`mlx.metallib` in the same folder. The zip appears a few minutes after the
release is published, once the build finishes.

## Build

```bash
cargo build --release
```

The first build compiles MLX and takes several minutes. It needs a network
connection. The command produces two files, and they have to stay side by side:

- `target/release/cohere-transcribe`
- `target/release/mlx.metallib`

Copying the binary alone fails at startup with a Metal error. `cargo install`
does that, so run the binary from `target/release/` or copy both files together.

## Model

The weights are gated. Before the first run:

1. Accept the license at <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>
2. Set `HF_TOKEN`, or run `huggingface-cli login` once

The model is cached under `$XDG_CACHE_HOME/cohere-transcriber/` (or
`~/.cache/cohere-transcriber/` when that variable is unset). The download is
about 3.8 GB, and the Hugging Face cache keeps another copy, so plan on
about 7.6 GB of disk. Set `HF_HOME` to put the cache somewhere else.

```bash
cohere-transcribe --download-only   # fetch or refresh the model, then exit
```

## Usage

```bash
cohere-transcribe recording.wav
cohere-transcribe -l fr audio.mp3
cohere-transcribe --no-punctuation audio.wav
cohere-transcribe call1.wav call2.flac   # prints the filename before each transcript
cohere-transcribe --help
```

Languages: `en`, `fr`, `de`, `es`, `it`, `pt`, `nl`, `pl`, `el`, `ar`, `ja`,
`zh`, `vi`, `ko`. Pass the one you want with `-l` (default `en`).

Audio can be WAV, FLAC, MP3, AAC (`.aac` or `.m4a`), or OGG Vorbis, converted
to 16 kHz mono. Opus, including the WebM files `yt-dlp` downloads from
YouTube, is not supported. Convert it first:

```bash
yt-dlp -x --audio-format m4a <url>
ffmpeg -i input.webm -ar 16000 -ac 1 out.wav
```

Recordings longer than the model's window (about 35 seconds) are split at
quiet points and joined back together. If one file fails, the command stops
and exits non-zero.

`-v` and `-vv` print more detail on stderr. `--max-tokens` caps how many
tokens are generated for each chunk (default 448). The model's position table
holds 1024 tokens including the prompt, and larger values are rejected.

## Options

| Flag | Meaning |
| --- | --- |
| `-m`, `--model-dir` | Directory for the weights. Created and filled on first use. |
| `-l`, `--language` | Language code. Default `en`. |
| `--no-punctuation` | Leave punctuation out of the transcript. |
| `--max-tokens` | Token cap per chunk. Default 448. |
| `--download-only` | Download or refresh the model and exit. |
| `-v` | More logging. Repeat for debug. |

Environment variables:

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` — Hugging Face access token
- `HF_HOME`, `HF_ENDPOINT` — cache location and download mirror
- `COHERE_MODEL_ID` — a different repo, such as a mirror of the weights
- `COHERE_MODEL_REVISION` — git revision to download (default `main`)

## Development

GitHub Actions runs these checks on macOS 26 (Apple Silicon) with Rust 1.88:

```bash
cargo fmt --check
cargo clippy --release --bin cohere-transcribe --tests -- -D warnings
cargo test --release --bin cohere-transcribe
```

Tests run without downloading the model. They link MLX, so they build on an
Apple Silicon Mac.

`mlx-c/` is an unmodified copy of [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c)
v0.6.0. Its build downloads MLX v0.31.1.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Portions of the engine are derived from
[second-state/cohere_transcribe_rs](https://github.com/second-state/cohere_transcribe_rs)
(Copyright 2026 Second State Inc.). The model weights have their own
license; see the
[model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).
The binary statically links MLX and mlx-c (both MIT). Attributions are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
