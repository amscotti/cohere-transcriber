# Third-party notices

This project is licensed under the Apache License, Version 2.0
(see [LICENSE](LICENSE) and [NOTICE](NOTICE)). It statically links the
libraries below, and portions of its source are derived from the project
listed under "Engine code".

## Engine code

- Portions of the engine are derived from
  [second-state/cohere_transcribe_rs](https://github.com/second-state/cohere_transcribe_rs)
  (Apache-2.0, Copyright 2026 Second State Inc.), which this project renames,
  restructures and extends.

## Apple MLX (ml-explore/mlx)

- License: MIT — https://github.com/ml-explore/mlx/blob/main/LICENSE
- Fetched at build time by mlx-c's CMake (pinned tag v0.31.1) and linked
  statically into the `cohere-transcribe` binary, including the compiled
  Metal shader library (`mlx.metallib`) shipped alongside the binary.
- MLX's own build also pulls in transitive dependencies (fmt, nlohmann/json,
  gguflib, metal-cpp, …); see the MLX repository for their bundled license
  files (metal-cpp is Apache-2.0; the others are MIT/BSD-style).

## mlx-c (ml-explore/mlx-c)

- License: MIT — https://github.com/ml-explore/mlx-c/blob/main/LICENSE
- Vendored, unmodified, at `mlx-c/` from tag v0.6.0
  (commit `0726ca922fc902c4c61ef9c27d94132be418e945`).
- Built via CMake. That build fetches MLX v0.31.1.

## Model weights and vocabulary

- `CohereLabs/cohere-transcribe-03-2026` weights and tokenizer files are
  downloaded at runtime from Hugging Face and carry the model card's own
  license. The embedded `vocab.json` (id→piece table) is derived from that
  tokenizer.
