# Releasing h3-hrx

## Identity and discovery

- Repository and Cargo package: `h3-hrx`.
- CLI: `h3-hrx`; Rust library: `h3_hrx`; diagnostics: `h3-hrx-dev`.
- Tagline: **MiniMax H3 video and audio generation on AMD Strix Halo, powered by Loom and HRX.**
- Repository: <https://github.com/zacharydenton/h3-hrx>.

GitHub topics:

```text
loom, hrx, strix-halo, amd, minimax-h3, hailuo, video-generation,
audio-generation, text-to-video, image-to-video, generative-ai,
diffusion-models, gpu-computing, inference, comfyui
```

Cargo keywords live in `Cargo.toml`. Keep the package description, README
tagline, CLI help, and GitHub description consistent. Use Strix Halo in
prominent copy; Radeon 8060S and `gfx1151` identify the tested device in
technical requirements. Describe benchmark results as recorded measurements
with their dimensions, precision, hardware, and timing boundaries.

## Validation

From the repository root, using a current stable Rust toolchain:

```sh
bash scripts/test.sh --cpu
cargo check --locked --no-default-features
cargo check --locked --manifest-path clients/rust/Cargo.toml
cargo check --locked --manifest-path clients/rustler/Cargo.toml
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
cargo package --locked
```

`cargo package` verifies that the packaged source builds, including the embedded
Loom kernels and tokenizer. Use `--allow-dirty` only for local preparation;
build the final package from a clean commit. Client examples and historical
experiments are not part of the published crate. CPU CI runs the checks above.

On the supported Strix Halo system, with the native bundle and checkpoints:

```sh
bash scripts/test.sh --full
python3 scripts/parity.py gate --require
```

The parity gate also needs ComfyUI reference dumps. See
[test coverage](testing.md) and `scripts/comfy_dump.py`; missing fixtures are
a failed gate. Python is used for this optional reference-validation tooling,
not for inference. Record the commit, toolchain, native bundle, GPU, and results.
Do not describe CPU-only validation as a fully validated model release.

Install the final package and run the README example on the supported hardware:

```sh
cargo install --locked --path . --bin h3-hrx
h3-hrx --version
h3-hrx --width 864 --height 480 --frames 124 --steps 31 --seed 7 \
  --out clip.mp4 < docs/prompts/cliff_rider_768p.txt
```

## Publication

Confirm the intended version, Apache-2.0 code license, separate model terms,
bundled-asset attribution, README links, and package contents. Review tracked
files and Git history for credentials and private data before making the
repository public. Keep generated clips, checkpoints, and local configuration
out of the repository.

Prepare release notes with supported hardware, limitations, and actual validation
results. Publishing a crate, creating a release tag, and changing repository
visibility are separate release actions after preparation and validation.
