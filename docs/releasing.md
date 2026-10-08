# Releasing h3-hrx

## Validation

Run from a clean checkout with a current stable Rust toolchain:

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --no-default-features --features cli,prompt-generation
cargo check --locked --no-default-features
cargo check --locked --manifest-path clients/rust/Cargo.toml
cargo check --locked --manifest-path clients/rustler/Cargo.toml
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
cargo package --locked
```

CPU CI runs these checks. Packaging verifies that the embedded kernels and
tokenizer are included and the archive builds against published HRX 0.9.0.

On Strix Halo with the native bundle, checkpoints and reference dumps available:

```sh
cargo test
python3 scripts/parity.py gate --require
```

The parity gate needs ComfyUI dumps from `scripts/comfy_dump.py`; missing fixtures
fail the gate. See [test coverage](testing.md) for dependencies and limits.
Record the commit, toolchain, native bundle, GPU and results.

Install and run a complete clip before publishing:

```sh
cargo install --locked --path . --bin h3
h3 --version
h3 --width 864 --height 480 --frames 124 --steps 31 --seed 7 \
  --out clip.mp4 < docs/prompts/cliff_rider_768p.txt
```

## Publication

H3 depends on the published HRX 0.9.0 crate and its pinned native bundle.
Verify the package without local dependency overrides before publishing.

Check the version, package contents, README links, Apache-2.0 code license,
separate model terms and [asset attribution](../assets/README.md).
Keep checkpoints, scratch renders and local configuration out of the package.
Curated media belongs in `docs/media/`, with settings in the showcase.

Release notes should state supported hardware, remaining limitations and the
checks actually run. Publishing a crate and creating a release tag are separate
actions from preparing and validating the source.
