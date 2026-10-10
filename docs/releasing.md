# Releasing h3-hrx

Ship a build that installs from published dependencies, passes numerical checks
and completes a render on the supported hardware. Record the commit, toolchain,
HRX bundle, Loom compiler and checks actually run.

## Build and package

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
tokenizer are included and the archive builds against its published dependencies.

## Validate on Strix Halo

With the native bundle, checkpoints and reference dumps available:

```sh
cargo test
python3 scripts/parity.py gate --require
```

The parity gate needs ComfyUI dumps from `scripts/comfy_dump.py`; missing fixtures
fail the gate. See [test coverage](testing.md) for dependencies and limits.

Install and run the documented 768p workload before publishing:

```sh
cargo install --locked --path . --bin h3
h3 --version
/usr/bin/time -f 'Elapsed: %e seconds' \
  h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 0 \
  --weight-io native-direct --memory-budget-mib 49152 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

Inspect the completed video and audio. Compare speed using the
[performance guide](performance.md); update advertised timings only from
completed, reproducible measurements with their workload and timing boundaries.

## Publication

HRX 0.10.1 includes the native large-transfer support required by H3. Verify the
package against published dependencies without local overrides before publishing.

Check the version, package contents, README links, Apache-2.0 code license,
separate model terms and [asset attribution](../assets/README.md).
Keep checkpoints, scratch renders and local configuration out of the package.
Curated media belongs in `docs/media/`, with settings in the showcase.

Release notes should state supported hardware, remaining limitations and the
checks actually run. Publishing a crate and creating a release tag are separate
actions from preparing and validating the source.
