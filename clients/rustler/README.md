This is an application-owned Rustler 0.38 adapter to the H3 Rust API. It does
calls the crate directly. Each resource has a bounded job channel; a dedicated thread
owns the model, runs inference and releases the GPU session after the resource
is collected. NIF inputs are copied into owned job data before the NIF returns.
Model files must remain immutable while the worker is alive.

Build and run the GPU smoke test without Mix or Python:

```sh
cargo build --manifest-path clients/rustler/Cargo.toml
H3_NIF="$PWD/clients/rustler/target/debug/libh3_nif" elixir clients/rustler/smoke.exs
```

The example exports `open/1`, `ping/2`, and `encode_audio/5`. For audio, pass a
checkpoint path to `open/1`, then an interleaved little-endian float32 binary,
frame count, and request ID to `encode_audio/5`. Results arrive as
`{:h3_result, id, {:ok, {latents, frames}}}` or `{:h3_result, id, {:error, reason}}`.
The example returns latents as a list; an application handling large outputs
should encode them as an owned binary. A full queue rejects the request.

Rustler is optional and confined to this adapter; neither model nor HRX requires
BEAM at build or run time.
The resource and scheduler choices follow the [Rustler resource documentation](https://docs.rs/rustler/0.38.0/rustler/struct.ResourceArc.html)
and [NIF scheduling documentation](https://docs.rs/rustler/0.38.0/rustler/attr.nif.html).
