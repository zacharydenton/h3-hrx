# Gufo attention transfer

The existing F16 experiments already cover the immediate Gufo candidates:
32-key score tiles, four/eight-wave schedules, and zero/two/four resident query
fragments. Start with those rather than adding another implementation.

The independent F64 attention test now includes all four score-tile variants
at 17, 40 and 96 tokens, with multiple heads and zero input headroom. The
existing upper-tile softmax-maximum test remains a separate adversarial check.

```sh
cargo test --release --test kernels attention_matches_scaled_dot_product_for_the_shipped_layouts -- --ignored --test-threads=1
cargo test --release --test kernels attention_preserves_the_upper_tile_softmax_maximum -- --ignored --test-threads=1
cargo build --release --example attention_f16
# tokens, heads, alternating pairs, dependent repetitions per graph
./target/release/examples/attention_f16 4096 4 5 3
./target/release/examples/attention_f16 16384 4 5 3
```

The harness compares identical Q/K/V/output addresses, includes dependent graph
replay, records capacity and compiler artifact locations and reflected launch layouts, and checks
finite outputs plus relative RMS against the production eight-wave kernel.
The baseline comparison is a regression check; the F64 tests are its independent
numerical prerequisite. The benchmark excludes allocation, packing and loading;
it cannot by itself justify a production route change. Reused resident operands
are explicit in every record.

Only extend a winning F16 schedule to encoder/decoder controls after including
real layout preparation and passing the existing model parity gates. INT8/INT4
use different numerical contracts and need their own evidence. A missing
checkpoint is an incomplete qualification.

The shared mapped-page preparation prototype lives in HRX's `FileView`. A
consumer integration belongs in `Weights::fill`, after recipe/segment selection,
not `Checkpoint::open` or schema lookup. Benchmark preparation plus upload
before changing the existing `will_need` calls. This branch does not bump a
runtime pin or apply an unqualified loader policy.

On 2026-09-24 all five focused attention tests passed, including the independent
F64 references and existing quantized-operand controls. Five alternating pairs
per candidate at 4096 and 16384 tokens (four heads, three dependent repetitions)
measured candidate/baseline time ratios from 1.254 to 1.588. The four schedules
are therefore rejected for default adoption on these workloads. Relative RMS
against the production F16 kernel stayed below 0.000028. These resident-input
measurements do not establish performance at all model shapes; no full video
run is warranted by a component regression. [Per-schedule results](gufo-attention-20260924.json)
retain both tested token counts.
