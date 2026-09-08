# Research archive

Historical development records, preserved as evidence for kernel choices.
They include superseded implementations, export-era commands, developer-local
paths, and references to untracked benchmark artifacts. Use the
[setup guide](../setup.md) for current installation instructions and the
[performance summary](../performance.md) for the recorded production results.

| Report | Scope |
| --- | --- |
| [Development notes](notes.md) | Chronological implementation and numerical investigations |
| [Int8 attention study](attention-int8-loom.md) | Initial shipped attention and comparisons |
| [Attention tuning](attention-int8-45.md) | Head-major layout, staging, and compiler experiments |
| [Int8 GEMM tuning](gemm-int8-tuning.md) | Tile shapes, operand pitches, and row groups |
| [Decoder tuning](vae-30s.md) | Progress from 44.1 s to a recorded 33.02 s resident decode |

The attention and GEMM reports have their benchmark JSON beside them. Paths
inside command examples are relative to the repository root unless noted.
