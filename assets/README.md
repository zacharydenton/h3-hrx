# Embedded tokenizer

`tokenizer.json` is the Qwen3-VL tokenizer from
[Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
published by the Qwen team under Apache-2.0. The license text is in
[`../LICENSE`](../LICENSE).

The bundled file is byte-identical to the
[upstream tokenizer at revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/0cfaf48183f594c314753d30a4c4974bc75f3ccb/tokenizer.json),
verified on 2026-09-13. No modifications have been made to that file.

SHA-256:

```text
a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7
```

The library embeds it at compile time for offline tokenization. `H3_TOKENIZER`
can select another file at runtime. Model checkpoints are downloaded separately
and are not bundled with this repository or its Cargo package.
