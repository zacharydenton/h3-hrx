# H3-World reference fixtures

Action clauses derive from `code/abot/action_script.py` in
[H3-World](https://github.com/Danzer1xxxxChan/H3-World/tree/f0c7be2acbbde8256b2473f6e7b088f58272acae),
revision `f0c7be2acbbde8256b2473f6e7b088f58272acae` (Apache-2.0).
`actions.json` enumerates all 512 bit patterns in the declared key order. It
executes upstream `_motion_clause` and `_camera_clause`, including purification.
The released inference path uses these rules; the COLMAP training helpers and
legacy FiLM implementation are intentionally excluded.

`tokens.json` encodes each distinct sentence with Python tokenizers 0.23.2 and
this repository's vendored Qwen vocabulary, without special tokens, matching
upstream `presentation_t2va`. The authors pin DiffSynth-Studio revision
`300e3e4da76e881d5e6bd97d897810c18f6e4893`.

The RGB fixtures use Pillow 12.3.0 with the inference script's Lanczos cover
resize and center crop. Input channel byte i is `(i*37 + i//7) % 256`.
Dimensions are in `resize.json`. The native floating filter is compared with a
one-byte tolerance for Pillow's fixed-point coefficients.

The adapter is resolved separately from `DANNY621/H3-World`, revision
`cb1a1fe209415bd3c744a74b1058bb8bfd507268`, file
`step-10000.safetensors`. Its model license remains MiniMax H3's community
license; weights and gameplay imagery are not vendored here.
