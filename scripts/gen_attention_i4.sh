#!/bin/sh
# Regenerate the shipped int4-QK attention kernels: the 8-wave and 4-wave forms, the LDS-padded long form
# (one workgroup per CU, rows >= 20000), and their tile-skip twins (attention_i4qks*).
set -e
cd "$(dirname "$0")/.."
python3 tools/gen_attention_i4qk.py
ATTN_WAVES=4 python3 tools/gen_attention_i4qk.py
ATTN_DBUF_STEM=attention_i4qkl_mha8_lds_f16_wmma ATTN_LDS_PAD=24000 python3 tools/gen_attention_i4qk.py
