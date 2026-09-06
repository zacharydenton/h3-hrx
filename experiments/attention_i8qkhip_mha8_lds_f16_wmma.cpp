// Dense gfx1151 INT8 KQ^T + FP16 V^T P^T, FP32 accumulation and online softmax.
// Same buffer ABI as the Loom INT8 attention kernel. Build via bench_attention_i8.py.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef TOKENS
#define TOKENS 16000
#endif
#ifndef CAPACITY
#define CAPACITY 16128
#endif
#ifndef HEADS
#define HEADS 56
#endif
#ifndef KEY_TILE
#define KEY_TILE 16
#endif
#ifndef WAVES
#define WAVES 8
#endif
#ifndef QUERY_TILES
#define QUERY_TILES 1
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME h3_attention_i8qkhip_mha8_lds_f16_wmma
#endif
using i4 = int __attribute__((ext_vector_type(4)));
using i8 = int __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));
using h16 = _Float16 __attribute__((ext_vector_type(16)));

extern "C" __global__ __launch_bounds__(WAVES * 32)
void KERNEL_NAME(uint64_t token_count, const int* __restrict__ qi,
                const float* __restrict__ qs, const int* __restrict__ ki,
                const float* __restrict__ ks, const _Float16* __restrict__ v,
                _Float16* __restrict__ out) {
  __shared__ int kt[2][KEY_TILE][36];
  __shared__ _Float16 vt[2][128][KEY_TILE + 8];
  __shared__ float st[2][KEY_TILE];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int lc = lane % 16, parity = lane / 16, head = blockIdx.y;
  const int query0 = (blockIdx.x * WAVES + wave) * (16 * QUERY_TILES) + lc;
  i4 q[QUERY_TILES][8];
  float qscale[QUERY_TILES];
  #pragma unroll
  for (int qt = 0; qt < QUERY_TILES; ++qt) {
    #pragma unroll
    for (int c = 0; c < 8; ++c)
      q[qt][c] = *reinterpret_cast<const i4*>(qi + ((query0 + qt * 16) * HEADS + head) * 32 + 4 * c);
    qscale[qt] = qs[(query0 + qt * 16) * HEADS + head] * 1.4426950408889634f;
  }
  f8 acc[QUERY_TILES][8] = {};
  float row_max[QUERY_TILES], row_sum[QUERY_TILES] = {};
  #pragma unroll
  for (int qt = 0; qt < QUERY_TILES; ++qt) row_max[qt] = -1.e9f;
  for (int key0 = 0; key0 < TOKENS; key0 += KEY_TILE) {
    const int buf = (key0 / KEY_TILE) % 2;
    // Coalesced global accesses, cooperatively reused by all query waves.
    #pragma unroll
    for (int i = tid; i < KEY_TILE * 8 + 128 * (KEY_TILE / 16); i += WAVES * 32) {
      if (i < KEY_TILE * 8) {
        const int key = i / 8, chunk = i % 8;
        *reinterpret_cast<i4*>(&kt[buf][key][4 * chunk]) =
            *reinterpret_cast<const i4*>(ki + ((key0 + key) * HEADS + head) * 32 + 4 * chunk);
      } else {
        const int vi = i - KEY_TILE * 8;
        const int channel = vi / (KEY_TILE / 16), sub = (vi % (KEY_TILE / 16)) * 16;
        *reinterpret_cast<h16*>(&vt[buf][channel][sub]) =
            *reinterpret_cast<const h16*>(v + (head * 128 + channel) * CAPACITY + key0 + sub);
      }
    }
    if (tid < KEY_TILE) st[buf][tid] = ks[(key0 + tid) * HEADS + head];
    __syncthreads();
    #pragma unroll
    for (int qt = 0; qt < QUERY_TILES; ++qt) {
    f8 scores[KEY_TILE / 16];
    float tile_max = -1.e9f;
    #pragma unroll
    for (int sub = 0; sub < KEY_TILE / 16; ++sub) {
      i8 dot = {};
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        i4 k = *reinterpret_cast<const i4*>(&kt[buf][sub * 16 + lc][4 * c]);
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, k, true, q[qt][c], dot, false);
      }
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        int key = sub * 16 + 2 * e + parity;
        float score = float(dot[e]) * qscale[qt] * st[buf][key];
        score = key0 + key < TOKENS ? score : -1.e9f;
        scores[sub][e] = score;
        tile_max = fmaxf(tile_max, score);
      }
    }
    tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16));
    const float new_max = fmaxf(row_max[qt], tile_max);
    const float rescale = exp2f(row_max[qt] - new_max);
    row_sum[qt] *= rescale;
    h16 prob[KEY_TILE / 16];
    #pragma unroll
    for (int sub = 0; sub < KEY_TILE / 16; ++sub) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        float weight = exp2f(scores[sub][e] - new_max);
        row_sum[qt] += weight;
        float other = __shfl_xor(weight, 16);
        prob[sub][2 * e] = _Float16(parity == 0 ? weight : other);
        prob[sub][2 * e + 1] = _Float16(parity == 0 ? other : weight);
      }
    }
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      acc[qt][c] *= rescale;
      #pragma unroll
      for (int sub = 0; sub < KEY_TILE / 16; ++sub) {
        h16 value = *reinterpret_cast<const h16*>(&vt[buf][c * 16 + lc][sub * 16]);
        acc[qt][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(value, prob[sub], acc[qt][c]);
      }
    }
    row_max[qt] = new_max;
    }
  }
  #pragma unroll
  for (int qt = 0; qt < QUERY_TILES; ++qt) {
  const int query = query0 + qt * 16;
  const float inv_sum = 1.f / (row_sum[qt] + __shfl_xor(row_sum[qt], 16));
  if (query < TOKENS && query < token_count) {
    #pragma unroll
    for (int c = 0; c < 8; ++c)
      #pragma unroll
      for (int e = 0; e < 8; ++e)
        out[(query * HEADS + head) * 128 + c * 16 + 2 * e + parity] = _Float16(acc[qt][c][e] * inv_sum);
  }
  }
}
