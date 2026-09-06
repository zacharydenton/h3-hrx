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
#ifndef SCHEDULE_MODE
#define SCHEDULE_MODE 0
#endif
#ifndef Q_LDS
#define Q_LDS 0
#endif
#ifndef BUFFERS
#define BUFFERS 2
#endif
#ifndef PV_GROUP
#define PV_GROUP 1
#endif
#ifndef SKIP_RESCALE
#define SKIP_RESCALE 0
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME h3_attention_i8qkhip_mha8_lds_f16_wmma
#endif
using i4 = int __attribute__((ext_vector_type(4)));
using i8 = int __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));
using h8 = _Float16 __attribute__((ext_vector_type(8)));
using h16 = _Float16 __attribute__((ext_vector_type(16)));

__device__ __forceinline__ float swap_halves(float value) {
  int bits = __builtin_bit_cast(int, value);
  bits = __builtin_amdgcn_permlanex16(0, bits, 0x76543210, 0xfedcba98, true, false);
  return __builtin_bit_cast(float, bits);
}

__device__ __forceinline__ i8 prefetch_tile(const int* ki, const _Float16* v, int head, int key0, int tid) {
  i8 payload = {};
  if (tid < 128) {
    i4 words = *reinterpret_cast<const i4*>(ki + ((key0 + tid / 8) * HEADS + head) * 32 + (tid % 8) * 4);
    #pragma unroll
    for (int i = 0; i < 4; ++i) payload[i] = words[i];
  } else {
    payload = __builtin_bit_cast(i8, *reinterpret_cast<const h16*>(v + (head * 128 + tid - 128) * CAPACITY + key0));
  }
  return payload;
}

extern "C" __global__ __launch_bounds__(WAVES * 32)
void KERNEL_NAME(uint64_t token_count, const int* __restrict__ qi,
                const float* __restrict__ qs, const int* __restrict__ ki,
                const float* __restrict__ ks, const _Float16* __restrict__ v,
                _Float16* __restrict__ out) {
  __shared__ int kt[BUFFERS][KEY_TILE][36];
  __shared__ _Float16 vt[BUFFERS][128][KEY_TILE + 8];
  __shared__ float st[BUFFERS][KEY_TILE];
  __shared__ int qtile[Q_LDS ? Q_LDS : 1][WAVES][16][32];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int lc = lane % 16, parity = lane / 16, head = blockIdx.y;
  const int query0 = (blockIdx.x * WAVES + wave) * (16 * QUERY_TILES) + lc;
  i4 q[QUERY_TILES][8];
  float qscale[QUERY_TILES];
  #pragma unroll
  for (int qt = 0; qt < QUERY_TILES; ++qt) {
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      i4 value = *reinterpret_cast<const i4*>(qi + ((query0 + qt * 16) * HEADS + head) * 32 + 4 * c);
      if (qt < Q_LDS) {
        if (parity == 0) *reinterpret_cast<i4*>(&qtile[qt][wave][lc][c * 4]) = value;
      } else q[qt][c] = value;
    }
    qscale[qt] = qs[(query0 + qt * 16) * HEADS + head] * 1.4426950408889634f;
  }
  f8 acc[QUERY_TILES][8] = {};
  float row_max[QUERY_TILES], row_sum[QUERY_TILES] = {};
  #pragma unroll
  for (int qt = 0; qt < QUERY_TILES; ++qt) row_max[qt] = -1.e9f;
  static_assert(!PREFETCH || (KEY_TILE == 16 && WAVES == 8));
  i8 prefetched;
  if constexpr (PREFETCH) prefetched = prefetch_tile(ki, v, head, 0, tid);
  for (int key0 = 0; key0 < TOKENS; key0 += KEY_TILE) {
    const int buf = (key0 / KEY_TILE) % BUFFERS;
    // Coalesced global accesses, cooperatively reused by all query waves.
    if constexpr (PREFETCH) {
      if (tid < 128) {
        i4 words = {prefetched[0], prefetched[1], prefetched[2], prefetched[3]};
        *reinterpret_cast<i4*>(&kt[buf][tid / 8][(tid % 8) * 4]) = words;
      } else *reinterpret_cast<h16*>(&vt[buf][tid - 128][0]) = __builtin_bit_cast(h16, prefetched);
    } else {
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
    }
    if (tid < KEY_TILE) st[buf][tid] = ks[(key0 + tid) * HEADS + head];
    __syncthreads();
    if constexpr (PREFETCH) prefetched = prefetch_tile(ki, v, head, key0 + 16, tid);
    #pragma unroll
    for (int qt = 0; qt < QUERY_TILES; ++qt) {
    if constexpr (SCHEDULE_MODE > 0) __builtin_amdgcn_sched_barrier(0);
    f8 scores[KEY_TILE / 16];
    float tile_max = -1.e9f;
    #pragma unroll
    for (int sub = 0; sub < KEY_TILE / 16; ++sub) {
      i8 dot = {};
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        if constexpr (SCHEDULE_MODE > 1) __builtin_amdgcn_sched_barrier(0);
        i4 k = *reinterpret_cast<const i4*>(&kt[buf][sub * 16 + lc][4 * c]);
        i4 query_fragment;
        if (qt < Q_LDS) query_fragment = *reinterpret_cast<const volatile i4*>(&qtile[qt][wave][lc][c * 4]);
        else query_fragment = q[qt][c];
        if (c == 0) {
          asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, 0 neg_lo:[1,1,0]"
              : "=v"(dot) : "v"(k), "v"(query_fragment));
        } else dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, k, true, query_fragment, dot, false);
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
    tile_max = fmaxf(tile_max, swap_halves(tile_max));
    if constexpr (SCHEDULE_MODE > 0) __builtin_amdgcn_sched_barrier(0);
    const float new_max = fmaxf(row_max[qt], tile_max);
    const float rescale = __builtin_amdgcn_exp2f(row_max[qt] - new_max);
    row_sum[qt] *= rescale;
    h16 prob[KEY_TILE / 16];
    #pragma unroll
    for (int sub = 0; sub < KEY_TILE / 16; ++sub) {
      f8 weights;
      h8 halves;
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        float weight = __builtin_amdgcn_exp2f(scores[sub][e] - new_max);
        weights[e] = weight;
        halves[e] = _Float16(weight);
      }
      row_sum[qt] += ((weights[0] + weights[1]) + (weights[2] + weights[3])) +
                     ((weights[4] + weights[5]) + (weights[6] + weights[7]));
      i4 own = __builtin_bit_cast(i4, halves);
      i8 packed;
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        int other = __builtin_amdgcn_permlanex16(0, own[e], 0x76543210, 0xfedcba98, true, false);
        int even = parity == 0 ? own[e] : other;
        int odd = parity == 0 ? other : own[e];
        packed[2 * e] = __builtin_amdgcn_perm(odd, even, 0x05040100);
        packed[2 * e + 1] = __builtin_amdgcn_perm(odd, even, 0x07060302);
      }
      prob[sub] = __builtin_bit_cast(h16, packed);
    }
    const bool needs_rescale = !SKIP_RESCALE || __any(new_max != row_max[qt]);
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      if (SCHEDULE_MODE > 0 && c % PV_GROUP == 0) __builtin_amdgcn_sched_barrier(0);
      if (needs_rescale) acc[qt][c] *= rescale;
      #pragma unroll
      for (int sub = 0; sub < KEY_TILE / 16; ++sub) {
        h16 value = *reinterpret_cast<const h16*>(&vt[buf][c * 16 + lc][sub * 16]);
        acc[qt][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(value, prob[sub], acc[qt][c]);
      }
    }
    row_max[qt] = new_max;
    }
    if constexpr (BUFFERS == 1) __syncthreads();
  }
  #pragma unroll
  for (int qt = 0; qt < QUERY_TILES; ++qt) {
  const int query = query0 + qt * 16;
  // WMMA needs every lane even when the last query tile is partial. Prevent
  // Clang from sinking a short, unrolled loop's final MMA into the store mask.
  #pragma unroll
  for (int c = 0; c < 8; ++c) asm volatile("" : "+v"(acc[qt][c]));
  asm volatile("" : "+v"(row_sum[qt]));
  const float inv_sum = 1.f / (row_sum[qt] + swap_halves(row_sum[qt]));
  if (query < TOKENS && query < token_count) {
    #pragma unroll
    for (int c = 0; c < 8; ++c)
      #pragma unroll
      for (int e = 0; e < 8; ++e)
        out[(query * HEADS + head) * 128 + c * 16 + 2 * e + parity] = _Float16(acc[qt][c][e] * inv_sum);
  }
  }
}
