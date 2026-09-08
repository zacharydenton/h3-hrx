//! The model's shapes, and the one layout rule the plans share with the kernel dispatch.

// The DiT.
pub const HID: usize = 5376;
pub const HEADS: usize = 56;
pub const HEAD_DIM: usize = 128;
pub const FFN: usize = 14336;
pub const ROPE_DIM: usize = 96;
pub const ROPE_HALF: usize = 48;
pub const INNER: usize = HEADS * HEAD_DIM;
pub const QKV: usize = 3 * INNER;
pub const BLOCKS: usize = 50;
pub const REFINER_BLOCKS: usize = 2;

pub const TEXT_DIM: usize = 5120;
pub const VIDEO_PATCH: usize = 96;
pub const AUDIO_CH: usize = 32;
pub const FINAL_N: usize = 128;
pub const CLASSES: usize = 12;
pub const MODALITIES: usize = 3;
pub const MODS_ROWS: usize = 6 * CLASSES;

/// ComfyUI's VISUAL_COND_TIMESTEP: reference latents at 0.999 * z + 0.001 * noise.
pub const VISUAL_COND_AUG: f32 = 0.999;

// The Qwen3-VL text encoder.
pub const TE_HID: usize = 5120;
pub const TE_HEADS: usize = 64;
pub const TE_KV: usize = 8;
pub const TE_FFN: usize = 25600;
pub const TE_ROPE_HALF: usize = 64;
pub const TE_LAYERS: usize = 50;

// Latents and rates.
pub const LATENT_CH: usize = 24;
pub const FPS: usize = 24;
pub const AUDIO_LATENTS_PER_S: usize = 40;

// The video VAE.
pub const VAE_HID: usize = 2048;
pub const VAE_HEADS: usize = 32;
pub const VAE_D: usize = 64;
pub const VAE_FFN: usize = 8192;
pub const VAE_ROPE_HALF: usize = 24;
pub const VAE_PT: usize = 4;
pub const VAE_PS: usize = 16;
pub const VAE_OUT: usize = 3 * VAE_PT * VAE_PS * VAE_PS;
pub const VAE_REG: usize = 4;
/// The decoder's 24 latent channels padded to the f16 GEMM's multiple of 64.
pub const VAE_KIN: usize = 64;
pub const VAE_BLOCKS: usize = 36;
pub const VAE_CHUNK: usize = 5;
pub const VAE_OVERLAP: usize = 2;
pub const VAE_TOKEN_DROP: usize = 3;
pub const VAE_TRATIO: usize = 4;
pub const VAE_CLIP: usize = 17;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const FRAME_RESCALE: f64 = 5.0 / 3.0;
pub const SPATIAL_SCALE: f64 = 32.0;
pub const FRAME_PER_TOKEN: [usize; 5] = [1, 4, 4, 4, 4];
pub const THREADS: u32 = 256;

/// A GEMM operand's row pitch in elements.
///
/// Rows whose byte pitch is a multiple of 1024 alias in the cache, so those get one k step of padding
/// (64 elements at 8 bits, 128 at 4 and 16, which the f16 attention's `out_stride` also requires) and the
/// kernels' step constraints still hold. Measured: the int8 out projection at K 7168 went 36.7 -> 41.2
/// TOPS, the video decoder's 16-bit down projection at K 8192 went 15.3 -> 26.0 TFLOP/s, its qkv at
/// K 2048 23.5 -> 26.0.
///
/// The plans and the kernel dispatch must agree on this, or an operand is read at the wrong stride.
pub fn gemm_pitch(k: usize, bits: usize) -> usize {
    let row_bytes = match bits {
        4 => k / 2,
        8 => k,
        _ => k * 2,
    };
    if row_bytes.is_multiple_of(1024) {
        k + if bits == 8 { 64 } else { 128 }
    } else {
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_pitch_pads_only_the_aliasing_widths() {
        // int8: the byte pitch is k, so a k that is a multiple of 1024 pads by 64
        assert_eq!(gemm_pitch(HID, 8), HID); // 5376 bytes, not a multiple of 1024
        assert_eq!(gemm_pitch(INNER, 8), INNER + 64); // 7168 bytes
        assert_eq!(gemm_pitch(FFN, 8), FFN + 64); // 14336 bytes
                                                  // 16-bit: the byte pitch is 2k, so the threshold is half, and the pad is 128
        assert_eq!(gemm_pitch(VAE_HID, 16), VAE_HID + 128); // 4096 bytes
        assert_eq!(gemm_pitch(VAE_FFN, 16), VAE_FFN + 128); // 16384 bytes
        assert_eq!(gemm_pitch(HID, 16), HID); // 10752 bytes, not a multiple of 1024
                                              // 4-bit: half a byte per element
        assert_eq!(gemm_pitch(2048, 4), 2048 + 128); // 1024 bytes
        assert_eq!(gemm_pitch(1024, 4), 1024); // 512 bytes
    }

    #[test]
    fn the_shapes_are_self_consistent() {
        assert_eq!(INNER, 7168);
        assert_eq!(QKV, 21504);
        assert_eq!(VAE_OUT, 3072);
        assert_eq!(MODS_ROWS, 72);
        assert_eq!(ROPE_DIM, 2 * ROPE_HALF);
    }
}

/// The prepare kernels' lane count for a row width: the widest that divides it both ways.
pub fn lanes_for(width: usize) -> Option<usize> {
    [320, 256, 160, 128, 96, 64, 32]
        .into_iter()
        .find(|&l| width.is_multiple_of(8 * l) && (width / 4).is_multiple_of(l))
}

/// The raster group rule, shared with the Python kernel harnesses: how many workgroup rows are grouped
/// so the tail wastes as little as possible. Four unless three or two pad less.
pub fn m_group_for(tokens: usize, tile: usize) -> u32 {
    let tiles = tokens.div_ceil(tile);
    if tiles == 1 {
        return 1;
    }
    let (mut best, mut best_pad) = (4u32, tiles.div_ceil(4) * 4);
    for g in [3u32, 2u32] {
        let pad = tiles.div_ceil(g as usize) * g as usize;
        if pad < best_pad {
            best = g;
            best_pad = pad;
        }
    }
    best
}

/// The long int8 down projection benefits from a smaller row group. Kept shape-specific: the one-row
/// group and a wider tile both lost when measured.
pub fn gemm_m_group_for(tokens: usize, k: usize, n: usize, bits: usize) -> u32 {
    if bits == 8 && k == FFN && n == HID && tokens >= 32768 {
        return 2;
    }
    m_group_for(tokens, 256)
}

/// Measured on the decoder's full 7x16x16 tile plus its five special tokens. The choices are specific
/// to these projections; smaller tiles keep the general rule.
pub fn vae_fast_m_group_for(tokens: usize, k: usize, n: usize) -> u32 {
    if tokens == 1797 {
        if k == 8192 && n == 2048 {
            return 1;
        }
        if k == 2048 && (n == 2048 || n == 6144 || n == 16384) {
            return 15;
        }
    }
    m_group_for(tokens, 128)
}

/// Workgroup rows for a GEMM, rounded up to a whole number of raster groups.
pub fn gemm_grid_y(tokens: usize, group: u32, tile: usize) -> u32 {
    (tokens.div_ceil(tile).div_ceil(group as usize) * group as usize) as u32
}

/// The GEMM operand element types the stacks run: the checkpoint's int8 ConvRot rows (rotated
/// activations with per-token scales), or its f16 / bf16 rows as stored (unrotated, no scales).
pub fn quantised(elem: &str) -> bool {
    elem == "i8"
}

pub fn elem_bits(elem: &str) -> usize {
    if quantised(elem) {
        8
    } else {
        16
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    #[test]
    fn lane_counts_match_the_widths_the_model_uses() {
        assert_eq!(lanes_for(2048), Some(256));
        assert_eq!(lanes_for(8192), Some(256));
        assert_eq!(lanes_for(25600), Some(320));
        assert_eq!(lanes_for(5376), Some(96));
        assert_eq!(lanes_for(HID), Some(96));
        assert_eq!(lanes_for(7), None);
    }

    #[test]
    fn the_row_group_rule_pads_as_little_as_it_can() {
        // The values tests/test_host_logic.cpp pins.
        assert_eq!(gemm_m_group_for(37723, FFN, HID, 8), 2);
        assert_eq!(gemm_m_group_for(32768, FFN, HID, 8), 2);
        // just below the threshold, and the wrong shape, both fall through to the general rule
        assert_eq!(
            gemm_m_group_for(32767, FFN, HID, 8),
            m_group_for(32767, 256)
        );
        assert_eq!(
            gemm_m_group_for(37723, FFN, HID, 16),
            m_group_for(37723, 256)
        );
        assert_eq!(
            gemm_m_group_for(37723, HID, FFN, 8),
            m_group_for(37723, 256)
        );
        // one tile is its own group
        assert_eq!(m_group_for(1, 256), 1);
        assert_eq!(m_group_for(256, 256), 1);
        assert_eq!(m_group_for(257, 256), 2);
    }

    #[test]
    fn the_decoder_projections_keep_their_measured_groups() {
        assert_eq!(vae_fast_m_group_for(1797, 8192, 2048), 1);
        for n in [2048, 6144, 16384] {
            assert_eq!(vae_fast_m_group_for(1797, 2048, n), 15, "n {n}");
        }
        // any other shape or token count falls through, at the 128 tile these kernels use
        assert_eq!(
            vae_fast_m_group_for(1797, 4096, 2048),
            m_group_for(1797, 128)
        );
        assert_eq!(vae_fast_m_group_for(517, 2048, 2048), m_group_for(517, 128));
    }

    #[test]
    fn grid_y_covers_the_tokens_and_is_a_whole_number_of_groups() {
        for tokens in [1usize, 255, 256, 257, 513, 16000, 32767, 32768, 37723] {
            for group in [1u32, 2, 3, 4, 15] {
                let gy = gemm_grid_y(tokens, group, 256);
                assert_eq!(gy % group, 0, "{tokens}/{group}");
                assert!(gy as usize * 256 >= tokens, "{tokens}/{group}");
                assert!((gy - group) as usize * 256 < tokens, "{tokens}/{group}");
            }
        }
    }

    #[test]
    fn element_types_map_to_their_widths() {
        assert!(quantised("i8") && !quantised("f16") && !quantised("bf16"));
        assert_eq!(elem_bits("i8"), 8);
        assert_eq!(elem_bits("f16"), 16);
        assert_eq!(elem_bits("bf16"), 16);
    }
}
