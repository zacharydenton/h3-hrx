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
