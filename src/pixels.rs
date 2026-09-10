//! Pixels in and out, and the two normalisations that are not the same.
//!
//! ImageNet's constants normalise the VAE encoder's input and denormalise the decoder's output; CLIP's
//! normalise the vision tower's patches. They are close enough to look interchangeable and are not —
//! Qwen2-VL's preprocessing is CLIP's, and using ImageNet there shifts every vision embedding.
use crate::model::*;

pub use crate::model::{IMAGENET_MEAN, IMAGENET_STD};
// Spelled out to the digit the preprocessing config gives, so they can be read against it directly;
// two of them carry more digits than an f32 holds and round to the same value regardless.
#[allow(clippy::excessive_precision)]
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// A unit-interval float as a byte.
///
/// The NaN guard is `!(v > 0)` rather than `v <= 0` so a NaN goes to zero instead of through the clamp.
/// The multiply is f32 and the rounding is done by adding a half in f64, which is what `lround` does:
/// rounding the f32 product directly would round a value just under an integer's halfway point up.
// The guard is deliberately the negation of a comparison rather than `v <= 0.0`: those differ on NaN,
// and a NaN pixel has to become zero rather than fall through to the clamp.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn unit_to_byte(v: f32) -> u8 {
    if !(v > 0.0) {
        return 0;
    }
    let scaled = v.clamp(0.0, 1.0) * 255.0;
    (f64::from(scaled) + 0.5) as u8
}

/// The VAE encoder's input row for one pixel: ImageNet-normalised, channels 3..7 left at zero.
pub fn imagenet_normalise(rgb: [f32; 3]) -> [f32; 3] {
    [
        (rgb[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (rgb[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (rgb[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    ]
}

/// The decoder's output: undo ImageNet, then to a byte.
pub fn imagenet_denormalise(v: f32, c: usize) -> u8 {
    unit_to_byte(v * IMAGENET_STD[c] + IMAGENET_MEAN[c])
}

/// The vision tower's patches, in merge order.
///
/// `[gh/2][gw/2][2][2]` blocks of 16x16, each patch `[3][2][16][16]` with the image in both temporal
/// slots — Qwen2-VL feeds a still image as a two-frame clip.
pub fn vision_patches(pixels: &[f32], gh: usize, gw: usize, w: usize) -> Vec<f32> {
    let n = gh * gw;
    let mut patches = vec![0.0f32; n * VISION_PATCH];
    for bh in 0..gh / 2 {
        for bw in 0..gw / 2 {
            for ih in 0..2 {
                for iw in 0..2 {
                    let pi = ((bh * (gw / 2) + bw) * 2 + ih) * 2 + iw;
                    let dst = &mut patches[pi * VISION_PATCH..(pi + 1) * VISION_PATCH];
                    for c in 0..3 {
                        for t in 0..2 {
                            for py in 0..16 {
                                for px in 0..16 {
                                    let y = (bh * 2 + ih) * 16 + py;
                                    let x = (bw * 2 + iw) * 16 + px;
                                    dst[((c * 2 + t) * 16 + py) * 16 + px] =
                                        (pixels[(y * w + x) * 3 + c] - CLIP_MEAN[c]) / CLIP_STD[c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    patches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_becomes_zero_rather_than_clamping() {
        assert_eq!(unit_to_byte(f32::NAN), 0);
        assert_eq!(unit_to_byte(-0.0), 0);
        assert_eq!(unit_to_byte(-1.0), 0);
        assert_eq!(unit_to_byte(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn the_endpoints_and_the_halfway_points_land_where_lround_puts_them() {
        assert_eq!(unit_to_byte(0.0), 0);
        assert_eq!(unit_to_byte(1.0), 255);
        assert_eq!(unit_to_byte(2.0), 255);
        assert_eq!(unit_to_byte(f32::INFINITY), 255);
        assert_eq!(unit_to_byte(1.0 / 255.0), 1);
        // 0.5 * 255 = 127.5 exactly, and a half rounds away from zero
        assert_eq!(unit_to_byte(0.5), 128);
        // one ulp below the halfway point must not round up, which is what the f64 add protects
        let just_under = f32::from_bits((0.5f32 * 255.0).to_bits() - 1);
        assert_eq!((f64::from(just_under) + 0.5) as u8, 127);
    }

    #[test]
    fn every_byte_round_trips_through_the_decoder_path() {
        // a byte, undone by ImageNet and redone, is the same byte
        for b in 0..=255u8 {
            for c in 0..3 {
                let unit = f32::from(b) / 255.0;
                let normalised = (unit - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                assert_eq!(
                    imagenet_denormalise(normalised, c),
                    b,
                    "byte {b} channel {c}"
                );
            }
        }
    }

    #[test]
    fn the_two_constant_sets_are_distinct() {
        assert_ne!(IMAGENET_MEAN, CLIP_MEAN);
        assert_ne!(IMAGENET_STD, CLIP_STD);
        // the difference is large enough to matter: about a tenth of a standard deviation on blue
        let shift = (CLIP_MEAN[2] - IMAGENET_MEAN[2]) / CLIP_STD[2];
        assert!(shift.abs() > 0.005, "{shift}");
    }

    #[test]
    fn vision_patches_walk_the_merge_order_with_the_image_in_both_slots() {
        // a 4x4 patch grid over a 64x64 image whose red channel encodes the pixel index
        let (gh, gw, w) = (4usize, 4usize, 64usize);
        let mut px = vec![0.0f32; w * w * 3];
        for y in 0..w {
            for x in 0..w {
                px[(y * w + x) * 3] = (y * w + x) as f32;
            }
        }
        let p = vision_patches(&px, gh, gw, w);
        let un = |v: f32| v * CLIP_STD[0] + CLIP_MEAN[0];
        // patch 0 is block (0,0) inner (0,0): the image's top-left 16x16
        assert_eq!(un(p[0]).round(), 0.0);
        // both temporal slots carry the same content
        let slot1 = 16 * 16;
        assert_eq!(p[0], p[slot1]);
        // patch 1 is inner (0,1): 16 columns across
        assert_eq!(un(p[VISION_PATCH]).round(), 16.0);
        // patch 2 is inner (1,0): 16 rows down
        assert_eq!(un(p[2 * VISION_PATCH]).round(), (16 * w) as f32);
        // patch 4 starts block (0,1): 32 columns across
        assert_eq!(un(p[4 * VISION_PATCH]).round(), 32.0);
    }
}
