//! Hashes every f16 conversion, both directions, over every bit pattern.
//!
//!   half_dump to16   -- all 2^32 f32 patterns through f32 -> f16
//!   half_dump to32   -- all 2^16 f16 patterns through f16 -> f32
//!
//! The C rolled its own conversions; this uses the `half` crate. Nothing about that is safe to assume,
//! because the pipeline stages operands through f16 in several places and a single rounding difference
//! moves a decoded pixel. Both directions are checked exhaustively against the C's text.
fn main() {
    let mode = std::env::args().nth(1).expect("to16 or to32");
    let mut h: u64 = 14695981039346656037;
    if mode == "to16" {
        for bits in 0..=0xffff_ffffu32 {
            let v = half::f16::from_f32(f32::from_bits(bits));
            h = (h ^ u64::from(v.to_bits())).wrapping_mul(1099511628211);
        }
    } else {
        for bits in 0..0x10000u32 {
            let f = half::f16::from_bits(bits as u16).to_f32();
            h = (h ^ u64::from(f.to_bits())).wrapping_mul(1099511628211);
        }
    }
    println!("{h:016x}");
}
