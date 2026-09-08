//! Checks the pixel conversions against the C.
//!
//!   pixels_dump all                       -- every f32 bit pattern through unit_to_byte, as one hash
//!   pixels_dump patches <in> <out> <gh> <gw> <w>
//!
//! `all` is exhaustive: all 2^32 inputs, folded into an FNV-1a 64 in bit-pattern order. NaNs,
//! subnormals, infinities and the halfway points are all in there, so the guard and the f64 rounding
//! either match the C everywhere or the hash differs.
use h3::pixels;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a[0].as_str() {
        "all" => {
            let mut h: u64 = 14695981039346656037;
            for bits in 0..=0xffff_ffffu32 {
                h = (h ^ u64::from(pixels::unit_to_byte(f32::from_bits(bits))))
                    .wrapping_mul(1099511628211);
            }
            println!("{h:016x}");
        }
        "patches" => {
            let (gh, gw, w): (usize, usize, usize) = (
                a[3].parse().unwrap(),
                a[4].parse().unwrap(),
                a[5].parse().unwrap(),
            );
            let bytes = std::fs::read(&a[1]).expect("input");
            let px: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let p = pixels::vision_patches(&px, gh, gw, w);
            let mut out = Vec::with_capacity(p.len() * 4);
            for v in &p {
                out.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&a[2], out).expect("output");
        }
        other => panic!("unknown mode {other}"),
    }
}
