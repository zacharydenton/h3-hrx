//! Dumps one of the four rotary tables.
//!
//!   rope_dump dit    <out> <rows> <positions>
//!   rope_dump te     <out> <n> [start count merged_h merged_w]...
//!   rope_dump vision <out> <gh> <gw>
//!   rope_dump vae    <out> <ft> <h> <w>
//!
//! The companion harness builds the same tables from lines sliced out of the C implementation (deleted; see git history), and the
//! two dumps are compared bitwise. The DiT's table is the one that matters most here: its angle is a
//! `float`, so a port that reached for `f64` would differ in the last bits everywhere.
use h3::model::*;
use h3::rope::{self, VisionSpan};
use std::io::Write;

fn put(path: &str, cos: &[f32], sin: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("output"));
    for v in cos.iter().chain(sin) {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let out = &a[1];
    match a[0].as_str() {
        "dit" => {
            let rows: usize = a[2].parse().unwrap();
            let bytes = std::fs::read(&a[3]).expect("positions");
            let pos: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            // the same frequencies the checkpoint stores: theta 1e4 over 32
            let inv: Vec<f32> = (0..16)
                .map(|j| 10_000.0f64.powf(-((2 * j) as f64) / 32.0) as f32)
                .collect();
            let (mut c, mut s) = (
                vec![0.0f32; rows * ROPE_HALF],
                vec![0.0f32; rows * ROPE_HALF],
            );
            rope::dit(&pos, &inv, &mut c, &mut s);
            put(out, &c, &s);
        }
        "te" => {
            let n: usize = a[2].parse().unwrap();
            let spans: Vec<VisionSpan> = a[3..]
                .chunks(4)
                .map(|q| VisionSpan {
                    start: q[0].parse().unwrap(),
                    count: q[1].parse().unwrap(),
                    merged_h: q[2].parse().unwrap(),
                    merged_w: q[3].parse().unwrap(),
                })
                .collect();
            let pos = rope::mrope_positions(n, &spans);
            let (mut c, mut s) = (
                vec![0.0f32; n * TE_ROPE_HALF],
                vec![0.0f32; n * TE_ROPE_HALF],
            );
            rope::te(&pos, &mut c, &mut s);
            put(out, &c, &s);
            let mut f = std::fs::File::create(format!("{out}.pos")).unwrap();
            for v in &pos {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        "vision" => {
            let (gh, gw): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
            let (mut c, mut s) = (vec![0.0f32; gh * gw * 36], vec![0.0f32; gh * gw * 36]);
            rope::vision(gh, gw, &mut c, &mut s);
            put(out, &c, &s);
        }
        "vae" => {
            let (ft, h, w): (usize, usize, usize) = (
                a[2].parse().unwrap(),
                a[3].parse().unwrap(),
                a[4].parse().unwrap(),
            );
            let n = ft * h * w;
            // the register and cls rows keep the identity rotation, so the buffers start there
            let (mut c, mut s) = (
                vec![1.0f32; n * VAE_ROPE_HALF],
                vec![0.0f32; n * VAE_ROPE_HALF],
            );
            rope::vae(ft, h, w, &mut c, &mut s);
            put(out, &c, &s);
        }
        other => panic!("unknown table {other}"),
    }
}
