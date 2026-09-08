//! Runs one of the tiling pieces over synthetic inputs.
//!
//!   tiles_dump split  <len>...
//!   tiles_dump latent <in> <out> <ah> <aw> <bh> <bw> <ext> <ydim> <t>
//!   tiles_dump pixels <in> <out> <frames> <th> <tw> <extent> <vertical>
//!   tiles_dump xfade  <in> <out> <plane> <nf> <overlap_frames> <ov>
//!
//! The companion harness runs the same four with `split_tiles` and all three blend lambdas sliced out
//! of `host/h3pipe.cpp`. They are separate functions there too, and this is what proves they stay
//! separate: a port that unified them would pass one case and fail the others.
use h3::model::LATENT_CH;
use h3::tiles;
use std::io::{Read, Write};

fn read(path: &str) -> Vec<f32> {
    let mut v = Vec::new();
    std::fs::File::open(path)
        .expect("input")
        .read_to_end(&mut v)
        .unwrap();
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write(path: &str, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("output"));
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n = |i: usize| -> usize { a[i].parse().expect("a number") };
    match a[0].as_str() {
        "split" => {
            for len in &a[1..] {
                let (starts, overlaps) = tiles::split_tiles(len.parse().unwrap());
                let s: Vec<String> = starts.iter().map(|v| v.to_string()).collect();
                let o: Vec<String> = overlaps.iter().map(|v| v.to_string()).collect();
                println!("{len} starts {} overlaps {}", s.join(" "), o.join(" "));
                // the C prints a trailing space before an empty list, which the join above does not
            }
        }
        "latent" => {
            let (ah, aw, bh, bw) = (n(3), n(4), n(5), n(6));
            let (ext, ydim, t) = (n(7), n(8) != 0, n(9));
            let all = read(&a[1]);
            let split = LATENT_CH * t * ah * aw;
            let r = tiles::blend_latent(&all[..split], ah, aw, &all[split..], bh, bw, ext, ydim, t);
            write(&a[2], &r);
        }
        "pixels" => {
            let (frames, th, tw, extent, vertical) = (n(3), n(4), n(5), n(6), n(7) != 0);
            let all = read(&a[1]);
            let split = 3 * frames * th * tw;
            let (nb, mut tile) = (all[..split].to_vec(), all[split..].to_vec());
            tiles::blend_pixels(&mut tile, &nb, extent, vertical, frames, th, tw);
            write(&a[2], &tile);
        }
        "xfade" => {
            let (plane, nf, overlap_frames, ov) = (n(3), n(4), n(5), n(6));
            let all = read(&a[1]);
            let split = 3 * ov * plane;
            let (overlap, mut chunk) = (all[..split].to_vec(), all[split..].to_vec());
            tiles::crossfade(&mut chunk, nf, &overlap, plane, overlap_frames);
            write(&a[2], &chunk);
        }
        other => panic!("unknown piece {other}"),
    }
}
