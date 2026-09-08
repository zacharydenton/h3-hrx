//! Prints the decoder's temporal chunk plan for each latent frame count given.
//!
//! The companion harness prints the same from lines sliced out of `host/h3pipe.cpp`.
use h3::tiles::chunk_plan;

fn main() {
    for a in std::env::args().skip(1) {
        let t: usize = a.parse().expect("a number");
        let p = chunk_plan(t);
        println!(
            "{t} pad {} padded {} chunks {} frames {} pre {} overlap {} padframes {}",
            p.pad_tokens,
            p.padded_tokens,
            p.chunks,
            p.chunk_frames,
            p.pre,
            p.overlap_frames,
            p.pad_frames
        );
    }
}
