//! Dumps a checkpoint's tensor table and a digest of each tensor's bytes, for comparing this reader
//! against an independent one:  cargo run --release --example checkpoint_dump -- file.safetensors
use h3::checkpoint::Checkpoint;

/// A position-sensitive checksum: the byte sum plus a sum weighted by index, so a shifted or
/// truncated span cannot collide with the right one. Both are cheap to reproduce with numpy.
fn digest(bytes: &[u8]) -> (u64, u64) {
    let mut sum: u64 = 0;
    let mut weighted: u64 = 0;
    for (i, b) in bytes.iter().enumerate() {
        sum = sum.wrapping_add(u64::from(*b));
        weighted = weighted.wrapping_add((i as u64).wrapping_mul(u64::from(*b)));
    }
    (sum, weighted)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: checkpoint_dump file.safetensors");
    let ck = Checkpoint::open(&path).expect("open");
    for (name, entry) in ck.entries() {
        let bytes = ck.bytes(entry);
        let (sum, weighted) = digest(bytes);
        println!(
            "{name}\t{:?}\t{:?}\t{}\t{}\t{sum}\t{weighted}",
            entry.dtype, entry.shape, entry.offset, entry.bytes
        );
        ck.done_with(bytes);
    }
}
