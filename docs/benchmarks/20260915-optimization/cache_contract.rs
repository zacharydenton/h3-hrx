//! CPU-only reproduction of the caller sequence in Dit::run_blocks at f212ae8.
//! Regression check: recording step zero must not bypass the second full evaluation.
#![allow(dead_code)]

#[path = "../../../src/cache.rs"]
mod cache;

fn main() {
    let mut cache = cache::StepCache::configured(cache::CachePolicy::Off, 0.25).unwrap().unwrap();
    assert!(!cache.consider(0, 20, [[0.0, 0.0]; 3]).skip);
    // run_blocks records the residual after every full suffix, including step 0.
    cache.recorded();
    let second = cache.consider(1, 20, [[0.01, 1.0]; 3]);
    println!("second evaluation skips: {}", second.skip);
    assert!(!second.skip, "warmup must include the second evaluation");
}
