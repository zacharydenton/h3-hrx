//! Replays a step cache's decisions from a file of per-step change measurements.
//!
//!   cache_dump <threshold> <measurements>
//!
//! The measurement file is `nsteps ngroups` followed by each step's partial sums, the pairs the
//! device's reduction hands back. The companion harness runs the same replay with the decision
//! arithmetic sliced out of `host/h3pipe.cpp`; the two outputs are diffed line for line.
use h3::cache::StepCache;
use h3::compile::num;

fn main() {
    let mut args = std::env::args().skip(1);
    let threshold: f32 = args
        .next()
        .expect("usage: cache_dump <threshold> <measurements>")
        .parse()
        .expect("a number");
    let text = std::fs::read_to_string(args.next().expect("measurements")).expect("read");
    let mut nums = text.split_ascii_whitespace();
    let mut next = || nums.next().expect("short file");
    let nsteps: usize = next().parse().unwrap();
    let groups: usize = next().parse().unwrap();

    let mut cache = StepCache::new(threshold).expect("a positive threshold");
    for step in 0..nsteps {
        // summed in the reduction's own order, f32 partials into f64 accumulators
        let (mut d, mut m) = (0.0f64, 0.0f64);
        for _ in 0..groups {
            d += f64::from(next().parse::<f32>().unwrap());
            m += f64::from(next().parse::<f32>().unwrap());
        }
        let ch = cache.consider(step, d, m);
        if !ch.skip {
            cache.recorded();
        }
        // %.17g, the same spelling the C's printf uses, so the two dumps are comparable as text
        println!(
            "step {step} rel {} acc {} {}",
            num(ch.relative),
            num(ch.accumulated),
            if ch.skip { "cached" } else { "full" }
        );
    }
    println!("skipped {}", cache.skipped());
}
