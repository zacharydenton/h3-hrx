#[allow(dead_code, unused_imports)]
#[path = "../src/bin/h3/media.rs"]
mod media;
#[allow(dead_code)]
#[path = "support/render.rs"]
mod render;
mod support;
use criterion::{
    measurement::{Measurement, ValueFormatter},
    Criterion, Throughput,
};
use h3_hrx::ResidencyPolicy;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

/// iter_custom sums one peak per render, so Criterion reports bytes/render.
struct Bytes;
impl Measurement for Bytes {
    type Intermediate = ();
    type Value = u64;
    fn start(&self) {}
    fn end(&self, _: ()) -> u64 {
        panic!("peak memory requires iter_custom")
    }
    fn add(&self, a: &u64, b: &u64) -> u64 {
        a + b
    }
    fn zero(&self) -> u64 {
        0
    }
    fn to_f64(&self, value: &u64) -> f64 {
        *value as f64
    }
    fn formatter(&self) -> &dyn ValueFormatter {
        self
    }
}
impl ValueFormatter for Bytes {
    fn scale_values(&self, _: f64, values: &mut [f64]) -> &'static str {
        for value in values {
            *value /= (1u64 << 20) as f64;
        }
        "MiB/render"
    }
    fn scale_throughputs(&self, _: f64, _: &Throughput, _: &mut [f64]) -> &'static str {
        unreachable!("no throughput configured")
    }
    fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
        "bytes/render"
    }
}

struct Peak {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<(u64, u64)>>,
}
impl Peak {
    fn start(manager: hrx::residency::ResidencyManager) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = std::thread::spawn(move || {
            let mut gpu = 0;
            let mut rss = 0;
            loop {
                gpu = gpu.max(manager.statistics().reserved_bytes as u64);
                let status = std::fs::read_to_string("/proc/self/status").unwrap();
                let kb = status
                    .lines()
                    .find_map(|line| line.strip_prefix("VmRSS:"))
                    .expect("Linux VmRSS")
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                rss = rss.max(kb * 1024);
                if stopping.load(Ordering::Relaxed) {
                    return (gpu, rss);
                }
                std::thread::park_timeout(Duration::from_millis(2));
            }
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }
    fn finish(mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        let worker = self.worker.take().unwrap();
        worker.thread().unpark();
        worker.join().unwrap()
    }
}
impl Drop for Peak {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

fn memory(c: &mut Criterion<Bytes>) {
    let (profile, params) = support::params();
    let mut group = c.benchmark_group(format!("memory/{profile}/{}", support::execution_id()));
    for case in render::cases()
        .into_iter()
        .filter(|c| matches!(c.name, "text/res_multistep" | "image_audio_references"))
    {
        for policy in [ResidencyPolicy::StageScoped, ResidencyPolicy::Budgeted] {
            for rss in [false, true] {
                let mut state = None;
                let metric = if rss {
                    "sampled_peak_process_rss"
                } else {
                    "sampled_peak_reservations"
                };
                group.bench_function(format!("{}/{policy:?}/{metric}", case.name), |b| {
                    let (render, runtime, expected) = state.get_or_insert_with(|| {
                        (
                            render::Render::new(case, params, policy),
                            support::Runtime::new(),
                            None,
                        )
                    });
                    b.iter_custom(|iterations| {
                        let mut peaks = 0;
                        for _ in 0..iterations {
                            let monitor = Peak::start(runtime.manager.clone());
                            let mut session =
                                runtime.session(render.config.clone(), render.options);
                            let output = render.run(&mut session);
                            drop(session);
                            let (gpu, host) = monitor.finish();
                            assert!(gpu > 0 && gpu <= support::budget_bytes() as u64);
                            assert_eq!(
                                runtime.manager.statistics().reserved_bytes,
                                0,
                                "session leaked allocations"
                            );
                            let actual = output.digest();
                            if let Some(expected) = expected.as_ref() {
                                assert_eq!(&actual, expected, "memory run changed native output");
                            } else {
                                *expected = Some(actual);
                            }
                            render.check_media();
                            peaks += if rss { host } else { gpu };
                        }
                        peaks
                    });
                });
            }
        }
    }
    group.finish();
}
fn main() {
    let mut criterion = support::criterion()
        .with_measurement(Bytes)
        .configure_from_args();
    memory(&mut criterion);
    criterion.final_summary();
}
