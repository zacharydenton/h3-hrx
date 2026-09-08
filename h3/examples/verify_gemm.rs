//! Dispatches a GEMM through the Rust builder and checks it against a float64 reference.
//!
//! This is what proves the dispatch layer as distinct from the layout: the stem it picks, the config it
//! passes, the order it binds its operands in, and the grid it launches. A mistake in any of those
//! produces wrong numbers rather than an error.
use h3::compile::Compiler;
use h3::dispatch::{Gemm, Tile};
use h3::model::gemm_pitch;
use half::f16;

/// Rounds through f16 the way the operands are stored, so the reference sees the same inputs.
fn narrow(v: f64) -> f16 {
    f16::from_f64(v)
}

/// A small deterministic normal-ish generator, so the check needs no rand dependency.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let exe = std::env::var("LOOM_COMPILE").unwrap_or_else(|_| "loom-compile".into());
    let gpu = hrx::Gpu::open().expect("gpu");
    let compiler = Compiler::new(exe, root.join("kernels"), root.join("build/kernel_cache"));

    let (k, n) = (2048usize, 2048usize);
    let k_stride = gemm_pitch(k, 16);
    let mut failures = 0;

    for m in [64usize, 300, 517] {
        let mut rng = Rng(0x5eed_1234 + m as u64);
        // A is [m][k_stride], W is [n][k_stride]: the columns past K are never read, so they are
        // filled with a value that would be obvious in the output if they were.
        let mut a = vec![f16::ZERO; m * k_stride];
        let mut w = vec![f16::ZERO; n * k_stride];
        for row in 0..m {
            for col in 0..k_stride {
                a[row * k_stride + col] = if col < k {
                    narrow(rng.next_f64() * 0.5)
                } else {
                    narrow(113.0)
                };
            }
        }
        for row in 0..n {
            for col in 0..k_stride {
                w[row * k_stride + col] = if col < k {
                    narrow(rng.next_f64() / (k as f64).sqrt())
                } else {
                    narrow(113.0)
                };
            }
        }
        let bias: Vec<f32> = (0..n).map(|_| (rng.next_f64() * 0.1) as f32).collect();

        let bytes = |v: &[f16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let a_buf = gpu.alloc(m * k_stride * 2).expect("a");
        let w_buf = gpu.alloc(n * k_stride * 2).expect("w");
        let b_buf = gpu.alloc(n * 4).expect("b");
        let out_buf = gpu.alloc(m * n * 2).expect("out");
        gpu.h2d(&a_buf, &bytes(&a)).expect("upload a");
        gpu.h2d(&w_buf, &bytes(&w)).expect("upload w");
        gpu.h2d(
            &b_buf,
            &bias
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("upload b");
        gpu.memset(&out_buf, 0, m * n * 2).expect("clear");

        let gemm = Gemm::build(
            &compiler,
            &gpu,
            "plain",
            "f16",
            true,
            true,
            k,
            n,
            m,
            1,
            k_stride,
            Tile::Plain,
            0,
        )
        .expect("build");
        gemm.run(
            &gpu,
            None,
            "gemm",
            m as u32,
            a_buf.binding(),
            w_buf.binding(),
            None,
            out_buf.binding(),
            None,
            Some(b_buf.binding()),
        )
        .expect("run");
        gpu.sync().expect("sync");

        let mut raw = vec![0u8; m * n * 2];
        gpu.d2h(&out_buf, &mut raw).expect("read back");
        let got: Vec<f64> = raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect();

        let mut want = vec![0.0f64; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += a[row * k_stride + i].to_f64() * w[col * k_stride + i].to_f64();
                }
                want[row * n + col] = acc + f64::from(bias[col]);
            }
        }

        let cos = cosine(&got, &want);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f64, f64::max);
        let ok = cos > 0.9999;
        println!(
            "{} gemm_f16_256b m={m} k={k} n={n} k_stride={k_stride} group={}: cosine={cos:.8} max_abs={max_abs:.3e}",
            if ok { "PASS" } else { "FAIL" },
            gemm.m_group()
        );
        if !ok {
            failures += 1;
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
