//! Moves a session's pieces to another thread and dispatches there.
//!
//! `Send` on the runtime handles is a safety claim, so it is worth more than a reading of the headers.
//! This builds a GEMM on the main thread, hands the GPU, the kernel and the buffers to a second thread,
//! runs the kernel there, reads the result back there, and checks it — then hands what is left to a
//! third thread so the allocations are also released somewhere other than where they were made.
use h3::compile::Compiler;
use h3::dispatch::{Gemm, Tile};
use h3::model::gemm_pitch;
use half::f16;

fn assert_send<T: Send>() {}

fn main() {
    // A compile-time statement of the claim, before any of it runs.
    assert_send::<hrx::Stream>();
    assert_send::<hrx::Buffer>();
    assert_send::<hrx::Kernel>();
    assert_send::<Gemm>();

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("h3/kernels"));

    let (m, k, n) = (64usize, 2048usize, 2048usize);
    let k_stride = gemm_pitch(k, 16);
    // A is all ones and W's row j is all j/K, so every output element has a known value: the reference
    // needs no float64 pass and a wrong dispatch is obvious.
    let a: Vec<f16> = (0..m * k_stride)
        .map(|i| {
            if i % k_stride < k {
                f16::ONE
            } else {
                f16::from_f32(113.0)
            }
        })
        .collect();
    let w: Vec<f16> = (0..n * k_stride)
        .map(|i| {
            if i % k_stride < k {
                f16::from_f32((i / k_stride) as f32 / k as f32)
            } else {
                f16::from_f32(113.0)
            }
        })
        .collect();
    let bias: Vec<f32> = vec![0.0; n];

    let bytes = |v: &[f16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let a_buf = stream.allocate(m * k_stride * 2).expect("a");
    let w_buf = stream.allocate(n * k_stride * 2).expect("w");
    let b_buf = stream.allocate(n * 4).expect("b");
    let out_buf = stream.allocate(m * n * 2).expect("out");
    stream
        .upload(a_buf.binding(), &bytes(&a))
        .expect("upload a");
    stream
        .upload(w_buf.binding(), &bytes(&w))
        .expect("upload w");
    stream
        .upload(
            b_buf.binding(),
            &bias
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("upload b");
    stream.fill(out_buf.slice(0, m * n * 2), 0).expect("clear");

    let gemm = Gemm::build(
        &compiler,
        &mut stream,
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
    drop(compiler); // the compiler stays behind; only the runtime handles travel

    // Everything the dispatch needs moves to a second thread.
    let (stream, out_buf) = std::thread::spawn(move || {
        gemm.run(
            &mut stream,
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
        .expect("run on another thread");

        let mut raw = vec![0u8; m * n * 2];
        stream
            .read_blocking(out_buf.binding(), &mut raw)
            .expect("read back");
        let got: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        // row of ones dotted with a constant row j/K over K terms is exactly j
        let mut worst = 0.0f32;
        for row in 0..m {
            for col in 0..n {
                worst = worst.max((got[row * n + col] - col as f32).abs() / (col as f32).max(1.0));
            }
        }
        assert!(
            worst < 1e-2,
            "dispatch from another thread gave wrong values: rel {worst}"
        );
        println!("dispatched and read back on a second thread: worst relative error {worst:.2e}");
        (stream, out_buf)
    })
    .join()
    .expect("second thread");

    // And released on a third, so the allocation and the device are both dropped away from where they
    // were made. The device outlives the buffer because the buffer holds a reference to it.
    std::thread::spawn(move || {
        drop(out_buf);
        drop(stream);
        println!("buffer and device released on a third thread");
    })
    .join()
    .expect("third thread");
}
