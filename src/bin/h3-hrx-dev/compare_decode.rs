//! Compare resident decoders using two explicit source directories and identical latents.
//! compare_decode BASELINE_DIR CHECKPOINT LATENTS HEIGHT WIDTH FRAMES
//! Baseline modules contain the original exports, grouped into the current module files.
//! H3_BASELINE_LOOM_LIBRARY optionally compares a compiler upgrade as well.
use h3_hrx::{compile::Compiler, dispatch::Profile, layout::shape_for, vvae::VideoVae};
use std::{path::Path, time::Instant};

pub fn run(args: Vec<String>) {
    let a: Vec<String> = std::iter::once("h3-hrx-dev".to_string())
        .chain(args)
        .collect();
    assert_eq!(
        a.len(),
        7,
        "BASELINE_DIR CHECKPOINT LATENTS HEIGHT WIDTH FRAMES"
    );
    let shape = shape_for(
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
        a[6].parse().unwrap(),
    )
    .unwrap();
    let input = std::fs::read(&a[3]).unwrap();
    assert!(input.len().is_multiple_of(4));
    let latents: Vec<_> = input
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        latents.len(),
        24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let baseline_exe = std::env::var_os("H3_BASELINE_LOOM_LIBRARY")
        .map(std::path::PathBuf::from)
        .or_else(|| exe.clone());
    let compilers = [Compiler::new(baseline_exe, &a[1]), Compiler::new(exe, root)];
    let mut stream = hrx::Stream::open().unwrap();
    // Safety: caller-owned checkpoint, which is not modified during this comparison.
    let mut vaes = [
        unsafe { VideoVae::open(&mut stream, &a[2]).unwrap() },
        unsafe { VideoVae::open(&mut stream, &a[2]).unwrap() },
    ];
    let mut profile = Profile::default();
    let bytes = shape.frames as usize * shape.lat_h as usize * 16 * shape.lat_w as usize * 16 * 3;
    let mut outputs = [vec![0; bytes], vec![0; bytes]];
    for version in 0..2 {
        vaes[version]
            .decode_video(
                &mut stream,
                &compilers[version],
                &mut profile,
                &shape,
                &latents,
                &mut outputs[version],
            )
            .unwrap();
    }
    assert!(outputs[0] == outputs[1], "decoded RGB differs");
    println!("batch,baseline_seconds,candidate_seconds,ratio");
    for batch in 0..10 {
        let mut times = [0.; 2];
        for order in 0..2 {
            let version = (batch + order) % 2;
            stream.synchronize().unwrap();
            let start = Instant::now();
            vaes[version]
                .decode_video(
                    &mut stream,
                    &compilers[version],
                    &mut profile,
                    &shape,
                    &latents,
                    &mut outputs[version],
                )
                .unwrap();
            stream.synchronize().unwrap();
            times[version] = start.elapsed().as_secs_f64();
        }
        assert!(
            outputs[0] == outputs[1],
            "decoded RGB differs in batch {batch}"
        );
        println!(
            "{batch},{:.6},{:.6},{:.5}",
            times[0],
            times[1],
            times[1] / times[0]
        );
    }
}
