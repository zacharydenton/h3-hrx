//! Runs the vision tower over one image with the Rust implementation.
//!
//!   vision_embed <text_encoder.safetensors> <pixels.f32> <merged.f32> <deepstack.f32> <h> <w>
//!
//! Pixels are `[h][w][3]` f32 in `[0, 1]`; the outputs are `[n/4][5120]` and `[3][n/4][5120]` f32, the
//! same arrays `h3_vision_embed` writes. The tower's weights live in the text encoder's
//! checkpoint, which is where Qwen3-VL keeps them.
use h3_hrx::compile::Compiler;
use h3_hrx::dispatch::Profile;
use h3_hrx::weights::Weights;
use std::io::Write;

fn write(path: &str, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("output"));
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

pub fn run(args: Vec<String>) {
    let a = args;
    let (height, width): (usize, usize) = (a[4].parse().unwrap(), a[5].parse().unwrap());

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("kernels"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let weights =
        unsafe { Weights::open(&a[0], h3_hrx::plan::te::plan) }.expect("text encoder checkpoint");

    let pixels: Vec<f32> = std::fs::read(&a[1])
        .expect("pixels")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(pixels.len(), height * width * 3, "wrong pixel count");

    let mut prof = Profile::from_env();
    let start = std::time::Instant::now();
    let e = h3_hrx::vision::embed(
        &mut stream,
        &compiler,
        &mut prof,
        &weights,
        &pixels,
        height,
        width,
    )
    .expect("embed");
    eprintln!(
        "embedded {height}x{width} to {} tokens in {:.2}s",
        e.tokens,
        start.elapsed().as_secs_f64()
    );
    write(&a[2], &e.merged);
    write(&a[3], &e.deepstack);
}
