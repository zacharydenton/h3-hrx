//! Runs the vision tower over one image with the Rust implementation.
//!
//!   vision_embed <text_encoder.safetensors> <pixels.f32> <merged.f32> <deepstack.f32> <h> <w>
//!
//! Pixels are `[h][w][3]` f32 in `[0, 1]`; the outputs are `[n/4][5120]` and `[3][n/4][5120]` f32, the
//! same arrays `h3_vision_embed` writes. The tower's weights live in the text encoder's
//! checkpoint, which is where Qwen3-VL keeps them.
use h3::compile::Compiler;
use h3::dispatch::Profile;
use h3::weights::Weights;
use std::io::Write;

fn write(path: &str, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("output"));
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (height, width): (usize, usize) = (a[4].parse().unwrap(), a[5].parse().unwrap());

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let exe = std::env::var("LOOM_COMPILE").unwrap_or_else(|_| "loom-compile".into());
    let gpu = hrx::Gpu::open().expect("gpu");
    let compiler = Compiler::new(exe, root.join("kernels"), root.join("build/kernel_cache"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let weights =
        unsafe { Weights::open(&a[0], h3::plan::te::plan) }.expect("text encoder checkpoint");

    let pixels: Vec<f32> = std::fs::read(&a[1])
        .expect("pixels")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(pixels.len(), height * width * 3, "wrong pixel count");

    let mut prof = Profile::from_env();
    let start = std::time::Instant::now();
    let e = h3::vision::embed(&gpu, &compiler, &mut prof, &weights, &pixels, height, width)
        .expect("embed");
    eprintln!(
        "embedded {height}x{width} to {} tokens in {:.2}s",
        e.tokens,
        start.elapsed().as_secs_f64()
    );
    write(&a[2], &e.merged);
    write(&a[3], &e.deepstack);
}
