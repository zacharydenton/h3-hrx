//! Runs the prompt through the encoder, condition_proj and the refiner.
//!
//!   text_in <dit.safetensors> <text_encoder.safetensors> <ids.i32> <out.f32> [<ids> <out>]...
//!
//! Ids are int32; each output is `[n][5376]` f32, the same array `h3_text_in` writes. Several
//! pairs run in one process because opening the encoder is the expensive part, not the pass. No vision
//! spans — those go through the tower first, and this exercises the text path on its own.
use h3::compile::Compiler;
use h3::dispatch::Profile;
use h3::dit::Dit;
use h3::model::HID;
use h3::te::TextEncoder;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("kernels"));

    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.

    let mut dit = unsafe { Dit::open(&mut stream, &a[0]) }.expect("DiT checkpoint");
    let mut te = unsafe { TextEncoder::open(&mut stream, &a[1]) }.expect("text encoder checkpoint");
    let mut prof = Profile::from_env();

    for pair in a[2..].chunks(2) {
        let ids: Vec<i32> = std::fs::read(&pair[0])
            .expect("ids")
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let start = std::time::Instant::now();
        dit.text_in(&mut stream, &compiler, &mut prof, &mut te, &ids, &[])
            .expect("text_in");
        let mut out = vec![0.0f32; ids.len() * HID];
        dit.read_rows(&mut stream, ids.len(), &mut out)
            .expect("read");
        eprintln!(
            "encoded {} tokens in {:.2}s",
            ids.len(),
            start.elapsed().as_secs_f64()
        );
        let mut f = std::io::BufWriter::new(std::fs::File::create(&pair[1]).expect("output"));
        for v in &out {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.flush().unwrap();
    }
}
