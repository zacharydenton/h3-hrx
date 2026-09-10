//! Builds a stack and reports which of its kernels were already in the cache.
//!
//!   verify_stack_kernels <checkpoint> <tokens>
//!
//! The C implementation filled the cache for a given shape. If the Rust stack, built for the same
//! shape, asks for kernels that are all already there, then every decision it made along the way — the
//! operand element type, the tile, the row groups, the pitches, the attention stem, whether each
//! projection needs a prepare — matches what the C decided. A single wrong choice appears as a compile.
use h3::compile::Compiler;
use h3::model::*;
use h3::stack::{Stack, StackDims};
use h3::weights::Weights;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: verify_stack_kernels <checkpoint> <tokens>");
    let tokens: usize = args.next().expect("tokens").parse().expect("a number");

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let cache = root.join("build/kernel_cache");
    let before: std::collections::HashSet<String> = std::fs::read_dir(&cache)
        .expect("cache")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();

    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("h3/kernels"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let weights = unsafe { Weights::open(&path, h3::plan::vvae::plan) }.expect("plan");

    // The video decoder's stack, exactly as the C constructs it.
    let dims = StackDims {
        hidden: VAE_HID,
        heads: VAE_HEADS,
        kv_heads: VAE_HEADS,
        head_dim: VAE_D,
        ffn: VAE_FFN,
        rope_dim: 48,
        classes: 1,
        wbits: 16,
        eps: 1e-5,
        bias: true,
        gate_first: false,
        causal: false,
        attn_i4: false,
        attn_qk_bits: 16,
        bf16: false,
    };
    // qnorm and knorm are ones for this stack: it has no per-head norm weights.
    let ones = std::sync::Arc::new(stream.allocate(VAE_D * 4).expect("ones"));
    let one_bytes: Vec<u8> = (0..VAE_D).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    stream
        .upload(ones.binding(), &one_bytes)
        .expect("upload ones");

    let stack = Stack::new(
        &compiler,
        &mut stream,
        dims,
        tokens,
        VAE_BLOCKS,
        &weights,
        |i| format!("blocks.{i}."),
        false,
        ones,
        "vae",
    )
    .expect("stack");

    let after: Vec<String> = std::fs::read_dir(&cache)
        .expect("cache")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.ends_with(".hsaco") && !before.contains(n))
        .collect();

    println!(
        "decoder stack at {tokens} tokens, capacity {}: {} kernels newly compiled",
        stack.capacity(),
        after.len()
    );
    for name in &after {
        println!("  compiled: {name}");
    }
    if !after.is_empty() {
        std::process::exit(1);
    }
    println!("every kernel it asked for was already in the cache");
}
