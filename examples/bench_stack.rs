//! Synchronized H3 DiT block-range benchmark with fixed real weights and inputs.
//! Usage: bench_stack CHECKPOINT TOKENS LAYERS SAMPLES OUTPUT [eager|graph]
//! Loading, input reset and readback are outside the timed forward. The result
//! is a transformer-stack measurement, not complete generation latency.
use h3_hrx::{
    compile::Compiler,
    dispatch::{Classes, Profile, Sink},
    model::*,
    stack::{Constants, LayerCond, Stack, StackDims},
    weights::Weights,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(6..=7).contains(&args.len()) {
        return Err(
            "usage: bench_stack CHECKPOINT TOKENS LAYERS SAMPLES OUTPUT [eager|graph]".into(),
        );
    }
    let tokens: usize = args[2].parse()?;
    let layers: usize = args[3].parse()?;
    let samples: usize = args[4].parse()?;
    let graph_mode = match args.get(6).map(String::as_str).unwrap_or("eager") {
        "eager" => false,
        "graph" => true,
        _ => return Err("expected eager or graph".into()),
    };
    if !(1..=65536).contains(&tokens) || !(1..=BLOCKS).contains(&layers) || samples < 3 {
        return Err("invalid tokens, layers or samples".into());
    }
    let residency = hrx::residency::ResidencyManager::new(80 << 30)?;
    let mut stream = hrx::Stream::open()?.with_memory_budget(residency.budget());
    let compiler = Compiler::new(
        None,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    );
    // SAFETY: The caller supplies an immutable checkpoint for this process.
    let weights = unsafe { Weights::open(&args[1], h3_hrx::plan::dit::plan) }?;
    let constants = Constants::new(&mut stream)?;
    let dims = StackDims {
        hidden: HID,
        heads: HEADS,
        kv_heads: HEADS,
        head_dim: HEAD_DIM,
        ffn: FFN,
        rope_dim: ROPE_DIM,
        classes: CLASSES,
        wbits: 8,
        eps: 1e-5,
        bias: false,
        gate_first: true,
        causal: false,
        attn_i4: false,
        attn_qk_bits: 8,
        bf16: false,
    };
    let mut stack = Stack::new(
        &compiler,
        &mut stream,
        dims,
        tokens,
        layers,
        &weights,
        |i| format!("blocks.{i}."),
        true,
        constants.ones.clone(),
        "bench",
    )?;
    let x = stream.allocate_zeroed(stack.capacity() * HID * 4)?;
    let mut classes = Classes::zeroed(&mut stream, tokens)?;
    classes.write(
        &mut stream,
        &(0..tokens)
            .map(|r| (r % CLASSES) as i32)
            .collect::<Vec<_>>(),
        CLASSES,
    )?;
    let upload = |stream: &mut hrx::Stream, data: Vec<f32>| -> hrx::Result<hrx::Buffer> {
        let b = stream.allocate(data.len() * 4)?;
        stream.upload_blocking(b.binding(), bytemuck::cast_slice(&data))?;
        Ok(b)
    };
    let cos = upload(
        &mut stream,
        (0..tokens * ROPE_HALF)
            .map(|i| ((i % 271) as f32 / 271.).cos())
            .collect(),
    )?;
    let sin = upload(
        &mut stream,
        (0..tokens * ROPE_HALF)
            .map(|i| ((i % 271) as f32 / 271.).sin())
            .collect(),
    )?;
    let table = upload(
        &mut stream,
        (0..CLASSES * 2 * HID)
            .map(|i| (i % 13) as f32 / 100.)
            .collect(),
    )?;
    let gate = upload(
        &mut stream,
        (0..CLASSES * HID)
            .map(|i| 0.1 + (i % 7) as f32 / 20.)
            .collect(),
    )?;
    let cond = |_| LayerCond {
        table_msa: table.binding(),
        gate_msa: gate.binding(),
        table_mlp: table.binding(),
        gate_mlp: gate.binding(),
    };
    let input: Vec<f32> = (0..tokens * HID)
        .map(|i| ((i * 17) % 127) as f32 / 64. - 1.)
        .collect();
    let mut times = Vec::new();
    let mut expected = Vec::new();
    let mut graph = None;
    let mut profile = Profile::from_env();
    for iteration in 0..samples + 2 {
        stream.upload_blocking(x.binding(), bytemuck::cast_slice(&input))?;
        stream.synchronize()?;
        let start = Instant::now();
        if let Some(recorded) = &mut graph {
            stream.launch(recorded)?;
        } else {
            stack.forward(
                &mut stream,
                &mut profile,
                x.binding(),
                classes.all(),
                cos.binding(),
                sin.binding(),
                &cond,
                0,
                None,
            )?;
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1000.;
        if iteration >= 2 {
            times.push(ms);
        }
        let mut actual = vec![0u8; tokens * HID * 4];
        stream.read_blocking(x.binding(), &mut actual)?;
        if actual
            .as_chunks::<4>()
            .0
            .iter()
            .any(|v| !f32::from_le_bytes(*v).is_finite())
        {
            return Err("nonfinite output".into());
        }
        if iteration == 0 {
            expected = actual;
        } else if actual != expected {
            return Err("replay differs".into());
        }
        if graph_mode && graph.is_none() {
            let mut recording = stream.graph()?;
            stack.emit(
                &mut Sink::Graph {
                    graph: &mut recording,
                    after: Default::default(),
                },
                &mut Profile::default(),
                x.binding(),
                classes.all(),
                cos.binding(),
                sin.binding(),
                &cond,
                0,
                None,
                false,
            )?;
            graph = Some(recording.finish()?);
        }
    }
    std::fs::write(&args[5], &expected)?;
    let distribution = hrx::benchmark::Distribution::from_samples(times.clone())?;
    println!(
        "{}",
        serde_json::json!({"tokens":tokens,"capacity":stack.capacity(),"layers":layers,"samples":samples,"graph":graph_mode,"median_ms":distribution.median_ms,"warm_ms":times,"reserved_bytes":residency.statistics().reserved_bytes,"output_sha256":hrx::bundle::digest(&expected),"scope":"resident transformer stack forward","profile":profile.take()})
    );
    Ok(())
}
