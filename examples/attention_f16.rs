//! Focused attention schedules on identical bindings, including graph replay.
//! Run independent `attention_matches_scaled_dot_product_for_the_shipped_layouts`
//! first. This diagnostic does not qualify a complete model or a default change.
use half::f16;
use hrx::{
    loom::{Compiler, Specialization},
    Constants, Stream,
};
use std::{path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |i: usize, default: usize| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(args
            .get(i)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let (tokens, heads, rounds, reps) = (
        number(0, 4096)?,
        number(1, 4)?,
        number(2, 5)?,
        number(3, 3)?,
    );
    if tokens == 0 || tokens > 65536 || heads == 0 || heads > 256 || rounds < 3 || reps == 0 {
        return Err(
            "usage: attention_f16 [tokens:1..65536] [heads:1..256] [rounds>=3] [reps>0]".into(),
        );
    }
    let capacity = (tokens + 32).div_ceil(128) * 128;
    let stride = heads * 128;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let compiler = Compiler::resolve(None)?;
    let mut stream = Stream::open()?;
    let mut buffers = Vec::new();
    for salt in [17, 41, 71] {
        let mut input = vec![f16::ZERO; capacity * stride];
        for (i, value) in input[..tokens * stride].iter_mut().enumerate() {
            *value = f16::from_f32((((i * 71 + salt) % 277) as f32 / 138. - 1.) * 0.5);
        }
        let buffer = stream.allocate(input.len() * 2)?;
        stream.upload(buffer.binding(), bytemuck::cast_slice(&input))?;
        buffers.push(buffer);
    }
    let output = stream.allocate(tokens * stride * 2)?;
    let mut bindings: Vec<_> = buffers.iter().map(|b| b.binding()).collect();
    bindings.push(output.binding());
    let mut kernels = Vec::new();
    for (module, stem, waves) in [
        ("attention_mha_family", "attention_mha8_lds_f16_wmma", 8),
        (
            "attention_mhat32_lds_f16_wmma",
            "attention_mhat32_lds_f16_wmma",
            4,
        ),
        (
            "attention_mha8t32_lds_f16_wmma",
            "attention_mha8t32_lds_f16_wmma",
            8,
        ),
        (
            "attention_mha8h2t32_lds_f16_wmma",
            "attention_mha8h2t32_lds_f16_wmma",
            8,
        ),
        (
            "attention_mha8h0t32_lds_f16_wmma",
            "attention_mha8h0t32_lds_f16_wmma",
            8,
        ),
    ] {
        let mut path = root.join("kernels").join(format!("{module}.loom"));
        if !path.exists() {
            path = root.join("experiments").join(format!("{module}.loom"));
        }
        let source = std::fs::read_to_string(&path)?;
        let mut request = Specialization::new(format!("h3_{stem}"));
        for (name, value) in [
            ("q_stride", stride),
            ("kv_stride", stride),
            ("out_stride", stride),
            ("tokens", tokens),
            ("token_capacity", capacity),
        ] {
            request.set_config(format!("h3.{stem}.{name}"), value.to_string());
        }
        request.set_config(
            format!("h3.{stem}.scale"),
            (1f32 / 128f32.sqrt()).to_string(),
        );
        let artifact = compiler.module(&source).compile(&request)?;
        eprintln!("{stem}: {}", artifact.path().display());
        // Trusted source with zero headroom for complete query/key tiles.
        let kernel = unsafe { stream.load_artifact(&artifact)? };
        eprintln!("{stem}: {:?}", kernel.info());
        let constants = Constants::indices(&kernel, &[tokens as u32])?;
        let mut graph = stream.graph()?;
        let mut after = Vec::new();
        for _ in 0..reps {
            let node = unsafe {
                graph.dispatch(
                    &after,
                    &kernel,
                    [tokens.div_ceil(16 * waves) as u32, heads as u32, 1],
                    [32 * waves as u32, 1, 1],
                    &constants,
                    &bindings,
                )?
            };
            after = vec![node];
        }
        kernels.push((stem, graph.finish()?));
    }
    let mut expected = vec![0u8; output.bytes()];
    stream.launch(&mut kernels[0].1)?;
    stream.read_blocking(output.binding(), &mut expected)?;
    let expected: Vec<_> = expected
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| f16::from_bits(u16::from_le_bytes(*v)).to_f64())
        .collect();
    if !expected.iter().all(|v| v.is_finite()) {
        return Err("nonfinite baseline output".into());
    }
    for candidate in 1..kernels.len() {
        stream.launch(&mut kernels[candidate].1)?;
        let mut bytes = vec![0; output.bytes()];
        stream.read_blocking(output.binding(), &mut bytes)?;
        let mut squared_error = 0.;
        let mut squared_reference = 0.;
        let mut max_error = 0f64;
        for (bytes, reference) in bytes.as_chunks::<2>().0.iter().zip(&expected) {
            let value = f16::from_bits(u16::from_le_bytes(*bytes)).to_f64();
            if !value.is_finite() {
                return Err("nonfinite candidate output".into());
            }
            squared_error += (value - reference).powi(2);
            squared_reference += reference.powi(2);
            max_error = max_error.max((value - reference).abs());
        }
        let relative_rms = (squared_error / squared_reference.max(1e-30)).sqrt();
        if !relative_rms.is_finite() || relative_rms > 0.003 {
            return Err(format!("attention relative RMS {relative_rms} exceeds 0.003").into());
        }
        for pair in 0..rounds {
            let mut times = [0.; 2];
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let index = if arm == 0 { 0 } else { candidate };
                stream.synchronize()?;
                let start = Instant::now();
                stream.launch(&mut kernels[index].1)?;
                stream.synchronize()?;
                times[arm] = start.elapsed().as_secs_f64() * 1000. / reps as f64;
            }
            println!(
                "{}",
                serde_json::json!({"candidate":kernels[candidate].0,"tokens":tokens,"heads":heads,
                "capacity":capacity,"pair":pair,"baseline_ms":times[0],"candidate_ms":times[1],
                "relative_rms":relative_rms,"max_abs_error":max_error,"shared_bindings":true,
                "scope":"instrumentation-free graph replay; resident reused inputs; no repacking",
                "promotion_qualified":false})
            );
        }
    }
    Ok(())
}
