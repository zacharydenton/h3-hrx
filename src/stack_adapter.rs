//! Experimental BF16 low-rank branches alongside the original quantized base.
//! Base projections stop before activation/gating, then the two paths are combined.
use super::*;

struct Projection {
    prepare: Prepare,
    rank_prepare: Prepare,
    down: Gemm,
    up: Gemm,
    base: Gemm,
    finish: crate::compile::Kernel,
    output: usize,
    result_width: usize,
    mode: usize,
    weights: Vec<Option<(hrx::Buffer, hrx::Buffer)>>,
}

pub(super) struct AdapterRuntime {
    projections: Vec<Projection>,
    input: Option<hrx::Buffer>,
    rank: hrx::Buffer,
    rank_operand: hrx::Buffer,
    base: hrx::Buffer,
    delta: hrx::Buffer,
    classes: usize,
}

fn matrices(
    stream: &mut hrx::Stream,
    adapter: &crate::adapter::Adapter,
    specs: &[&crate::adapter::LinearAdapter],
    rank_width: usize,
) -> Result<(hrx::Buffer, hrx::Buffer)> {
    let first = specs[0];
    let ip = gemm_pitch(first.input, 16);
    let rp = gemm_pitch(rank_width, 16);
    let mut a = vec![0u8; rank_width * ip * 2];
    let mut b = vec![0u8; first.output * rp * 2];
    let mut rank_offset = 0;
    for spec in specs {
        let file = adapter.source(spec);
        for (suffix, rows, cols, scale) in [
            ("lora_A.weight", spec.rank, spec.input, 1.0),
            ("lora_B.weight", spec.output, spec.rank, spec.scale),
        ] {
            let entry = file
                .at(&format!("{}.{suffix}", spec.prefix))
                .map_err(|e| crate::compile::Error::Io(e.to_string()))?;
            let source = file.bytes(entry);
            for row in 0..rows {
                let src_row = if suffix == "lora_B.weight" && spec.gate_up {
                    (row / 32) * 16 + row % 16 + if row % 32 >= 16 { rows / 2 } else { 0 }
                } else {
                    row
                };
                for col in 0..cols {
                    let value = crate::adapter::float_at(entry.dtype, source, src_row * cols + col)
                        .map_err(|e| crate::compile::Error::Io(e.to_string()))?
                        * scale;
                    let value = half::bf16::from_f32(value);
                    if !value.is_finite() {
                        return err("non-finite scaled LoRA weight");
                    }
                    let (dest, offset) = if suffix == "lora_A.weight" {
                        (&mut a, ((rank_offset + row) * ip + col) * 2)
                    } else {
                        (&mut b, (row * rp + rank_offset + col) * 2)
                    };
                    dest[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
            file.done_with(source);
        }
        rank_offset += spec.rank;
    }
    let a_buf = stream.allocate(a.len())?;
    stream.upload(a_buf.binding(), &a)?;
    let b_buf = stream.allocate(b.len())?;
    stream.upload(b_buf.binding(), &b)?;
    Ok((a_buf, b_buf))
}

impl AdapterRuntime {
    pub(super) fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
        stack: &Stack,
        adapter: &crate::adapter::Adapter,
        refiner: bool,
    ) -> Result<Self> {
        let d = &stack.d;
        if d.hidden != HID || d.heads != HEADS || d.ffn != FFN || d.bias || !d.gate_first {
            return err("LoRA adapters require the unbiased H3 DiT/refiner stack");
        }
        let elem = if d.wbits == 8 {
            "i8"
        } else if d.bf16 {
            "bf16"
        } else {
            return err("unsupported LoRA base weight precision");
        };
        let mut projections = Vec::new();
        let mut max_rank_width = 256;
        for (op, input, output, mode) in [
            ("attn.qkv_proj", HID, QKV, 0),
            ("attn.out_proj", INNER, HID, 2),
            ("mlp.fc1", HID, 2 * FFN, 1),
            ("mlp.fc2", FFN, HID, 2),
        ] {
            let group = if refiner {
                "token_refiner.blocks"
            } else {
                "blocks"
            };
            let specs = (0..stack.layers)
                .map(|layer| {
                    let prefix = format!("diffusion_model.{group}.{layer}.{op}");
                    adapter
                        .projections
                        .iter()
                        .filter(|s| s.prefix == prefix)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let rank = specs
                .iter()
                .map(|layer| layer.iter().map(|s| s.rank).sum::<usize>())
                .max()
                .unwrap_or(0)
                .max(1);
            // Prepare works on multiples of 256; padding the rank adds exact zero terms.
            let rank_width = rank.div_ceil(256) * 256;
            max_rank_width = max_rank_width.max(rank_width);
            let ip = gemm_pitch(input, 16);
            let rp = gemm_pitch(rank_width, 16);
            let result_width = if mode == 1 { output / 2 } else { output };
            let mut weights = Vec::new();
            for layer in &specs {
                weights.push(if layer.is_empty() {
                    None
                } else {
                    Some(matrices(stream, adapter, layer, rank_width)?)
                });
            }
            let finish = c.get(
                stream,
                "adapter_finish",
                "h3_adapter_finish",
                &vec![
                    ("h3.adapter_finish.width".into(), result_width.to_string()),
                    ("h3.adapter_finish.mode".into(), mode.to_string()),
                    ("h3.adapter_finish.classes".into(), d.classes.to_string()),
                ],
            )?;
            projections.push(Projection {
                prepare: Prepare::build_with_input(
                    c,
                    stream,
                    if op == "attn.qkv_proj" || op == "mlp.fc1" {
                        "norm"
                    } else {
                        "plain"
                    },
                    "bf16",
                    input,
                    d.eps,
                    d.classes,
                    ip,
                    if op == "attn.out_proj" {
                        ActivationType::F16
                    } else {
                        ActivationType::F32
                    },
                )?,
                rank_prepare: Prepare::build_with_input(
                    c,
                    stream,
                    "plain",
                    "bf16",
                    rank_width,
                    d.eps,
                    1,
                    rp,
                    ActivationType::F32,
                )?,
                down: Gemm::build(
                    c,
                    stream,
                    "f32",
                    "bf16",
                    false,
                    true,
                    input,
                    rank_width,
                    stack.tokens,
                    d.classes,
                    ip,
                    Tile::Plain,
                    0,
                )?,
                up: Gemm::build(
                    c,
                    stream,
                    "f32",
                    "bf16",
                    false,
                    true,
                    rank_width,
                    output,
                    stack.tokens,
                    d.classes,
                    rp,
                    Tile::Plain,
                    0,
                )?,
                base: Gemm::build(
                    c,
                    stream,
                    "f32",
                    elem,
                    false,
                    true,
                    input,
                    output,
                    stack.tokens,
                    d.classes,
                    gemm_pitch(input, d.wbits),
                    Tile::Plain,
                    0,
                )?,
                finish,
                output,
                result_width,
                mode,
                weights,
            });
        }
        let cap = stack.capacity;
        let result = Self {
            projections,
            // The original BF16 input is dead after A*x. The base GEMM writes
            // later in the same dependency chain, so its FP32 scratch can own it.
            input: if env_once("H3_REUSE_SCRATCH") == Some("1") {
                None
            } else {
                Some(stream.allocate(cap * gemm_pitch(FFN, 16) * 2)?)
            },
            rank: stream.allocate(cap * max_rank_width * 4)?,
            rank_operand: stream.allocate(cap * gemm_pitch(max_rank_width, 16) * 2)?,
            base: stream.allocate(cap * 2 * FFN * 4)?,
            delta: stream.allocate(cap * 2 * FFN * 4)?,
            classes: d.classes,
        };
        c.flush(stream)?;
        Ok(result)
    }

    pub(super) fn has(&self, projection: usize, layer: usize) -> bool {
        self.projections[projection].weights[layer].is_some()
    }

    fn input_view(&self) -> View<'_> {
        self.input.as_ref().unwrap_or(&self.base).binding()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit<'g>(
        &'g self,
        sink: &mut Sink<'_, 'g>,
        prof: &mut Profile,
        projection: usize,
        layer: usize,
        tokens: u32,
        original: View<'g>,
        norm: Option<(View<'g>, View<'g>, ClassRows<'g>)>,
        base_input: View<'g>,
        base_weight: View<'g>,
        scales: Option<(View<'g>, View<'g>)>,
        output: View<'g>,
        residual: Option<(View<'g>, ClassRows<'g>)>,
    ) -> Result<()> {
        let p = &self.projections[projection];
        let (a, b) = p.weights[layer].as_ref().expect("active projection");
        let (base_stage, rank_stage, up_stage) = match projection {
            0 => ("gemm qkv f32", "adapter rank qkv", "adapter up qkv"),
            1 => ("gemm out f32", "adapter rank out", "adapter up out"),
            2 => (
                "gemm gate/up f32",
                "adapter rank gate/up",
                "adapter up gate/up",
            ),
            _ => ("gemm down f32", "adapter rank down", "adapter up down"),
        };
        p.prepare.emit(
            sink,
            Some(prof),
            "adapter prepare",
            tokens,
            original,
            norm,
            self.input_view(),
            None,
        )?;
        p.down.emit(
            sink,
            Some(prof),
            rank_stage,
            tokens,
            self.input_view(),
            a.binding(),
            None,
            self.rank.binding(),
            None,
            None,
        )?;
        p.rank_prepare.emit(
            sink,
            Some(prof),
            "adapter rank prepare",
            tokens,
            self.rank.binding(),
            None,
            self.rank_operand.binding(),
            None,
        )?;
        p.up.emit(
            sink,
            Some(prof),
            up_stage,
            tokens,
            self.rank_operand.binding(),
            b.binding(),
            None,
            self.delta.binding(),
            None,
            None,
        )?;
        p.base.emit(
            sink,
            Some(prof),
            base_stage,
            tokens,
            base_input,
            base_weight,
            scales,
            self.base.binding(),
            None,
            None,
        )?;
        let rows = tokens as usize;
        let (gate, cls) = match residual {
            Some((gate, cls)) => (gate, cls.against("adapter finish", rows, self.classes)?),
            None => (self.base.binding(), self.base.binding()),
        };
        emit(
            sink,
            &p.finish,
            Some(prof),
            "adapter finish",
            [(rows * p.result_width).div_ceil(256) as u32, 1, 1],
            [THREADS, 1, 1],
            &[tokens],
            &[self.base.binding(), self.delta.binding(), output, gate, cls],
            &[
                rows * p.output * 4,
                rows * p.output * 4,
                rows * p.result_width * if p.mode == 0 { 2 } else { 4 },
                if p.mode == 2 {
                    self.classes * p.result_width * 4
                } else {
                    0
                },
                if p.mode == 2 { rows * 4 } else { 0 },
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    fn adapter_cases() -> Vec<(&'static str, crate::adapter::Adapter)> {
        let mut cases = Vec::new();
        for (name, preset) in [
            ("Four", crate::adapter::TurboPreset::Four),
            ("Eight", crate::adapter::TurboPreset::Eight),
        ] {
            let path = preset.resolve(true).unwrap();
            // Safety: cached test artifacts remain immutable.
            cases.push((
                name,
                unsafe { crate::adapter::Adapter::open(&path) }.unwrap(),
            ));
        }
        let path = crate::models::Resolver::new()
            .repository("pablodawson", "MiniMax-H3-360-Orbit-LoRA")
            .revision(Some("5ddbc2dbbe95edbbdaf5017c3e934b1d01791697".into()))
            .offline(true)
            .find("minimax_h3_flf2v_lora_v1.safetensors")
            .unwrap();
        // Safety: cached test artifacts remain immutable.
        let orbit = unsafe {
            crate::adapter::Adapter::open_loras(&[crate::adapter::Lora::new(&path, 1.0)])
        }
        .unwrap()
        .unwrap();
        assert_eq!(orbit.projections.len(), 208);
        assert!(orbit
            .projections
            .iter()
            .all(|s| s.rank == 16 && s.scale == 1.0));
        cases.push(("Orbit", orbit));
        // Overlapping, signed branches on just one projection in each stack.
        // The remaining projections must continue through their base kernels.
        let mut mixed = unsafe {
            crate::adapter::Adapter::open_loras(&[
                crate::adapter::Lora::new(&path, 0.5),
                crate::adapter::Lora::new(
                    crate::adapter::TurboPreset::Four.resolve(true).unwrap(),
                    -0.125,
                ),
            ])
        }
        .unwrap()
        .unwrap();
        mixed
            .projections
            .retain(|s| s.prefix.ends_with(".0.mlp.fc1"));
        cases.push(("MixedPartialOrbit", mixed));
        cases
    }

    fn read(stream: &mut hrx::Stream, buffer: &hrx::Buffer, bytes: usize) -> Vec<u8> {
        read_view(stream, buffer.binding(), bytes)
    }
    fn read_view(stream: &mut hrx::Stream, view: View<'_>, bytes: usize) -> Vec<u8> {
        let mut out = vec![0; bytes];
        stream
            .read_blocking(view.slice(0, bytes).unwrap(), &mut out)
            .unwrap();
        out
    }
    fn bfloat(bytes: &[u8], i: usize) -> f32 {
        bf16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f32()
    }
    fn close(actual: &[f32], expected: &[f32], context: &str) {
        let error: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|b| f64::from(*b).powi(2)).sum();
        let relative = (error / norm.max(1e-30)).sqrt();
        eprintln!("{context}: relative L2 {relative:.6}");
        assert!(relative < 0.005, "{context}: relative L2 {relative}");
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore = "requires idle gfx1151, base checkpoint and cached Turbo and Orbit adapters"]
    fn real_adapted_stacks_match_eager_and_recorded_execution() {
        assert!(
            std::env::var_os("H3_ADAPTER_BLOCK_FIXTURE").is_none()
                || env_once("H3_REUSE_SCRATCH") != Some("1"),
            "export reference intermediates with separate scratch; test reused scratch without H3_ADAPTER_BLOCK_FIXTURE"
        );
        let c = Compiler::new(None, std::path::PathBuf::new());
        let mut stream = hrx::Stream::open().unwrap();
        let path = crate::models::Resolver::new()
            .offline(true)
            .find(crate::models::DIT_FL2VA)
            .unwrap();
        // Safety: these cached fixtures remain immutable throughout the test.
        let weights = unsafe { Weights::open(path, crate::plan::dit::plan) }.unwrap();
        let constants = Constants::new(&mut stream).unwrap();
        let rows = 17;
        for (preset, checkpoint) in adapter_cases() {
            for refiner in [false, true] {
                let classes = if refiner { 1 } else { 4 };
                let dims = StackDims {
                    hidden: HID,
                    heads: HEADS,
                    kv_heads: HEADS,
                    head_dim: HEAD_DIM,
                    ffn: FFN,
                    rope_dim: ROPE_DIM,
                    classes,
                    wbits: if refiner { 16 } else { 8 },
                    eps: 1e-5,
                    bias: false,
                    gate_first: true,
                    causal: false,
                    attn_i4: false,
                    attn_qk_bits: if refiner { 16 } else { 8 },
                    bf16: refiner,
                };
                let mut stack = Stack::new(
                    &c,
                    &mut stream,
                    dims,
                    rows,
                    1,
                    &weights,
                    |_| {
                        if refiner {
                            "h3.refiner.0.".into()
                        } else {
                            "blocks.0.".into()
                        }
                    },
                    true,
                    constants.ones.clone(),
                    "adapter-graph-test",
                )
                .unwrap();
                stack
                    .attach_adapter(&c, &mut stream, &checkpoint, refiner)
                    .unwrap();
                let x = stream.allocate(stack.capacity * HID * 4).unwrap();
                stream.fill(x.binding(), 0).unwrap();
                let mut cls = crate::dispatch::Classes::zeroed(&mut stream, rows).unwrap();
                cls.write(
                    &mut stream,
                    &(0..rows).map(|r| (r % classes) as i32).collect::<Vec<_>>(),
                    classes,
                )
                .unwrap();
                let cos = stream.allocate(rows * ROPE_DIM / 2 * 4).unwrap();
                let sin = stream.allocate(rows * ROPE_DIM / 2 * 4).unwrap();
                stream
                    .upload(
                        cos.binding(),
                        bytemuck::cast_slice(&vec![1.0f32; rows * ROPE_DIM / 2]),
                    )
                    .unwrap();
                stream.fill(sin.binding(), 0).unwrap();
                let table = stream.allocate(classes * 2 * HID * 4).unwrap();
                let gate = stream.allocate(classes * HID * 4).unwrap();
                stream
                    .upload(
                        table.binding(),
                        bytemuck::cast_slice(
                            &(0..classes * 2 * HID)
                                .map(|i| (i % 13) as f32 / 100.0)
                                .collect::<Vec<_>>(),
                        ),
                    )
                    .unwrap();
                stream
                    .upload(
                        gate.binding(),
                        bytemuck::cast_slice(
                            &(0..classes * HID)
                                .map(|i| 0.1 + (i % 7) as f32 / 20.0)
                                .collect::<Vec<_>>(),
                        ),
                    )
                    .unwrap();
                let cond = |_| LayerCond {
                    table_msa: table.binding(),
                    gate_msa: gate.binding(),
                    table_mlp: table.binding(),
                    gate_mlp: gate.binding(),
                };
                let mut graph = None;
                for iteration in 0..3 {
                    let input: Vec<f32> = (0..rows * HID)
                        .map(|i| ((i * 17 + iteration * 31) % 127) as f32 / 64.0 - 1.0)
                        .collect();
                    stream
                        .upload(x.binding(), bytemuck::cast_slice(&input))
                        .unwrap();
                    stack
                        .forward(
                            &mut stream,
                            &mut Profile::default(),
                            x.binding(),
                            cls.slice(0, rows),
                            cos.binding(),
                            sin.binding(),
                            &cond,
                            0,
                            None,
                        )
                        .unwrap();
                    let expected = read(&mut stream, &x, rows * HID * 4);
                    stream
                        .upload(x.binding(), bytemuck::cast_slice(&input))
                        .unwrap();
                    // Record explicitly so this test exercises replay even when
                    // the caller has not enabled the optional H3_GRAPH path.
                    if graph.is_none() {
                        let mut recording = stream.graph().unwrap();
                        stack
                            .emit(
                                &mut Sink::Graph {
                                    graph: &mut recording,
                                    after: Default::default(),
                                },
                                &mut Profile::default(),
                                x.binding(),
                                cls.slice(0, rows),
                                cos.binding(),
                                sin.binding(),
                                &cond,
                                0,
                                None,
                                false,
                            )
                            .unwrap();
                        graph = Some(recording.finish().unwrap());
                    }
                    stream.launch(graph.as_mut().unwrap()).unwrap();
                    let actual = read(&mut stream, &x, rows * HID * 4);
                    assert!(
                        actual == expected,
                        "{preset} refiner={refiner} replay={iteration}"
                    );
                    assert!(actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite()));
                    if iteration == 0 && preset != "MixedPartialOrbit" {
                        if let Some(dir) = std::env::var_os("H3_ADAPTER_BLOCK_FIXTURE") {
                            let dir = std::path::PathBuf::from(dir).join(format!(
                                "{preset}-{}",
                                if refiner { "refiner" } else { "dit" }
                            ));
                            std::fs::create_dir_all(&dir).unwrap();
                            std::fs::write(dir.join("input.f32"), bytemuck::cast_slice(&input))
                                .unwrap();
                            std::fs::write(dir.join("output.f32"), &actual).unwrap();
                            for (name, view, bytes) in [
                                ("attention.f16", stack.attn.binding(), rows * INNER * 2),
                                ("hidden.f32", stack.gu_view(), rows * FFN * 4),
                            ] {
                                std::fs::write(dir.join(name), read_view(&mut stream, view, bytes))
                                    .unwrap();
                            }
                            // Fused QKV is overwritten by gate/up when scratch reuse is enabled.
                            if stack.gu.is_some() {
                                std::fs::write(
                                    dir.join("qkv.f16"),
                                    read(&mut stream, &stack.fused, rows * QKV * 2),
                                )
                                .unwrap();
                            }
                            std::fs::write(
                                dir.join("table.f32"),
                                read(&mut stream, &table, classes * 2 * HID * 4),
                            )
                            .unwrap();
                            std::fs::write(
                                dir.join("gate.f32"),
                                read(&mut stream, &gate, classes * HID * 4),
                            )
                            .unwrap();
                            std::fs::write(dir.join("fixture.json"), format!("{{\"rows\":{rows},\"classes\":{classes},\"refiner\":{refiner},\"preset\":\"{preset}\",\"rope\":\"identity\",\"layer\":0}}\n")).unwrap();
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires idle gfx1151, base checkpoint and cached Turbo and Orbit adapters"]
    fn real_low_rank_branches_match_independent_cpu_products() {
        let c = Compiler::new(None, std::path::PathBuf::new());
        let mut stream = hrx::Stream::open().unwrap();
        let path = crate::models::Resolver::new()
            .offline(true)
            .find(crate::models::DIT_FL2VA)
            .unwrap();
        // Safety: these cached fixtures remain immutable throughout the test.
        let weights = unsafe { Weights::open(path, crate::plan::dit::plan) }.unwrap();
        let constants = Constants::new(&mut stream).unwrap();
        let rows = 3;
        let mut cls = crate::dispatch::Classes::zeroed(&mut stream, rows).unwrap();
        cls.write(&mut stream, &vec![0; rows], 1).unwrap();
        for (preset, checkpoint) in adapter_cases() {
            for refiner in [false, true] {
                let dims = StackDims {
                    hidden: HID,
                    heads: HEADS,
                    kv_heads: HEADS,
                    head_dim: HEAD_DIM,
                    ffn: FFN,
                    rope_dim: ROPE_DIM,
                    classes: 1,
                    wbits: if refiner { 16 } else { 8 },
                    eps: 1e-5,
                    bias: false,
                    gate_first: true,
                    causal: false,
                    attn_i4: false,
                    attn_qk_bits: 16,
                    bf16: refiner,
                };
                let mut stack = Stack::new(
                    &c,
                    &mut stream,
                    dims,
                    rows,
                    1,
                    &weights,
                    |_| {
                        if refiner {
                            "h3.refiner.0.".into()
                        } else {
                            "blocks.0.".into()
                        }
                    },
                    true,
                    constants.ones.clone(),
                    "adapter-test",
                )
                .unwrap();
                stack
                    .attach_adapter(&c, &mut stream, &checkpoint, refiner)
                    .unwrap();
                let adapter = stack.adapter.as_ref().unwrap();
                let block = &stack.blocks[0];
                for (op, input_width, op_name) in [
                    (0, HID, "attn.qkv_proj"),
                    (1, INNER, "attn.out_proj"),
                    (2, HID, "mlp.fc1"),
                    (3, FFN, "mlp.fc2"),
                ] {
                    if !adapter.has(op, 0) {
                        continue;
                    }
                    let p = &adapter.projections[op];
                    let original_values: Vec<f32> = (0..rows * input_width)
                        .map(|i| ((i * 17 % 127) as f32 - 63.0) / 64.0)
                        .collect();
                    let original_bytes = if op != 1 {
                        bytemuck::cast_slice(&original_values).to_vec()
                    } else {
                        original_values
                            .iter()
                            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                            .collect()
                    };
                    let original = stream.allocate(original_bytes.len()).unwrap();
                    stream.upload(original.binding(), &original_bytes).unwrap();
                    let norm = (op == 0 || op == 2).then_some((
                        constants.ones.binding(),
                        constants.zeros.binding(),
                        cls.slice(0, rows),
                    ));
                    let prepare = match op {
                        0 | 2 => &stack.prep_norm,
                        1 => stack.prep_attn.as_ref().unwrap(),
                        _ => stack.prep_down.as_ref().unwrap(),
                    };
                    prepare
                        .run(
                            &mut stream,
                            None,
                            "base prepare",
                            rows as u32,
                            original.binding(),
                            norm,
                            stack.a_q.binding(),
                            Some(stack.a_s.binding()),
                        )
                        .unwrap();
                    let (w, scale) = match op {
                        0 => (&block.qkv_q, &block.qkv_s),
                        1 => (&block.out_q, &block.out_s),
                        2 => (&block.gu_q, &block.gu_s),
                        _ => (&block.down_q, &block.down_s),
                    };
                    let output = stream.allocate(rows * p.result_width * 4).unwrap();
                    stream.fill(output.binding(), 0).unwrap();
                    let residual =
                        (p.mode == 2).then_some((constants.ones.binding(), cls.slice(0, rows)));
                    // Snapshot before the full projection reuses this input storage.
                    p.prepare
                        .run(
                            &mut stream,
                            None,
                            "adapter prepare fixture",
                            rows as u32,
                            original.binding(),
                            norm,
                            adapter.input_view(),
                            None,
                        )
                        .unwrap();
                    let ip = gemm_pitch(input_width, 16);
                    let input = read_view(&mut stream, adapter.input_view(), rows * ip * 2);
                    adapter
                        .emit(
                            &mut Sink::Stream(&mut stream),
                            &mut Profile::default(),
                            op,
                            0,
                            rows as u32,
                            original.binding(),
                            norm,
                            stack.a_q.binding(),
                            w.binding(),
                            scale.as_ref().map(|s| (s.binding(), stack.a_s.binding())),
                            output.binding(),
                            residual,
                        )
                        .unwrap();
                    let group = if refiner {
                        "token_refiner.blocks"
                    } else {
                        "blocks"
                    };
                    let prefix = format!("diffusion_model.{group}.0.{op_name}");
                    let specs: Vec<_> = checkpoint
                        .projections
                        .iter()
                        .filter(|s| s.prefix == prefix)
                        .collect();
                    let rank: usize = specs.iter().map(|s| s.rank).sum();
                    let rank_width = rank.div_ceil(256) * 256;
                    let rp = gemm_pitch(rank_width, 16);
                    let ranks = read(&mut stream, &adapter.rank_operand, rows * rp * 2);
                    let mut expected_rank = vec![0.0; rows * rank];
                    let mut actual_rank = expected_rank.clone();
                    let mut offset = 0;
                    for spec in &specs {
                        let file = checkpoint.source(spec);
                        let entry = file.at(&format!("{prefix}.lora_A.weight")).unwrap();
                        let a = file.bytes(entry);
                        for row in 0..rows {
                            for r in 0..spec.rank {
                                let mut sum = 0.0f32;
                                for k in 0..input_width {
                                    sum += bfloat(&input, row * ip + k)
                                        * bfloat(a, r * input_width + k);
                                }
                                expected_rank[row * rank + offset + r] =
                                    bf16::from_f32(sum).to_f32();
                                actual_rank[row * rank + offset + r] =
                                    bfloat(&ranks, row * rp + offset + r);
                            }
                        }
                        offset += spec.rank;
                    }
                    close(
                        &actual_rank,
                        &expected_rank,
                        &format!("{preset} {prefix} A"),
                    );
                    let delta = read(&mut stream, &adapter.delta, rows * p.output * 4);
                    let mut expected = vec![0.0; rows * p.output];
                    let mut actual = expected.clone();
                    for row in 0..rows {
                        for column in 0..p.output {
                            // Original B keeps whole gate/up halves, unlike the GPU's packed rows.
                            let original_column = if specs[0].gate_up {
                                (column / 32) * 16
                                    + column % 16
                                    + if column % 32 >= 16 { p.output / 2 } else { 0 }
                            } else {
                                column
                            };
                            let mut sum = 0.0f32;
                            let mut offset = 0;
                            for spec in &specs {
                                let file = checkpoint.source(spec);
                                let entry = file.at(&format!("{prefix}.lora_B.weight")).unwrap();
                                let b = file.bytes(entry);
                                for r in 0..spec.rank {
                                    sum += bfloat(&ranks, row * rp + offset + r)
                                        * bf16::from_f32(
                                            bfloat(b, original_column * spec.rank + r) * spec.scale,
                                        )
                                        .to_f32();
                                }
                                offset += spec.rank;
                            }
                            expected[row * p.output + column] = sum;
                            actual[row * p.output + column] = f32::from_le_bytes(
                                delta[(row * p.output + column) * 4
                                    ..(row * p.output + column) * 4 + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                        }
                    }
                    close(&actual, &expected, &format!("{preset} {prefix} B"));
                }
                stream.synchronize().unwrap();
            }
        }
    }
}
