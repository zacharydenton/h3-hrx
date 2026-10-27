//! Compiler qualification that does not open a GPU.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires the provisioned Loom compiler; no GPU required"
)]
fn attention_exports_compile_for_long_clips() -> hrx::Result<()> {
    let compiler = hrx::loom::Compiler::resolve(None)?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if !path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("attention_")
        {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        // Qualify every attention export, including the four/eight-wave and
        // int4 skip variants, using the sequence length of the reported failure.
        for line in source
            .lines()
            .filter(|line| line.starts_with("config.decl @h3.") && line.contains(".tokens "))
        {
            let namespace = line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .trim_start_matches('@')
                .trim_end_matches(".tokens");
            let stem = namespace.trim_start_matches("h3.");
            let lengths: &[usize] = if stem == "attention_i8qkhm_mha8_k64_lds_f16_wmma" {
                &[119_585, 478_340, 1_048_544]
            } else {
                &[119_585]
            };
            for &tokens in lengths {
                let mut spec = hrx::loom::Specialization::new(format!("h3_{stem}"));
                let causal = stem.contains("gqa");
                for (key, value) in [
                    ("q_stride", if causal { 8192 } else { 7168 }),
                    ("kv_stride", if causal { 1024 } else { 7168 }),
                    ("out_stride", if causal { 8192 } else { 7296 }),
                    ("tokens", tokens),
                    ("token_capacity", (tokens + 16).div_ceil(256) * 256),
                ] {
                    spec.set_config(format!("{namespace}.{key}"), value.to_string());
                }
                spec.set_config(format!("{namespace}.scale"), "0.08838834764831845");
                if source.contains(&format!("@{namespace}.skip_tau")) {
                    spec.set_config(format!("{namespace}.skip_tau"), "0.001");
                }
                compiler.module(&source).compile(&spec)?;
            }
        }
    }
    Ok(())
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires the provisioned Loom compiler; no GPU required"
)]
fn reflected_convolution_proves_nonnegative_gather_rows() -> hrx::Result<()> {
    let compiler = hrx::loom::Compiler::resolve(None)?;
    for (frames, height, width) in [(3usize, 6usize, 6usize), (1, 256, 256)] {
        for taps in [1usize, 3] {
            for stride in [1usize, 2] {
                for residual in [false, true] {
                    let stem = if residual {
                        "conv3d_f16_wmma_add"
                    } else {
                        "conv3d_f16_wmma"
                    };
                    let mut spec = hrx::loom::Specialization::new(format!("h3_{stem}"));
                    for (key, value) in [
                        ("frames", frames),
                        ("height", height),
                        ("width", width),
                        ("stride", stride),
                        ("tstride", if taps == 1 { 1 } else { 2 }),
                        ("taps_t", taps),
                        ("cin_pad", 8),
                        ("cin_stride", 16),
                        ("rows_bound", (frames * height * width).div_ceil(64) * 64),
                        ("k_size", (9 * taps * 8).div_ceil(32) * 32),
                        ("n_size", 64),
                    ] {
                        spec.set_config(format!("h3.{stem}.{key}"), value.to_string());
                    }
                    compiler
                        .module(include_str!("../kernels/conv3d_f16_family.loom"))
                        .compile(&spec)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires the provisioned Loom compiler; no GPU required"
)]
fn head_major_operand_preparation_proves_scale_rows() -> hrx::Result<()> {
    let compiler = hrx::loom::Compiler::resolve(None)?;
    let (heads, inner) = (56usize, 7168usize);
    let separate = include_str!("../kernels/prepare_qk_i8hm.loom");
    let fused = include_str!("../kernels/prepare_qk_rope_i8hm.loom");
    // 89,088 is the capacity of the 248-frame 768x1344 clip that failed to compile.
    for capacity in [89_088usize, 1_048_544] {
        // The fused kernel reads Q, then K one projection width further along.
        for (stem, source, row_stride, head_offset) in [
            ("prepare_qk_i8hm", separate, inner, 0),
            ("prepare_qk_rope_i8hm", fused, 3 * inner, 0),
            ("prepare_qk_rope_i8hm", fused, 3 * inner, inner),
        ] {
            let mut spec = hrx::loom::Specialization::new(format!("h3_{stem}"));
            for (key, value) in [
                ("row_stride", row_stride),
                ("head_offset", head_offset),
                ("heads", heads),
                ("token_capacity", capacity),
            ] {
                spec.set_config(format!("h3.{stem}.{key}"), value.to_string());
            }
            spec.set_config(format!("h3.{stem}.extra_scale"), "1");
            if stem == "prepare_qk_rope_i8hm" {
                spec.set_config(format!("h3.{stem}.eps"), "0.000001");
            }
            compiler.module(source).compile(&spec)?;
        }
    }
    Ok(())
}
