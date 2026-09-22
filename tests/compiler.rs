//! Compiler qualification that does not open a GPU.
#[test]
#[ignore = "requires the provisioned Loom compiler; no GPU required"]
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
