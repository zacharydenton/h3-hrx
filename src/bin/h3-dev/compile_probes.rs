//! Compile the bounded operand/attention diagnostics without opening the GPU.
use std::path::Path;
pub fn run(args: Vec<String>) {
    assert!(args.is_empty(), "compile-probes takes no arguments");
    let compiler = hrx::loom::Compiler::resolve(None).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut requests = Vec::new();
    for heads in [2usize, 56] {
        for offset in [0, heads * 128] {
            requests.push((
                "kernels",
                "prepare_qk_rope_i8hm",
                vec![
                    ("row_stride", (heads * 384).to_string()),
                    ("heads", heads.to_string()),
                    ("head_offset", offset.to_string()),
                    ("token_capacity", "256".into()),
                    ("extra_scale", "1".into()),
                    ("eps", "1e-5".into()),
                ],
                true,
            ));
        }
        requests.push((
            "kernels",
            "transpose_qkv_v_f16",
            vec![
                ("width", (heads * 128).to_string()),
                ("row_capacity", "256".into()),
            ],
            true,
        ));
    }
    requests.push((
        "experiments",
        "prepare_qk_i8hm_shuffle",
        vec![
            ("row_stride", "2176".into()),
            ("heads", "17".into()),
            ("head_offset", "0".into()),
            ("token_capacity", "512".into()),
            ("extra_scale", "1".into()),
        ],
        false,
    ));
    requests.push((
        "experiments",
        "attention_i8qk45phase_hm_mha8_lds_f16_wmma",
        vec![
            ("q_stride", "7168".into()),
            ("kv_stride", "7168".into()),
            ("out_stride", "7168".into()),
            ("tokens", "38048".into()),
            ("token_capacity", "38144".into()),
            ("scale", "0.08838834764831845".into()),
        ],
        false,
    ));
    let mut failed = false;
    for (dir, stem, cfg, required) in requests {
        let source = std::fs::read_to_string(root.join(dir).join(format!("{stem}.loom"))).unwrap();
        let mut req = hrx::loom::Specialization::new(format!("h3_{stem}"));
        req.set_report(hrx::loom::ReportMode::Summary);
        req.replace_config(
            cfg.into_iter()
                .map(|(k, v)| (format!("h3.{stem}.{k}"), v))
                .collect(),
        );
        match compiler.module(&source).compile(&req) {
            Ok(a) => println!("{{\"stem\":\"{stem}\",\"compiled\":true,\"sha256\":\"{}\",\"compiler\":\"{}\",\"report\":{}}}",
                hrx::bundle::digest(a.bytes()), a.compiler_identity(), a.report().map_or("null".into(), |report| report.json().to_string())),
            Err(e) => {
                eprintln!("{stem}: {e}");
                println!("{{\"stem\":\"{stem}\",\"compiled\":false,\"required\":{required}}}");
                failed |= required;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
