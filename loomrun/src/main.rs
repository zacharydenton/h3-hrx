//! loomrun — launch one Loom-compiled kernel on the GPU and dump its buffers.
//!
//!   loomrun --hsaco k.hsaco --kernel name --grid 201 --block 256 \
//!           --i32 201 --in x.bin --in gamma.bin --in beta.bin --out y.bin:308736
//!
//! Arguments appear in the kernel's own declaration order: `--i32`/`--f32` are by-value arguments and
//! `--in`/`--inout`/`--out` are buffers. `--in` is uploaded, `--out path:bytes` is allocated and written
//! back after the launch, and `--inout` is both.
//!
//! `--rotate-input INDEX:COUNT` uses COUNT separately allocated copies of one input (zero-based argument
//! index), cycling between timed launches. This measures streaming weight traffic instead of repeatedly
//! hitting one matrix.
//!
//! It runs on libhrx rather than HIP, so a kernel test needs no ROCm headers or hipcc. Timing is wall
//! clock around a synchronised run of `--repeat` launches; the JSON line on stdout is what
//! `tools/kernel_test.py` reads.
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

/// Usage errors exit 64, as the C implementation did; runtime failures exit 1.
const EXIT_USAGE: u8 = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    In,
    Out,
    InOut,
}

enum Arg {
    Scalar(u32),
    Buffer {
        path: PathBuf,
        bytes: usize,
        direction: Direction,
    },
}

struct Options {
    hsaco: PathBuf,
    kernel: String,
    grid: [u32; 3],
    block: [u32; 3],
    repeat: u32,
    rotate: Option<(usize, usize)>,
    verbose: bool,
    args: Vec<Arg>,
}

fn parse_triple(text: &str, what: &str) -> Result<[u32; 3], String> {
    let mut out = [1u32, 1, 1];
    for (i, part) in text.split(',').enumerate() {
        if i >= 3 {
            return Err(format!("{what} takes at most three components"));
        }
        out[i] = part
            .trim()
            .parse()
            .map_err(|_| format!("{what} component {:?} is not a number", part))?;
    }
    Ok(out)
}

fn parse_args(argv: &[String]) -> Result<Options, String> {
    let (mut hsaco, mut kernel) = (None, None);
    let (mut grid, mut block) = ([1u32, 1, 1], [1u32, 1, 1]);
    let (mut repeat, mut verbose, mut rotate) = (1u32, false, None);
    let mut args: Vec<Arg> = Vec::new();
    let mut i = 0;
    let next = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{name} needs a value"))
    };
    while i < argv.len() {
        let a = argv[i].as_str();
        match a {
            "--hsaco" => hsaco = Some(PathBuf::from(next(&mut i, a)?)),
            "--kernel" => kernel = Some(next(&mut i, a)?),
            "--grid" => grid = parse_triple(&next(&mut i, a)?, "--grid")?,
            "--block" => block = parse_triple(&next(&mut i, a)?, "--block")?,
            "--repeat" => {
                repeat = next(&mut i, a)?
                    .parse()
                    .map_err(|_| "--repeat must be a number".to_string())?
            }
            "--verbose" => verbose = true,
            "--rotate-input" => {
                let spec = next(&mut i, a)?;
                let (index, count) = spec
                    .split_once(':')
                    .ok_or_else(|| "--rotate-input wants INDEX:COUNT (COUNT 2..64)".to_string())?;
                let index: usize = index
                    .parse()
                    .map_err(|_| "--rotate-input wants INDEX:COUNT (COUNT 2..64)".to_string())?;
                let count: usize = count
                    .parse()
                    .map_err(|_| "--rotate-input wants INDEX:COUNT (COUNT 2..64)".to_string())?;
                if !(2..=64).contains(&count) {
                    return Err("--rotate-input wants INDEX:COUNT (COUNT 2..64)".into());
                }
                rotate = Some((index, count));
            }
            "--i32" => {
                let v: i32 = next(&mut i, a)?
                    .parse()
                    .map_err(|_| "--i32 wants an integer".to_string())?;
                args.push(Arg::Scalar(v as u32));
            }
            "--f32" => {
                let v: f32 = next(&mut i, a)?
                    .parse()
                    .map_err(|_| "--f32 wants a number".to_string())?;
                args.push(Arg::Scalar(v.to_bits()));
            }
            "--in" | "--inout" => {
                let direction = if a == "--in" {
                    Direction::In
                } else {
                    Direction::InOut
                };
                args.push(Arg::Buffer {
                    path: PathBuf::from(next(&mut i, a)?),
                    bytes: 0,
                    direction,
                });
            }
            "--out" => {
                let spec = next(&mut i, a)?;
                let (path, bytes) = spec
                    .rsplit_once(':')
                    .ok_or_else(|| "--out wants path:bytes".to_string())?;
                let bytes: usize = bytes
                    .parse()
                    .map_err(|_| "--out wants path:bytes".to_string())?;
                args.push(Arg::Buffer {
                    path: PathBuf::from(path),
                    bytes,
                    direction: Direction::Out,
                });
            }
            _ => return Err(format!("unknown option {a}")),
        }
        i += 1;
    }
    let hsaco = hsaco.ok_or_else(|| "need --hsaco and --kernel".to_string())?;
    let kernel = kernel.ok_or_else(|| "need --hsaco and --kernel".to_string())?;
    if repeat < 1 {
        return Err("--repeat must be positive".into());
    }
    // The rotated argument must name an uploaded input: an output has nothing to copy.
    if let Some((index, _)) = rotate {
        let ok = matches!(
            args.get(index),
            Some(Arg::Buffer {
                direction: Direction::In,
                ..
            })
        );
        if !ok {
            return Err("--rotate-input must name an --in buffer argument".into());
        }
    }
    Ok(Options {
        hsaco,
        kernel,
        grid,
        block,
        repeat,
        rotate,
        verbose,
        args,
    })
}

/// Which rotated copy launch `r` reads, counting the warm-up as launch zero so the timed launches
/// continue the cycle rather than restarting it. Copy 0 is the original upload.
fn rotation_slot(r: u32, warmup: bool, count: usize) -> usize {
    (r as usize + usize::from(warmup)) % count
}

/// Whether a warm-up launch precedes the timed ones. A single requested launch gets none: an in-place
/// kernel must run exactly once or the correctness check sees its transform applied twice.
fn warms_up(repeat: u32) -> bool {
    repeat > 1
}

fn run(opt: Options) -> Result<(), String> {
    let gpu = hrx::Gpu::open().map_err(|e| e.to_string())?;
    let kernel = gpu
        .load(&opt.hsaco, &opt.kernel)
        .map_err(|e| e.to_string())?;

    // Upload in declaration order, keeping scalars and buffers in their own sequences: the export
    // reports the constant block and the binding count separately.
    let mut scalars: Vec<u32> = Vec::new();
    let mut buffers: Vec<hrx::Buffer> = Vec::new();
    let mut buffer_of_arg: Vec<Option<usize>> = Vec::new();
    for arg in &opt.args {
        match arg {
            Arg::Scalar(v) => {
                scalars.push(*v);
                buffer_of_arg.push(None);
            }
            Arg::Buffer {
                path,
                bytes,
                direction,
            } => {
                let host = if *direction == Direction::Out {
                    Vec::new()
                } else {
                    std::fs::read(path)
                        .map_err(|e| format!("cannot open {}: {e}", path.display()))?
                };
                let size = if *direction == Direction::Out {
                    *bytes
                } else {
                    host.len()
                };
                let buffer = gpu.alloc(size).map_err(|e| e.to_string())?;
                gpu.memset(&buffer, 0, buffer.bytes())
                    .map_err(|e| e.to_string())?;
                if !host.is_empty() {
                    gpu.h2d(&buffer, &host).map_err(|e| e.to_string())?;
                }
                buffer_of_arg.push(Some(buffers.len()));
                buffers.push(buffer);
            }
        }
    }

    // Rotation copies the chosen input so consecutive launches read different addresses.
    let mut rotated: Vec<hrx::Buffer> = Vec::new();
    if let Some((index, count)) = opt.rotate {
        let slot = buffer_of_arg[index].expect("validated as a buffer argument");
        let source = &buffers[slot];
        for _ in 1..count {
            let copy = gpu.alloc(source.bytes()).map_err(|e| e.to_string())?;
            gpu.d2d(&copy, source, source.bytes())
                .map_err(|e| e.to_string())?;
            rotated.push(copy);
        }
        gpu.sync().map_err(|e| e.to_string())?;
    }

    let mut bindings: Vec<hrx::View<'_>> = buffers.iter().map(|b| b.binding()).collect();
    if opt.verbose {
        let info = kernel.info();
        eprintln!(
            "constants={} bindings={} grid={:?} block={:?}",
            info.constant_byte_length, info.binding_count, opt.grid, opt.block
        );
    }

    let warmup = warms_up(opt.repeat);
    if warmup {
        gpu.dispatch(&kernel, opt.grid, opt.block, &scalars, &bindings)
            .map_err(|e| e.to_string())?;
        gpu.sync().map_err(|e| e.to_string())?;
    }

    let rotate_slot = opt
        .rotate
        .map(|(index, _)| buffer_of_arg[index].expect("validated"));
    let started = Instant::now();
    for r in 0..opt.repeat {
        if let (Some(slot), Some((_, count))) = (rotate_slot, opt.rotate) {
            let which = rotation_slot(r, warmup, count);
            // Copy 0 is the original upload; the extra allocations follow it.
            bindings[slot] = if which == 0 {
                buffers[slot].binding()
            } else {
                rotated[which - 1].binding()
            };
        }
        gpu.dispatch(&kernel, opt.grid, opt.block, &scalars, &bindings)
            .map_err(|e| e.to_string())?;
    }
    gpu.sync().map_err(|e| e.to_string())?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;

    println!(
        "{{\"launches\": {}, \"total_ms\": {:.6}, \"per_launch_us\": {:.3}, \"input_copies\": {}}}",
        opt.repeat,
        elapsed_ms,
        1000.0 * elapsed_ms / f64::from(opt.repeat),
        opt.rotate.map_or(1, |(_, count)| count)
    );

    for (arg, slot) in opt.args.iter().zip(&buffer_of_arg) {
        if let (
            Arg::Buffer {
                path, direction, ..
            },
            Some(slot),
        ) = (arg, slot)
        {
            if *direction == Direction::In {
                continue;
            }
            let buffer = &buffers[*slot];
            let mut host = vec![0u8; buffer.bytes()];
            gpu.d2h(buffer, &mut host).map_err(|e| e.to_string())?;
            std::fs::write(path, &host)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opt = match parse_args(&argv) {
        Ok(opt) => opt,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    match run(opt) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("loomrun: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rejects_bad_rotate_specs() {
        // Every spelling the C implementation refused, refused for the same reason.
        for spec in ["-1:3", "2:1", "2:65", "3:3", "0:3", "2:3junk"] {
            let argv = args(&[
                "--hsaco",
                "k.hsaco",
                "--kernel",
                "k",
                "--i32",
                "1",
                "--in",
                "a.bin",
                "--out",
                "o.bin:16",
                "--rotate-input",
                spec,
            ]);
            assert!(parse_args(&argv).is_err(), "{spec} should be rejected");
        }
    }

    #[test]
    fn accepts_a_rotate_spec_naming_an_input() {
        let argv = args(&[
            "--hsaco",
            "k.hsaco",
            "--kernel",
            "k",
            "--i32",
            "1",
            "--in",
            "a.bin",
            "--out",
            "o.bin:16",
            "--rotate-input",
            "1:3",
        ]);
        let opt = parse_args(&argv).unwrap();
        assert_eq!(opt.rotate, Some((1, 3)));
    }

    #[test]
    fn requires_the_kernel_and_module() {
        assert!(parse_args(&args(&["--kernel", "k"])).is_err());
        assert!(parse_args(&args(&["--hsaco", "k.hsaco"])).is_err());
        assert!(parse_args(&args(&["--hsaco", "k.hsaco", "--kernel", "k"])).is_ok());
    }

    #[test]
    fn rejects_a_zero_repeat_and_unknown_options() {
        assert!(parse_args(&args(&["--hsaco", "k", "--kernel", "k", "--repeat", "0"])).is_err());
        assert!(parse_args(&args(&["--hsaco", "k", "--kernel", "k", "--nope"])).is_err());
        assert!(parse_args(&args(&["--hsaco", "k", "--kernel", "k", "--i32"])).is_err());
    }

    #[test]
    fn parses_grids_blocks_and_out_specs() {
        let argv = args(&[
            "--hsaco",
            "k",
            "--kernel",
            "k",
            "--grid",
            "3,4,5",
            "--block",
            "256",
            "--out",
            "/tmp/a:b/out.bin:4096",
        ]);
        let opt = parse_args(&argv).unwrap();
        assert_eq!((opt.grid, opt.block), ([3, 4, 5], [256, 1, 1]));
        // the path is split at the LAST colon, so a colon in a directory name survives
        match &opt.args[0] {
            Arg::Buffer {
                path,
                bytes,
                direction,
            } => {
                assert_eq!(path.to_str().unwrap(), "/tmp/a:b/out.bin");
                assert_eq!(*bytes, 4096);
                assert!(*direction == Direction::Out);
            }
            _ => panic!("expected a buffer argument"),
        }
    }

    #[test]
    fn rotation_cycles_through_the_copies_across_the_warmup() {
        // The C++ launcher's fake-HIP test asserted this shape: seven timed launches after one
        // warm-up, three copies, and launch i reading copy i % 3.
        let (repeat, count) = (7u32, 3usize);
        let warmup = warms_up(repeat);
        assert_eq!(repeat + u32::from(warmup), 8);
        let seen: Vec<usize> =
            std::iter::once(rotation_slot(0, false, count)) // the warm-up itself
                .chain((0..repeat).map(|r| rotation_slot(r, warmup, count)))
                .collect();
        assert_eq!(seen.len(), 8);
        for (i, slot) in seen.iter().enumerate() {
            assert_eq!(*slot, i % count, "launch {i}");
        }
    }

    #[test]
    fn a_single_launch_has_no_hidden_warmup() {
        assert!(!warms_up(1));
        assert!(warms_up(2));
        // and it reads the original upload, not a copy
        assert_eq!(rotation_slot(0, false, 3), 0);
    }

    #[test]
    fn scalars_and_buffers_keep_declaration_order() {
        let argv = args(&[
            "--hsaco", "k", "--kernel", "k", "--i32", "7", "--in", "a", "--f32", "0.5", "--in", "b",
        ]);
        let opt = parse_args(&argv).unwrap();
        let scalars: Vec<u32> = opt
            .args
            .iter()
            .filter_map(|a| match a {
                Arg::Scalar(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(scalars, vec![7u32, 0.5f32.to_bits()]);
        assert_eq!(
            opt.args
                .iter()
                .filter(|a| matches!(a, Arg::Buffer { .. }))
                .count(),
            2
        );
    }
}
