//! What a launch costs through H3's prepared dispatch path, eagerly and as a replayed graph.
//!
//! Both arms run identical launches over identical allocations, so the difference between them is
//! the runtime's and nothing else. The two sizes are the point: a recording's per-node cost is
//! roughly fixed, so a graph looks best against a kernel that does nothing, and the question this
//! answers is whether it still wins once a node does the work a real stack's node does.
use h3::compile::Compiler;
use h3::dispatch::{Prepare, Sink};

type Fallible = Result<(), Box<dyn std::error::Error>>;

/// Median of nine batches after three warmups: nanoseconds per launch, recorded and completed.
fn measure(
    stream: &mut hrx::Stream,
    launches: usize,
    mut batch: impl FnMut(&mut hrx::Stream) -> Fallible,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let (mut host, mut complete) = (Vec::new(), Vec::new());
    for round in 0..12 {
        stream.synchronize()?;
        let began = std::time::Instant::now();
        batch(stream)?;
        let recorded = began.elapsed();
        stream.synchronize()?;
        if round >= 3 {
            host.push(recorded.as_secs_f64() * 1e9 / launches as f64);
            complete.push(began.elapsed().as_secs_f64() * 1e9 / launches as f64);
        }
    }
    host.sort_by(f64::total_cmp);
    complete.sort_by(f64::total_cmp);
    Ok((host[4], complete[4]))
}

fn main() -> Fallible {
    let mut stream = hrx::Stream::open()?;
    let compiler = Compiler::new(None, "");
    const LAUNCHES: usize = 256;

    for (width, tokens) in [(256usize, 1u32), (4096, 1024)] {
        let prepare = Prepare::build(&compiler, &mut stream, "plain", "f16", width, 0.0, 1, width)?;
        compiler.flush(&mut stream)?;
        let bytes = width * tokens as usize * 2;
        let input = stream.allocate(bytes)?;
        let output = stream.allocate(bytes)?;
        stream.fill(input.binding(), 0)?;
        stream.fill(output.binding(), 0x7f)?;

        let (host, eager) = measure(&mut stream, LAUNCHES, |stream| {
            for _ in 0..LAUNCHES {
                prepare.run(
                    stream,
                    None,
                    "prepare",
                    tokens,
                    input.binding(),
                    None,
                    output.binding(),
                    None,
                )?;
            }
            Ok(())
        })?;

        // The same launches as a chain. Every node writes `output`, so every edge is real: this is
        // the shape a recording of any of this model's stacks has.
        let mut graph = stream.graph()?;
        {
            let mut sink = Sink::Graph {
                graph: &mut graph,
                after: None,
            };
            for _ in 0..LAUNCHES {
                prepare.emit(
                    &mut sink,
                    None,
                    "prepare",
                    tokens,
                    input.binding(),
                    None,
                    output.binding(),
                    None,
                )?;
            }
        }
        let mut replay = graph.finish()?;
        let (_, replayed) = measure(&mut stream, LAUNCHES, |stream| {
            stream.launch(&mut replay)?;
            Ok(())
        })?;

        println!(
            "prepare {width}x{tokens}: eager {host:.0} ns host, {eager:.0} ns completed; \
             graph {replayed:.0} ns completed ({:.2}x)",
            replayed / eager
        );
        let mut sample = vec![0u8; 512];
        stream.read_blocking(output.binding().slice(0, 512)?, &mut sample)?;
        assert_eq!(sample, vec![0u8; 512]);
    }

    // The audio VAE's own residual convolution, at the shape its first upsample level runs: this is
    // the path a profile makes look launch-bound, so it is the one worth asking directly.
    for (chan, len) in [(1024usize, 5usize), (8, 165_600)] {
        let (kk, dil) = (7usize, 3usize);
        let ns = "h3.conv1d4_f32.";
        let cfg: h3::compile::Cfg = vec![
            (format!("{ns}cin"), chan.to_string()),
            (format!("{ns}cout"), chan.to_string()),
            (format!("{ns}ksize"), kk.to_string()),
            (format!("{ns}dilation"), dil.to_string()),
            (format!("{ns}pad"), ((kk * dil - dil) / 2).to_string()),
            (format!("{ns}accumulate"), "0".into()),
            (
                format!("{ns}len_bound"),
                len.div_ceil(256).max(1).saturating_mul(256).to_string(),
            ),
        ];
        let kernel = compiler.get(&mut stream, "conv1d4_f32", "h3_conv1d4_f32", &cfg)?;
        compiler.flush(&mut stream)?;
        let plane = stream.allocate(chan * len.div_ceil(256).max(1) * 256 * 4)?;
        let out = stream.allocate(chan * len.div_ceil(256).max(1) * 256 * 4)?;
        let weight = stream.allocate(chan * chan * kk * 4)?;
        let bias = stream.allocate(chan * 4)?;
        for b in [&plane, &out, &weight, &bias] {
            stream.fill(b.binding(), 0)?;
        }
        let grid = [len.div_ceil(256) as u32, chan as u32, 1];
        let need = [plane.bytes(), weight.bytes(), bias.bytes(), out.bytes()];
        let views = [
            plane.binding(),
            weight.binding(),
            bias.binding(),
            out.binding(),
        ];

        let (host, eager) = measure(&mut stream, LAUNCHES, |stream| {
            for _ in 0..LAUNCHES {
                h3::dispatch::emit(
                    &mut Sink::Stream(stream),
                    &kernel,
                    None,
                    "res conv",
                    grid,
                    [64, 1, 1],
                    &[len as u32],
                    &views,
                    &need,
                )?;
            }
            Ok(())
        })?;

        let mut graph = stream.graph()?;
        {
            let mut sink = Sink::Graph {
                graph: &mut graph,
                after: None,
            };
            for _ in 0..LAUNCHES {
                h3::dispatch::emit(
                    &mut sink,
                    &kernel,
                    None,
                    "res conv",
                    grid,
                    [64, 1, 1],
                    &[len as u32],
                    &views,
                    &need,
                )?;
            }
        }
        let mut replay = graph.finish()?;
        let (_, replayed) = measure(&mut stream, LAUNCHES, |stream| {
            stream.launch(&mut replay)?;
            Ok(())
        })?;
        println!(
            "conv1d4 {chan}x{len}: eager {host:.0} ns host, {eager:.0} ns completed;              graph {replayed:.0} ns completed ({:.2}x)",
            replayed / eager
        );
    }
    Ok(())
}
