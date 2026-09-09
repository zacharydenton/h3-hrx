//! Completed and host recording costs through H3's prepared dispatch path.
use h3::{compile::Compiler, dispatch::Prepare};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = hrx::Stream::open()?;
    let compiler = Compiler::new(None, "", "");
    let prepare = Prepare::build(&compiler, &mut stream, "plain", "f16", 256, 0.0, 1, 256)?;
    compiler.flush(&mut stream)?;
    let input = stream.allocate(512)?;
    let output = stream.allocate(512)?;
    stream.fill(input.binding(), 0)?;
    stream.fill(output.binding(), 0x7f)?;
    let mut host = Vec::new();
    let mut complete = Vec::new();
    for round in 0..12 {
        stream.synchronize()?;
        let began = std::time::Instant::now();
        for _ in 0..2048 {
            prepare.run(
                &mut stream,
                None,
                "prepare",
                1,
                input.binding(),
                None,
                output.binding(),
                None,
            )?;
        }
        let recorded = began.elapsed();
        stream.synchronize()?;
        if round >= 3 {
            host.push(recorded.as_secs_f64() * 1e9 / 2048.0);
            complete.push(began.elapsed().as_secs_f64() * 1e9 / 2048.0);
        }
    }
    host.sort_by(f64::total_cmp);
    complete.sort_by(f64::total_cmp);
    let mut bytes = [0; 512];
    stream.read(output.binding(), &mut bytes)?;
    assert_eq!(bytes, [0; 512]);
    println!(
        "H3 Prepare::run: {:.0} ns host/dispatch, {:.0} ns completed/dispatch",
        host[4], complete[4]
    );
    Ok(())
}
