//! Resolve and validate a pinned Turbo adapter without opening a GPU context.
pub fn run(args: Vec<String>) {
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut offline = false;
        let mut preset = None;
        for arg in args {
            match arg.as_str() {
                "--offline" => offline = true,
                "turbo-768p-4" if preset.is_none() => {
                    preset = Some(h3_hrx::adapter::TurboPreset::Four)
                }
                "turbo-768p-8" if preset.is_none() => {
                    preset = Some(h3_hrx::adapter::TurboPreset::Eight)
                }
                _ => {
                    return Err(
                        "usage: inspect-adapter [--offline] turbo-768p-4|turbo-768p-8".into(),
                    )
                }
            }
        }
        let preset = preset.ok_or("select turbo-768p-4 or turbo-768p-8")?;
        let path = preset.resolve(offline)?;
        // Safety: this diagnostic never modifies the Hub checkpoint.
        let adapter = unsafe { h3_hrx::adapter::Adapter::open(&path) }?;
        println!("path: {}", path.display());
        println!(
            "validated {} projections, including both token-refiner blocks",
            adapter.projections.len()
        );
        let (video, audio) = preset.schedules();
        println!(
            "{} evaluations, Euler, video sigmas {:?}, audio sigmas {:?}",
            preset.evaluations(),
            video.sigmas,
            audio.sigmas
        );
        let scales: std::collections::BTreeSet<_> = adapter
            .projections
            .iter()
            .map(|p| p.scale.to_string())
            .collect();
        println!("alpha/rank scales: {scales:?}");
        println!("Inspection only: native adapter execution is not enabled by this command.");
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
