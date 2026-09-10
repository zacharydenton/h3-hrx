//! Resolves a checkpoint path the way the pipeline will: local directory, then the shared Hugging Face
//! cache, then the hub.  resolve [--offline] <relative-path>
pub fn run(args: Vec<String>) {
    let mut args = args;
    let offline = args.first().map(|a| a == "--offline").unwrap_or(false);
    if offline {
        args.remove(0);
    }
    let relative = args
        .first()
        .expect("usage: resolve [--offline] <relative-path>");
    let resolver = h3::models::Resolver::new().offline(offline);
    match resolver.find(relative) {
        Ok(path) => println!("{}", path.display()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
