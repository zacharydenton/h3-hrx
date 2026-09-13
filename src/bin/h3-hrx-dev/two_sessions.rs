//! Sequential and overlapping session creation, and one from another thread.
pub fn run(_args: Vec<String>) {
    for i in 0..3 {
        match hrx::Stream::open() {
            Ok(_g) => println!("sequential open {i}: OK"),
            Err(e) => {
                println!("sequential open {i}: {e}");
                std::process::exit(1)
            }
        }
    }
    let a = hrx::Stream::open();
    let b = hrx::Stream::open();
    println!(
        "overlapping: {} / {}",
        if a.is_ok() { "OK" } else { "FAILED" },
        if b.is_ok() { "OK" } else { "FAILED" }
    );
    if a.is_err() || b.is_err() {
        std::process::exit(1)
    }
    let t: Vec<_> = (0..4)
        .map(|i| std::thread::spawn(move || (i, hrx::Stream::open().is_ok())))
        .collect();
    for h in t {
        let (i, ok) = h.join().unwrap();
        println!("concurrent open {i}: {}", if ok { "OK" } else { "FAILED" });
        if !ok {
            std::process::exit(1)
        }
    }
    println!("all sessions created");
}
