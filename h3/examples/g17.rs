//! Round-trips float spellings through the %.17g formatter: reads one per line, parses each as the
//! double it denotes, formats it back, and reports any that do not reproduce.
use std::io::BufRead;

fn main() {
    let mut bad = 0;
    for line in std::io::stdin().lock().lines() {
        let want = line.expect("read");
        if want.is_empty() {
            continue;
        }
        let value: f64 = want.parse().expect("a double");
        let got = h3::compile::num(value);
        if got != want {
            println!("MISMATCH  want {want}  got {got}");
            bad += 1;
        } else {
            println!("ok        {want}");
        }
    }
    if bad > 0 {
        std::process::exit(1);
    }
}
