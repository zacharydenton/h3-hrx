//! Encodes escaped lines from stdin and prints their ids, for comparing this tokenizer against
//! transformers. `\n`, `\t` and `\\` are unescaped; every other character is literal.
use std::io::{BufRead, Write};

fn unescape(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn main() {
    let tok = h3::tokenizer::Tokenizer::new().expect("tokenizer");
    let stdin = std::io::stdin();
    let mut stdout = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let text = unescape(&line.expect("read"));
        let ids = tok.encode(&text).expect("encode");
        let joined: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        writeln!(stdout, "{}", joined.join(" ")).expect("write");
    }
}
