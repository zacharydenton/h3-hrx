//! Opt-in benchmark events, independent of the synchronizing kernel profiler.
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn quoted(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c < '\u{20}' => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub(crate) fn checkpoint(model: &str, path: &std::path::Path) {
    event("checkpoint", || {
        let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        format!(
            ",\"model\":{},\"path\":{},\"resolved_path\":{},\"file_bytes\":{}",
            quoted(model),
            quoted(&path.to_string_lossy()),
            quoted(&resolved.to_string_lossy()),
            path.metadata().map(|m| m.len()).unwrap_or(0)
        )
    });
}

pub(crate) fn event(stage: &str, fields: impl FnOnce() -> String) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static START: OnceLock<Instant> = OnceLock::new();
    if !*ENABLED.get_or_init(|| {
        std::env::var_os("H3_STAGE_TRACE").is_some_and(|v| !v.is_empty() && v != "0")
    }) {
        return;
    }
    let elapsed = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    eprintln!("H3_STAGE {{\"stage\":\"{stage}\",\"elapsed_seconds\":{elapsed},\"unix_seconds\":{unix}{}}}", fields());
}
