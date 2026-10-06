//! ffmpeg and ffprobe as child processes, and the WAV writer. Arguments are passed as an argv, never
//! through a shell, so paths with quotes or spaces need no escaping.
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

pub const FPS: u32 = 24;
pub const RATE: u32 = 32000;

/// Everything the child writes to stdout, or an error naming the program.
fn capture(program: &str, args: &[&str]) -> Result<Vec<u8>> {
    let out = Command::new(program)
        .args(args)
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| format!("cannot run {program} (is it installed?)"))?;
    if !out.status.success() {
        bail!("{program} failed ({})", out.status);
    }
    Ok(out.stdout)
}

/// Feeds `data` to a child on stdin and waits for it.
fn pipe_into(program: &str, args: &[&str], data: &[u8]) -> Result<()> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .spawn()
        .with_context(|| format!("cannot run {program} (is it installed?)"))?;
    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(data)
        .with_context(|| format!("short write to {program}"))?;
    let status = child.wait()?;
    if !status.success() {
        bail!("{program} failed ({status})");
    }
    Ok(())
}

/// The first video stream's size; a still image is one such stream.
pub fn probe_size(path: &Path) -> Result<(i32, i32)> {
    let path = path.to_str().context("path is not valid UTF-8")?;
    let out = capture(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            path,
        ],
    )?;
    let text = String::from_utf8_lossy(&out);
    let line = text.lines().next().unwrap_or_default();
    let (w, h) = line
        .trim()
        .split_once(',')
        .with_context(|| format!("cannot read the size of {path}"))?;
    let (w, h): (i32, i32) = (w.trim().parse()?, h.trim().parse()?);
    if w <= 0 || h <= 0 {
        bail!("{path} has a zero size");
    }
    Ok((w, h))
}

/// Bounded stdout for original RefMod sources. Abort rather than silently truncate
/// large media; always reap the child, including on a read error.
fn capture_limited(args: &[&str], max_bytes: usize) -> Result<Vec<u8>> {
    let mut child = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("cannot run ffmpeg")?;
    let mut bytes = Vec::new();
    let read = child
        .stdout
        .take()
        .unwrap()
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes);
    if read.is_err() || bytes.len() > max_bytes {
        let _ = child.kill();
        let _ = child.wait();
        read?;
        bail!("refmod source exceeds host media budget (--refmod-media-budget-mib)");
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg failed ({status})");
    }
    Ok(bytes)
}

/// Only presentation frames, with a conservative four-float-buffer budget.
pub fn decode_source_visual(
    path: &Path,
    still: bool,
    size: Option<(i32, i32)>,
    canvas: (i32, i32),
    budget: usize,
) -> Result<(Vec<u8>, i32, i32)> {
    let (w, h) = probe_size(path)?;
    let (w, h) = size.unwrap_or_else(|| h3_hrx::resize::fit(w, h, canvas.0, canvas.1));
    let stride = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(3))
        .context("refmod source dimensions overflow")?;
    if stride > budget / 16 {
        bail!("refmod source exceeds host media budget (--refmod-media-budget-mib)");
    }
    let filter = if still && size.is_some() {
        format!("scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}")
    } else if still {
        format!("scale={w}:{h}")
    } else {
        format!("fps=2:start_time=0,scale={w}:{h}")
    };
    let p = path.to_str().context("source path is not UTF-8")?;
    let mut args = vec!["-v", "error", "-i", p, "-an", "-vf", &filter];
    if still {
        args.extend(["-frames:v", "1"]);
    }
    args.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
    let raw = capture_limited(&args, budget / 16)?;
    if raw.is_empty() || !raw.len().is_multiple_of(stride) {
        bail!("invalid decoded refmod source visual");
    }
    Ok((raw, w, h))
}

pub fn decode_source_audio(path: &Path, budget: usize) -> Result<Vec<f32>> {
    let p = path.to_str().context("source path is not UTF-8")?;
    let raw = capture_limited(
        &[
            "-v", "error", "-i", p, "-vn", "-ac", "2", "-ar", "32000", "-f", "f32le", "-",
        ],
        budget / 4,
    )?;
    if raw.is_empty() || !raw.len().is_multiple_of(8) {
        bail!("invalid decoded refmod source audio");
    }
    let n = raw.len() / 8;
    let mut samples = vec![0.0; n * 2];
    for (i, pair) in raw.as_chunks::<8>().0.iter().enumerate() {
        samples[i] = f32::from_le_bytes(pair[..4].try_into().unwrap());
        samples[n + i] = f32::from_le_bytes(pair[4..].try_into().unwrap());
    }
    Ok(samples)
}

/// One frame of any format ffmpeg reads, as RGB8 `[h][w][3]`.
pub fn decode_image(path: &Path) -> Result<(Vec<u8>, i32, i32)> {
    let (w, h) = probe_size(path)?;
    let p = path.to_str().context("path is not valid UTF-8")?;
    let rgb = capture(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-i",
            p,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
    )?;
    let want = w as usize * h as usize * 3;
    if rgb.len() != want {
        bail!("{p}: ffmpeg produced {} bytes, expected {want}", rgb.len());
    }
    Ok((rgb, w, h))
}

/// Any audio file as planar stereo `[2][n]` at 32 kHz; mono is duplicated and other rates resampled.
pub fn decode_audio(path: &Path) -> Result<(Vec<f32>, i32)> {
    decode_audio_inner(path, None)
}

/// Bound decoding before allocating audio samples for refmod creation.
pub fn decode_audio_limited(path: &Path, seconds: f32) -> Result<(Vec<f32>, i32)> {
    decode_audio_inner(path, Some(seconds))
}

fn decode_audio_inner(path: &Path, seconds: Option<f32>) -> Result<(Vec<f32>, i32)> {
    let p = path.to_str().context("path is not valid UTF-8")?;
    let duration = seconds.map(|s| s.to_string());
    let mut args = vec!["-v", "error", "-i", p];
    if let Some(s) = &duration {
        args.extend(["-t", s.as_str()]);
    }
    args.extend(["-vn", "-f", "f32le", "-ac", "2", "-ar", "32000", "-"]);
    let raw = capture("ffmpeg", &args)?;
    if raw.len() < 8 {
        bail!("{p}: no audio samples");
    }
    let n = raw.len() / 8; // one interleaved stereo frame is two f32
    let mut samples = vec![0.0f32; 2 * n];
    for i in 0..n {
        let at = |k: usize| f32::from_le_bytes(raw[k * 4..k * 4 + 4].try_into().unwrap());
        samples[i] = at(2 * i);
        samples[n + i] = at(2 * i + 1);
    }
    Ok((samples, n as i32))
}

/// 16-bit PCM WAV: interleaved stereo at `RATE` from planar float samples `[2][n]`.
pub fn wav_bytes(samples: &[f32], n: u32) -> Vec<u8> {
    let data_bytes = n * 4;
    let mut w = Vec::with_capacity(44 + data_bytes as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&2u16.to_le_bytes()); // stereo
    w.extend_from_slice(&RATE.to_le_bytes());
    w.extend_from_slice(&(RATE * 4).to_le_bytes()); // byte rate
    w.extend_from_slice(&4u16.to_le_bytes()); // block align
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_bytes.to_le_bytes());
    for i in 0..n as usize {
        for c in 0..2usize {
            let v = samples[c * n as usize + i].clamp(-1.0, 1.0);
            w.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes());
        }
    }
    w
}

/// One RGB8 frame as an image file; ffmpeg picks the encoder from the extension.
pub fn write_still(path: &Path, frame: &[u8], width: i32, height: i32) -> Result<()> {
    let p = path.to_str().context("path is not valid UTF-8")?;
    let size = format!("{width}x{height}");
    pipe_into(
        "ffmpeg",
        &[
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &size,
            "-i",
            "-",
            "-frames:v",
            "1",
            p,
        ],
        frame,
    )
}

/// RGB8 frames plus the written WAV into an H.264 + AAC mp4.
pub fn mux(out: &Path, wav: &Path, frames: &[u8], width: i32, height: i32) -> Result<()> {
    let (o, w) = (
        out.to_str().context("path is not valid UTF-8")?,
        wav.to_str().context("path is not valid UTF-8")?,
    );
    let size = format!("{width}x{height}");
    let fps = FPS.to_string();
    pipe_into(
        "ffmpeg",
        &[
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &size,
            "-r",
            &fps,
            "-i",
            "-",
            "-i",
            w,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            o,
        ],
        frames,
    )
}

/// Decode an entire reference clip at H3's 24 fps; no implicit audio selection.
pub fn decode_reference_video(
    path: &Path,
    width: i32,
    height: i32,
) -> Result<(Vec<f32>, usize, i32, i32)> {
    let (w, h) = probe_size(path)?;
    let (w, h) = h3_hrx::resize::fit(w, h, width, height);
    let p = path.to_str().context("video path is not UTF-8")?;
    let filter = format!("fps=24,scale={w}:{h}");
    let raw = capture(
        "ffmpeg",
        &[
            "-v", "error", "-i", p, "-an", "-vf", &filter, "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-",
        ],
    )?;
    let stride = w as usize * h as usize * 3;
    if raw.is_empty() || !raw.len().is_multiple_of(stride) {
        bail!("invalid decoded reference video");
    }
    let frames = raw.len() / stride;
    Ok((
        raw.iter().map(|b| *b as f32 / 255.0).collect(),
        frames,
        w,
        h,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_and_samples() {
        let n = 800u32;
        let samples = vec![0.5f32; 2 * n as usize];
        let w = wav_bytes(&samples, n);
        assert_eq!(w.len(), 44 + n as usize * 4);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(&w[36..40], b"data");
        assert_eq!([w[44], w[45]], [0xff, 0x3f]); // 0.5 * 32767 = 16383 = 0x3fff, little-endian
    }

    #[test]
    fn wav_clamps_out_of_range_samples() {
        let samples = vec![2.0f32, -2.0];
        let w = wav_bytes(&samples, 1);
        assert_eq!(i16::from_le_bytes([w[44], w[45]]), 32767);
        assert_eq!(i16::from_le_bytes([w[46], w[47]]), -32767);
    }

    #[test]
    fn wav_interleaves_the_planes() {
        let samples = [1.0f32, 1.0, -1.0, -1.0]; // left plane, then right
        let w = wav_bytes(&samples, 2);
        assert_eq!(i16::from_le_bytes([w[44], w[45]]), 32767); // frame 0 left
        assert_eq!(i16::from_le_bytes([w[46], w[47]]), -32767); // frame 0 right
    }
}
