//! libh3pipe from Rust, declared by hand (no bindgen): prompt -> frames + samples, written as <out>.rgb and <out>.wav.
//!   cargo run --release -- "A red fox ..." [frames] [steps] [out]     (from the repository root, or set H3_ROOT)
//! build.rs links ../../build/libh3pipe.so and sets its rpath.
use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::io::Write;

#[repr(C)]
pub struct H3pipeConfig {
    dit_file: *const c_char, te_file: *const c_char, video_vae_file: *const c_char, audio_vae_file: *const c_char,
    kernel_sources: *const c_char, cache_dir: *const c_char, loom_compile: *const c_char,
    attn_qk_bits: c_int,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct H3pipeParams { height: c_int, width: c_int, frames: c_int, steps: c_int, seed: u64, video_shift: f32, audio_shift: f32, sampler: c_int, cache_threshold: f32 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct H3pipeShape { frames: c_int, latent_t: c_int, lat_h: c_int, lat_w: c_int, audio_t: c_int, text_rows_max: c_int }
#[repr(C)] pub struct H3pipeSession { _private: [u8; 0] }
#[repr(C)] pub struct H3tok { _private: [u8; 0] }
type Progress = Option<unsafe extern "C" fn(user: *mut c_void, step: c_int, steps: c_int, seconds: c_double) -> c_int>;

#[link(name = "h3pipe")]
extern "C" {
    fn h3pipe_abi_version() -> u32;
    fn h3tok_create(tokenizer_json: *const c_char, error: *mut c_char, error_capacity: usize) -> *mut H3tok;
    fn h3tok_destroy(t: *mut H3tok);
    fn h3tok_encode(t: *const H3tok, utf8: *const c_char, ids: *mut i32, capacity: usize) -> c_int;
    fn h3pipe_create(config: *const H3pipeConfig, out: *mut *mut H3pipeSession, error: *mut c_char, error_capacity: usize) -> c_int;
    fn h3pipe_destroy(s: *mut H3pipeSession);
    fn h3pipe_shape_for(params: *const H3pipeParams, out: *mut H3pipeShape) -> c_int;
    fn h3pipe_denoise(s: *mut H3pipeSession, ids: *const i32, n_ids: c_int, params: *const H3pipeParams, noise_video: *const f32, noise_audio: *const f32,
                      video: *mut f32, video_elements: usize, audio: *mut f32, audio_elements: usize, progress: Progress, user: *mut c_void, error: *mut c_char, error_capacity: usize) -> c_int;
    fn h3pipe_decode_video(s: *mut H3pipeSession, params: *const H3pipeParams, video: *const f32, video_elements: usize, frames: *mut u8, frame_bytes: usize, error: *mut c_char, error_capacity: usize) -> c_int;
    fn h3pipe_decode_audio(s: *mut H3pipeSession, audio: *const f32, audio_elements: usize, audio_t: c_int, samples: *mut f32, sample_elements: usize, error: *mut c_char, error_capacity: usize) -> c_int;
}

unsafe extern "C" fn progress(_user: *mut c_void, step: c_int, steps: c_int, seconds: c_double) -> c_int {
    eprintln!("  step {step}/{steps}  {seconds:.1} s"); 0
}

fn err_text(buf: &[c_char]) -> String { unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned() }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("usage: minimal \"prompt\" [frames] [steps] [out]"); std::process::exit(64); }
    let root = std::env::var("H3_ROOT").unwrap_or_else(|_| ".".into());
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let out = args.get(4).cloned().unwrap_or_else(|| "minimal".into());
    let c = |s: String| CString::new(s).unwrap();
    let mut err = vec![0 as c_char; 4096];
    unsafe {
        assert_eq!(h3pipe_abi_version(), 6, "libh3pipe ABI");
        let tok = h3tok_create(std::ptr::null(), err.as_mut_ptr(), err.len());   // the tokenizer compiled into libh3pipe
        if tok.is_null() { eprintln!("tokenizer: {}", err_text(&err)); std::process::exit(1); }
        let mut ids = vec![0i32; 4096];
        let n = h3tok_encode(tok, c(args[1].clone()).as_ptr(), ids.as_mut_ptr(), ids.len());
        h3tok_destroy(tok);
        assert!(n >= 0 && n as usize <= ids.len(), "cannot tokenize the prompt"); ids.truncate(n as usize);

        let models = std::env::var("H3_MODELS").unwrap_or(format!("{home}/comfy-models"));
        let files: Vec<CString> = ["diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                                   "vae/minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"].iter().map(|f| c(format!("{models}/{f}"))).collect();
        let dirs: Vec<CString> = ["kernels", "build/kernel_cache"].iter().map(|d| c(format!("{root}/{d}"))).collect();
        let loom = c(std::env::var("LOOM_COMPILE").unwrap_or_else(|_| "loom-compile".into()));
        let cfg = H3pipeConfig { dit_file: files[0].as_ptr(), te_file: files[1].as_ptr(), video_vae_file: files[2].as_ptr(), audio_vae_file: files[3].as_ptr(),
                                 kernel_sources: dirs[0].as_ptr(), cache_dir: dirs[1].as_ptr(), loom_compile: loom.as_ptr(), attn_qk_bits: 8 };
        let mut s: *mut H3pipeSession = std::ptr::null_mut();
        if h3pipe_create(&cfg, &mut s, err.as_mut_ptr(), err.len()) != 0 { eprintln!("create: {}", err_text(&err)); std::process::exit(1); }

        let p = H3pipeParams { height: 480, width: 864, frames: args.get(2).map_or(124, |v| v.parse().unwrap()), steps: args.get(3).map_or(31, |v| v.parse().unwrap()), seed: 0, video_shift: 0.0, audio_shift: 0.0, sampler: 1, cache_threshold: 0.0 };
        let mut sh = H3pipeShape::default(); h3pipe_shape_for(&p, &mut sh);
        let mut video = vec![0f32; 24 * sh.latent_t as usize * sh.lat_h as usize * sh.lat_w as usize]; let mut audio = vec![0f32; 64 * sh.audio_t as usize];
        eprintln!("{} frames, {}x{}x{} latents, {} audio latents, {} prompt tokens", sh.frames, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t, ids.len());
        if h3pipe_denoise(s, ids.as_ptr(), ids.len() as c_int, &p, std::ptr::null(), std::ptr::null(), video.as_mut_ptr(), video.len(), audio.as_mut_ptr(), audio.len(), Some(progress), std::ptr::null_mut(), err.as_mut_ptr(), err.len()) != 0 {
            eprintln!("denoise: {}", err_text(&err)); std::process::exit(1);
        }
        let mut frames = vec![0u8; sh.frames as usize * p.height as usize * p.width as usize * 3]; let mut samples = vec![0f32; 1600 * sh.audio_t as usize];
        if h3pipe_decode_video(s, &p, video.as_ptr(), video.len(), frames.as_mut_ptr(), frames.len(), err.as_mut_ptr(), err.len()) != 0 { eprintln!("decode video: {}", err_text(&err)); std::process::exit(1); }
        if h3pipe_decode_audio(s, audio.as_ptr(), audio.len(), sh.audio_t, samples.as_mut_ptr(), samples.len(), err.as_mut_ptr(), err.len()) != 0 { eprintln!("decode audio: {}", err_text(&err)); std::process::exit(1); }
        h3pipe_destroy(s);

        std::fs::write(format!("{out}.rgb"), &frames).unwrap();
        let n = sh.audio_t as u32 * 800; let bytes = n * 4;
        let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{out}.wav")).unwrap());
        w.write_all(b"RIFF").unwrap(); w.write_all(&(36 + bytes).to_le_bytes()).unwrap(); w.write_all(b"WAVEfmt ").unwrap();
        w.write_all(&16u32.to_le_bytes()).unwrap(); w.write_all(&1u16.to_le_bytes()).unwrap(); w.write_all(&2u16.to_le_bytes()).unwrap(); w.write_all(&32000u32.to_le_bytes()).unwrap();
        w.write_all(&128000u32.to_le_bytes()).unwrap(); w.write_all(&4u16.to_le_bytes()).unwrap(); w.write_all(&16u16.to_le_bytes()).unwrap(); w.write_all(b"data").unwrap(); w.write_all(&bytes.to_le_bytes()).unwrap();
        for i in 0..n as usize { for ch in 0..2 { let v = samples[ch * n as usize + i].clamp(-1.0, 1.0); w.write_all(&((v * 32767.0) as i16).to_le_bytes()).unwrap(); } }
        eprintln!("wrote {out}.rgb ({} x {}x{} rgb24) and {out}.wav", sh.frames, p.width, p.height);
    }
}
