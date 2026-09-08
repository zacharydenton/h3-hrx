//! The C ABI, as a thin facade over `Session`.
//!
//! Everything awkward about a C interface lives here and nowhere else: raw pointers, out-parameters
//! and the per-session lock. The layer underneath takes slices and returns `Result`.
//!
//! Three rules hold throughout. Every entry point catches panics, because unwinding across the ABI is
//! undefined behaviour. Every pointer-and-count pair becomes a slice exactly once, after checking the
//! count against what the shape says it must be — a short buffer gets `H3_INVALID_ARGUMENT`, not a
//! write past the end. And a failing call leaves its message in a thread-local that `h3_last_error`
//! reads, rather than making all twelve signatures carry a buffer and a capacity they rarely use.
// The types keep their C spelling so the header and this file read the same way, and so cbindgen
// emits them unchanged.
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
use crate::dit::{DenoiseParams, KeyframeInput, Noise, RefInput};
use crate::error::{code, Error};
use crate::session::{Config, Session};
use crate::vvae::Clip;
use std::ffi::{c_char, c_int, c_void, CStr};

pub const ABI_VERSION: u32 = 8;

/// What an entry point returns. The distinction is the caller's: `INVALID_ARGUMENT` means the request
/// was not one this library serves, `CANCELLED` that a progress callback stopped the run, and `ERROR`
/// that something failed on the way.
#[repr(C)]
pub enum h3_status {
    H3_OK = 0,
    H3_ERROR = 1,
    H3_CANCELLED = 2,
    H3_INVALID_ARGUMENT = 64,
}

const OK: c_int = h3_status::H3_OK as c_int;
const ERROR: c_int = h3_status::H3_ERROR as c_int;

/// Where the checkpoints live and how kernels are built. Every checkpoint is optional: one is opened
/// only when a call needs it, and a NULL leaves the calls that would use it unavailable.
#[repr(C)]
pub struct h3_config {
    /// ComfyUI's minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors, read as it is
    pub dit_file: *const c_char,
    /// qwen3vl_32b_minimax_h3_int8_convrot.safetensors: the encoder's layers, its embedding table,
    /// and the vision tower
    pub te_file: *const c_char,
    /// minimax_h3_video_vae_fp16.safetensors: the video decoder and encoder
    pub video_vae_file: *const c_char,
    /// minimax_h3_audio_vae_fp32.safetensors: the vocoder and the audio encoder
    pub audio_vae_file: *const c_char,
    /// the repo's kernels/ directory (.loom files)
    pub kernel_sources: *const c_char,
    /// where compiled .hsaco files live (created)
    pub cache_dir: *const c_char,
    /// path to the loom-compile binary
    pub loom_compile: *const c_char,
    /// the DiT attention's QK^T operands: 16 (f16), 8 (int8, the parity path) or 4 (int4); 0 means 8
    pub attn_qk_bits: c_int,
}

/// What a run asks for.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct h3_params {
    /// pixels, multiples of 32
    pub height: c_int,
    pub width: c_int,
    /// snapped up to the next 17n + 5
    pub frames: c_int,
    /// sigma grid points, so steps - 1 model evaluations, as diffusers counts them
    pub steps: c_int,
    pub seed: u64,
    /// 0 means the model's defaults, 12 and 3
    pub video_shift: f32,
    pub audio_shift: f32,
    /// 0 = Euler per stream schedule; 1 = res_multistep on the video sigma grid with the audio
    /// carried as (sigma_v / sigma_a) x_a, which is what the stock workflows use
    pub sampler: c_int,
    /// first-block step cache: skip blocks 1..49 while the accumulated relative change of block 0's
    /// output stays below this. 0 turns it off; 0.05 to 0.15 is the useful range
    pub cache_threshold: f32,
}

/// The shapes a request produces, after snapping: video latents `[24][latent_t][lat_h][lat_w]` and
/// audio latents `[2][32][audio_t]`.
#[repr(C)]
#[derive(Default)]
pub struct h3_shape {
    pub frames: c_int,
    pub latent_t: c_int,
    pub lat_h: c_int,
    pub lat_w: c_int,
    pub audio_t: c_int,
    pub text_rows_max: c_int,
}

/// A reference block (ref2va), in presentation order. Unused pointers are NULL and unused counts 0.
#[repr(C)]
pub struct h3_ref {
    /// 0 = image (one latent frame), 1 = audio, 2 = video (with an optional soundtrack)
    pub kind: c_int,
    pub video_latent: *const f32,
    pub latent_t: c_int,
    pub lat_h: c_int,
    pub lat_w: c_int,
    pub audio_latent: *const f32,
    pub audio_t: c_int,
    /// images: the reference as presented to the text encoder, f32 [height][width][3] in [0, 1] with
    /// both sides multiples of 32. The ids carry (height/32)*(width/32) placeholders (-1) for it
    pub pixels: *const f32,
    pub height: c_int,
    pub width: c_int,
}

/// A keyframe (fl2va): one latent frame pinned at a frame index, presented before any reference.
#[repr(C)]
pub struct h3_keyframe {
    /// 0 for the first frame, or frames - 1 after snapping for the last
    pub frame_index: c_int,
    /// [24][1][lat_h][lat_w] on the generation's own latent grid
    pub video_latent: *const f32,
    /// the same frame as pixels for the encoder's presentation, f32 [height][width][3]
    pub pixels: *const f32,
    pub height: c_int,
    pub width: c_int,
    /// optional, and never denoised
    pub audio_latent: *const f32,
    pub audio_t: c_int,
}

/// Called after every denoising step; return nonzero to cancel.
pub type h3_progress = Option<
    unsafe extern "C" fn(user: *mut c_void, step: c_int, steps: c_int, seconds: f64) -> c_int,
>;

/// The session as the ABI hands it out. The lock is here rather than in `Session` because it exists
/// for the ABI's sake: a C caller may use one session from any thread, while a Rust caller gets
/// `&mut self` and needs no lock at all.
pub struct h3_session {
    inner: std::sync::Mutex<Session>,
}

thread_local! {
    /// The last failure on this thread, kept alive until the next one so `h3_last_error` can hand out
    /// a pointer into it.
    static LAST_ERROR: std::cell::RefCell<std::ffi::CString> =
        std::cell::RefCell::new(std::ffi::CString::default());
}

fn set_error(message: &str) {
    // interior NULs cannot reach C, so they are replaced rather than truncating the message there
    let cleaned = message.replace('\0', "?");
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = std::ffi::CString::new(cleaned).unwrap_or_default();
    });
}

/// The last failing call's message on this thread, or an empty string.
///
/// The pointer stays valid until the next failing call on the same thread; copy it if you need it
/// longer. It is never NULL.
#[no_mangle]
pub extern "C" fn h3_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Runs a call, turning a panic or an error into the ABI's code and recording its message.
///
/// The panic hook is not touched, so a panic still prints where it happened before being converted.
fn guard(f: impl FnOnce() -> Result<(), Error> + std::panic::UnwindSafe) -> c_int {
    match std::panic::catch_unwind(f) {
        Ok(Ok(())) => OK,
        Ok(Err(e)) => {
            set_error(&e.to_string());
            code(&e)
        }
        Err(p) => {
            let what = p
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked".into());
            set_error(&format!("internal error: {what}"));
            ERROR
        }
    }
}

unsafe fn path(p: *const c_char) -> Option<std::path::PathBuf> {
    (!p.is_null())
        .then(|| std::path::PathBuf::from(CStr::from_ptr(p).to_string_lossy().into_owned()))
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if p.is_null() || n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

/// A count that must be at least what the shape needs.
fn enough(what: &str, given: usize, need: usize) -> Result<(), Error> {
    if given < need {
        return Err(Error::Invalid(format!(
            "{what}: {given} elements given, {need} needed"
        )));
    }
    Ok(())
}

impl From<&h3_params> for DenoiseParams {
    fn from(p: &h3_params) -> Self {
        Self {
            height: p.height,
            width: p.width,
            frames: p.frames,
            steps: p.steps.max(0) as usize,
            seed: p.seed,
            sampler: p.sampler,
            video_shift: f64::from(p.video_shift),
            audio_shift: f64::from(p.audio_shift),
            cache_threshold: p.cache_threshold,
        }
    }
}

#[no_mangle]
pub extern "C" fn h3_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn h3_create(
    config: *const h3_config,
    out_session: *mut *mut h3_session,
) -> c_int {
    guard(|| {
        if config.is_null() || out_session.is_null() {
            return Err(Error::Invalid("config and out_session are required".into()));
        }
        let c = &*config;
        let (Some(kernel_sources), Some(cache_dir), Some(loom_compile)) = (
            path(c.kernel_sources),
            path(c.cache_dir),
            path(c.loom_compile),
        ) else {
            return Err(Error::Invalid(
                "kernel_sources, cache_dir and loom_compile are required".into(),
            ));
        };
        let session = Session::new(Config {
            dit: path(c.dit_file),
            te: path(c.te_file),
            video_vae: path(c.video_vae_file),
            audio_vae: path(c.audio_vae_file),
            kernel_sources,
            cache_dir,
            loom_compile: loom_compile.to_string_lossy().into_owned(),
            attn_qk_bits: if c.attn_qk_bits == 0 {
                8
            } else {
                c.attn_qk_bits.max(0) as usize
            },
        })?;
        *out_session = Box::into_raw(Box::new(h3_session {
            inner: std::sync::Mutex::new(session),
        }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn h3_destroy(s: *mut h3_session) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

#[no_mangle]
pub unsafe extern "C" fn h3_shape_for(params: *const h3_params, out: *mut h3_shape) -> c_int {
    if params.is_null() || out.is_null() {
        return crate::error::code(&Error::Invalid(String::new()));
    }
    let p = &*params;
    match Session::shape_for(p.height, p.width, p.frames) {
        Some(sh) => {
            *out = h3_shape {
                frames: sh.frames,
                latent_t: sh.latent_t,
                lat_h: sh.lat_h,
                lat_w: sh.lat_w,
                audio_t: sh.audio_t,
                text_rows_max: sh.text_rows_max,
            };
            OK
        }
        None => crate::error::code(&Error::Invalid(String::new())),
    }
}

/// Locks the session and runs a call on it.
unsafe fn with(
    s: *mut h3_session,
    f: impl FnOnce(&mut Session) -> Result<(), Error>,
) -> Result<(), Error> {
    if s.is_null() {
        return Err(Error::Invalid("session is NULL".into()));
    }
    // A poisoned lock means an earlier call panicked inside; the session is not trustworthy after
    // that, so say so rather than handing back state of unknown shape.
    let mut guard = (*s).inner.lock().map_err(|_| {
        Error::Other("the session was left inconsistent by an earlier failure".into())
    })?;
    f(&mut guard)
}

#[no_mangle]
pub unsafe extern "C" fn h3_text_in(
    s: *mut h3_session,
    ids: *const i32,
    n_ids: c_int,
    out: *mut f32,
    out_elements: usize,
) -> c_int {
    guard(|| {
        let n = n_ids.max(0) as usize;
        if n == 0 {
            return Err(Error::Invalid("ids must hold at least one token".into()));
        }
        enough("text_in out", out_elements, n * crate::model::HID)?;
        let ids = slice(ids, n).to_vec();
        with(s, |session| {
            let rows = std::slice::from_raw_parts_mut(out, n * crate::model::HID);
            session.text_in(&ids, rows)
        })
    })
}

/// The reference and keyframe arrays, borrowed from the caller's memory for one call.
unsafe fn refs_of<'a>(p: *const h3_ref, n: c_int) -> Vec<RefInput<'a>> {
    slice(p, n.max(0) as usize)
        .iter()
        .map(|r| {
            let vlen = (r.latent_t * r.lat_h * r.lat_w).max(0) as usize * crate::model::LATENT_CH;
            let alen = r.audio_t.max(0) as usize * 2 * crate::avae::AUDIO_CH;
            let plen = (r.height * r.width).max(0) as usize * 3;
            RefInput {
                kind: r.kind,
                video_latent: (!r.video_latent.is_null()).then(|| slice(r.video_latent, vlen)),
                latent_t: r.latent_t,
                lat_h: r.lat_h,
                lat_w: r.lat_w,
                audio_latent: (!r.audio_latent.is_null()).then(|| slice(r.audio_latent, alen)),
                audio_t: r.audio_t,
                pixels: (!r.pixels.is_null()).then(|| slice(r.pixels, plen)),
                height: r.height,
                width: r.width,
            }
        })
        .collect()
}

unsafe fn keyframes_of<'a>(
    p: *const h3_keyframe,
    n: c_int,
    lat_h: c_int,
    lat_w: c_int,
) -> Vec<KeyframeInput<'a>> {
    let vlen = (lat_h * lat_w).max(0) as usize * crate::model::LATENT_CH;
    slice(p, n.max(0) as usize)
        .iter()
        .map(|k| KeyframeInput {
            frame_index: k.frame_index,
            video_latent: slice(k.video_latent, vlen),
            audio_latent: (!k.audio_latent.is_null()).then(|| {
                slice(
                    k.audio_latent,
                    k.audio_t.max(0) as usize * 2 * crate::avae::AUDIO_CH,
                )
            }),
            audio_t: k.audio_t,
            pixels: (!k.pixels.is_null())
                .then(|| slice(k.pixels, (k.height * k.width).max(0) as usize * 3)),
            height: k.height,
            width: k.width,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
unsafe fn denoise_into(
    s: *mut h3_session,
    ids: *const i32,
    n_ids: c_int,
    params: *const h3_params,
    keyframes: *const h3_keyframe,
    n_keyframes: c_int,
    refs: *const h3_ref,
    n_refs: c_int,
    noise_video: *const f32,
    noise_audio: *const f32,
    video_latents: *mut f32,
    video_elements: usize,
    audio_latents: *mut f32,
    audio_elements: usize,
    progress: h3_progress,
    user: *mut c_void,
) -> Result<(), Error> {
    if params.is_null() {
        return Err(Error::Invalid("params is NULL".into()));
    }
    let p = &*params;
    let n = n_ids.max(0) as usize;
    if n == 0 {
        return Err(Error::Invalid("ids must hold at least one token".into()));
    }
    let Some(sh) = Session::shape_for(p.height, p.width, p.frames) else {
        return Err(Error::Invalid(
            "no such shape: height and width must be multiples of 32 and the frame count in range"
                .into(),
        ));
    };
    let vlen = crate::model::LATENT_CH * (sh.latent_t * sh.lat_h * sh.lat_w).max(0) as usize;
    let alen = 2 * crate::avae::AUDIO_CH * sh.audio_t.max(0) as usize;
    enough("video latents", video_elements, vlen)?;
    enough("audio latents", audio_elements, alen)?;

    let ids = slice(ids, n).to_vec();
    let dp = DenoiseParams::from(p);
    let noise = Noise {
        video: (!noise_video.is_null()).then(|| slice(noise_video, vlen)),
        audio: (!noise_audio.is_null()).then(|| slice(noise_audio, alen)),
    };
    let refs = refs_of(refs, n_refs);
    let kfs = keyframes_of(keyframes, n_keyframes, sh.lat_h, sh.lat_w);

    // The callback runs while the session lock is held, which is what the C does too: a progress
    // callback that re-entered the session would deadlock there and does here.
    let mut cb = progress.map(|f| {
        move |step: usize, steps: usize, seconds: f64| {
            f(user, step as c_int, steps as c_int, seconds) != 0
        }
    });
    let out = with(s, |session| {
        let latents = session.denoise(
            &ids,
            &dp,
            noise,
            &refs,
            &kfs,
            cb.as_mut()
                .map(|c| c as &mut dyn FnMut(usize, usize, f64) -> bool),
        )?;
        std::slice::from_raw_parts_mut(video_latents, vlen).copy_from_slice(&latents.video);
        std::slice::from_raw_parts_mut(audio_latents, alen).copy_from_slice(&latents.audio);
        Ok(())
    });
    out
}

/// Prompt ids to model-space latents.
///
/// `keyframes` and `refs` may be NULL with a count of zero, which is the plain text-to-video case:
/// there is one entry point rather than two, because a reference-free run is not a different call.
/// `noise_video` and `noise_audio` are optional standard-normal draws in the output layouts, for
/// reproducible comparisons; without them the seed's own generator is used.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn h3_denoise(
    s: *mut h3_session,
    ids: *const i32,
    n_ids: c_int,
    params: *const h3_params,
    keyframes: *const h3_keyframe,
    n_keyframes: c_int,
    refs: *const h3_ref,
    n_refs: c_int,
    noise_video: *const f32,
    noise_audio: *const f32,
    video_latents: *mut f32,
    video_elements: usize,
    audio_latents: *mut f32,
    audio_elements: usize,
    progress: h3_progress,
    user: *mut c_void,
) -> c_int {
    guard(|| {
        denoise_into(
            s,
            ids,
            n_ids,
            params,
            keyframes,
            n_keyframes,
            refs,
            n_refs,
            noise_video,
            noise_audio,
            video_latents,
            video_elements,
            audio_latents,
            audio_elements,
            progress,
            user,
        )
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn h3_decode_video(
    s: *mut h3_session,
    params: *const h3_params,
    video_latents: *const f32,
    video_elements: usize,
    frames: *mut u8,
    frame_bytes: usize,
) -> c_int {
    guard(|| {
        if params.is_null() {
            return Err(Error::Invalid("params is NULL".into()));
        }
        let p = &*params;
        let Some(sh) = Session::shape_for(p.height, p.width, p.frames) else {
            return Err(Error::Invalid("no such shape".into()));
        };
        let vlen = crate::model::LATENT_CH * (sh.latent_t * sh.lat_h * sh.lat_w).max(0) as usize;
        let pixels = (sh.frames * p.height * p.width).max(0) as usize * 3;
        enough("video latents", video_elements, vlen)?;
        enough("frames", frame_bytes, pixels)?;
        let z = slice(video_latents, vlen).to_vec();
        with(s, |session| {
            let out = std::slice::from_raw_parts_mut(frames, pixels);
            session.decode_video(&sh, &z, out)
        })
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn h3_encode_video(
    s: *mut h3_session,
    pixels: *const f32,
    frames: c_int,
    height: c_int,
    width: c_int,
    latents: *mut f32,
    latent_elements: usize,
    latent_t: *mut c_int,
) -> c_int {
    guard(|| {
        let (f, h, w) = (
            frames.max(0) as usize,
            height.max(0) as usize,
            width.max(0) as usize,
        );
        if f == 0 || h == 0 || w == 0 {
            return Err(Error::Invalid(
                "frames, height and width are required".into(),
            ));
        }
        let px = slice(pixels, f * h * w * 3).to_vec();
        let clip = Clip {
            pixels: &px,
            frames: f,
            height: h,
            width: w,
        };
        with(s, |session| {
            let (z, t) = session.encode_video(clip)?;
            enough("video latents", latent_elements, z.len())?;
            std::slice::from_raw_parts_mut(latents, z.len()).copy_from_slice(&z);
            if !latent_t.is_null() {
                *latent_t = t as c_int;
            }
            Ok(())
        })
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn h3_decode_audio(
    s: *mut h3_session,
    audio_latents: *const f32,
    audio_elements: usize,
    audio_t: c_int,
    samples: *mut f32,
    sample_elements: usize,
) -> c_int {
    guard(|| {
        let t = audio_t.max(0) as usize;
        if t == 0 {
            return Err(Error::Invalid("audio_t must be at least one".into()));
        }
        let alen = 2 * crate::avae::AUDIO_CH * t;
        let slen = 2 * t * crate::avae::HOP;
        enough("audio latents", audio_elements, alen)?;
        enough("samples", sample_elements, slen)?;
        let z = slice(audio_latents, alen).to_vec();
        with(s, |session| {
            let out = std::slice::from_raw_parts_mut(samples, slen);
            session.decode_audio(&z, t, out)
        })
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn h3_encode_audio(
    s: *mut h3_session,
    samples: *const f32,
    n_samples: c_int,
    latents: *mut f32,
    latent_elements: usize,
    audio_t: *mut c_int,
) -> c_int {
    guard(|| {
        let n = n_samples.max(0) as usize;
        if n == 0 {
            return Err(Error::Invalid("n_samples must be at least one".into()));
        }
        let x = slice(samples, 2 * n).to_vec();
        with(s, |session| {
            let (z, t) = session.encode_audio(&x, n)?;
            enough("audio latents", latent_elements, z.len())?;
            std::slice::from_raw_parts_mut(latents, z.len()).copy_from_slice(&z);
            if !audio_t.is_null() {
                *audio_t = t as c_int;
            }
            Ok(())
        })
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn h3_vision_embed(
    s: *mut h3_session,
    pixels: *const f32,
    height: c_int,
    width: c_int,
    merged: *mut f32,
    merged_elements: usize,
    deepstack: *mut f32,
    deepstack_elements: usize,
    tokens: *mut c_int,
) -> c_int {
    guard(|| {
        let (h, w) = (height.max(0) as usize, width.max(0) as usize);
        let px = slice(pixels, h * w * 3).to_vec();
        with(s, |session| {
            let e = session.vision_embed(&px, h, w)?;
            enough("merged", merged_elements, e.merged.len())?;
            enough("deepstack", deepstack_elements, e.deepstack.len())?;
            std::slice::from_raw_parts_mut(merged, e.merged.len()).copy_from_slice(&e.merged);
            std::slice::from_raw_parts_mut(deepstack, e.deepstack.len())
                .copy_from_slice(&e.deepstack);
            if !tokens.is_null() {
                *tokens = e.tokens as c_int;
            }
            Ok(())
        })
    })
}

// --- the tokenizer's own ABI --------------------------------------------------------------------
//
// It lives in the same library so a caller needs no Python to turn a prompt into ids, and it is
// separate from the session because it needs no device, no checkpoints and no lock: encoding is a
// pure function of the text.

/// A loaded vocabulary.
pub struct h3_tokenizer {
    inner: crate::tokenizer::Tokenizer,
}

/// `tokenizer_json`: an HF tokenizer.json, or NULL for the one compiled into the library
/// (`H3_TOKENIZER=<file>` overrides that). Returns NULL on failure; `h3_last_error` says why.
#[no_mangle]
pub unsafe extern "C" fn h3_tokenizer_create(tokenizer_json: *const c_char) -> *mut h3_tokenizer {
    let built = std::panic::catch_unwind(|| {
        let from = path(tokenizer_json);
        match from {
            Some(p) => crate::tokenizer::Tokenizer::from_file(&p),
            None => crate::tokenizer::Tokenizer::new(),
        }
    });
    match built {
        Ok(Ok(inner)) => Box::into_raw(Box::new(h3_tokenizer { inner })),
        Ok(Err(e)) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error("internal error: the tokenizer panicked");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn h3_tokenizer_destroy(t: *mut h3_tokenizer) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// The number of ids the text encodes to, writing up to `capacity` of them; -1 on failure.
///
/// The count is returned even when it exceeds the buffer, so a caller can size a second call to it.
#[no_mangle]
pub unsafe extern "C" fn h3_tokenizer_encode(
    t: *const h3_tokenizer,
    utf8: *const c_char,
    ids: *mut i32,
    capacity: usize,
) -> c_int {
    if t.is_null() || utf8.is_null() {
        return -1;
    }
    let text = match CStr::from_ptr(utf8).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (*t).inner.encode(text))) {
        Ok(Ok(v)) => {
            if !ids.is_null() {
                let n = v.len().min(capacity);
                std::ptr::copy_nonoverlapping(v.as_ptr(), ids, n);
            }
            v.len() as c_int
        }
        _ => -1,
    }
}

/// The vocabulary size: the largest id plus one among the model's tokens.
#[no_mangle]
pub unsafe extern "C" fn h3_tokenizer_vocab_size(t: *const h3_tokenizer) -> c_int {
    if t.is_null() {
        return -1;
    }
    (*t).inner.vocab_size() as c_int
}
