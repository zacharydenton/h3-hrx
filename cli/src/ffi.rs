//! `libh3pipe`'s C ABI (`host/h3pipe.h`, `host/h3tok.h`), declared by hand: no bindgen, so the whole
//! surface a client needs is visible here. `docs/abi.md` is the contract; `H3PIPE_ABI_VERSION` is checked
//! at startup, because a struct that has gained a field is otherwise read as silent garbage.
use std::ffi::{c_char, c_double, c_int, c_void};

pub const ABI_VERSION: u32 = 7;

#[repr(C)]
pub struct Config {
    pub dit_file: *const c_char,
    pub te_file: *const c_char,
    pub video_vae_file: *const c_char,
    pub audio_vae_file: *const c_char,
    pub kernel_sources: *const c_char,
    pub cache_dir: *const c_char,
    pub loom_compile: *const c_char,
    pub attn_qk_bits: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Params {
    pub height: c_int,
    pub width: c_int,
    pub frames: c_int,
    pub steps: c_int,
    pub seed: u64,
    pub video_shift: f32,
    pub audio_shift: f32,
    pub sampler: c_int,
    pub cache_threshold: f32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Shape {
    pub frames: c_int,
    pub latent_t: c_int,
    pub lat_h: c_int,
    pub lat_w: c_int,
    pub audio_t: c_int,
    pub text_rows_max: c_int,
}

/// A reference block for ref2va, in presentation order. Unused pointers stay null and unused counts zero,
/// so `Default` is the correct starting point for every kind.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ref {
    pub kind: c_int,
    pub video_latent: *const f32,
    pub latent_t: c_int,
    pub lat_h: c_int,
    pub lat_w: c_int,
    pub audio_latent: *const f32,
    pub audio_t: c_int,
    pub pixels: *const f32,
    pub height: c_int,
    pub width: c_int,
}

impl Default for Ref {
    fn default() -> Self {
        Self {
            kind: 0,
            video_latent: std::ptr::null(),
            latent_t: 0,
            lat_h: 0,
            lat_w: 0,
            audio_latent: std::ptr::null(),
            audio_t: 0,
            pixels: std::ptr::null(),
            height: 0,
            width: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Keyframe {
    pub frame_index: c_int,
    pub video_latent: *const f32,
    pub pixels: *const f32,
    pub height: c_int,
    pub width: c_int,
    pub audio_latent: *const f32,
    pub audio_t: c_int,
}

#[repr(C)]
pub struct Session {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Tokenizer {
    _private: [u8; 0],
}

pub type Progress = Option<
    unsafe extern "C" fn(user: *mut c_void, step: c_int, steps: c_int, seconds: c_double) -> c_int,
>;

#[link(name = "h3pipe")]
extern "C" {
    pub fn h3pipe_abi_version() -> u32;
    pub fn h3pipe_create(
        config: *const Config,
        out_session: *mut *mut Session,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    pub fn h3pipe_destroy(s: *mut Session);
    pub fn h3pipe_shape_for(params: *const Params, out: *mut Shape) -> c_int;
    pub fn h3pipe_denoise(
        s: *mut Session,
        ids: *const i32,
        n_ids: c_int,
        params: *const Params,
        noise_video: *const f32,
        noise_audio: *const f32,
        video_latents: *mut f32,
        video_elements: usize,
        audio_latents: *mut f32,
        audio_elements: usize,
        progress: Progress,
        user: *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn h3pipe_denoise_refs(
        s: *mut Session,
        ids: *const i32,
        n_ids: c_int,
        params: *const Params,
        keyframes: *const Keyframe,
        n_keyframes: c_int,
        refs: *const Ref,
        n_refs: c_int,
        noise_video: *const f32,
        noise_audio: *const f32,
        video_latents: *mut f32,
        video_elements: usize,
        audio_latents: *mut f32,
        audio_elements: usize,
        progress: Progress,
        user: *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    pub fn h3pipe_decode_video(
        s: *mut Session,
        params: *const Params,
        video_latents: *const f32,
        video_elements: usize,
        frames: *mut u8,
        frame_bytes: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    pub fn h3pipe_decode_audio(
        s: *mut Session,
        audio_latents: *const f32,
        audio_elements: usize,
        audio_t: c_int,
        samples: *mut f32,
        sample_elements: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    pub fn h3pipe_encode_video(
        s: *mut Session,
        pixels: *const f32,
        frames: c_int,
        height: c_int,
        width: c_int,
        latents: *mut f32,
        latent_elements: usize,
        latent_t: *mut c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    pub fn h3pipe_encode_audio(
        s: *mut Session,
        samples: *const f32,
        n_samples: c_int,
        latents: *mut f32,
        latent_elements: usize,
        audio_t: *mut c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;

    pub fn h3tok_create(
        tokenizer_json: *const c_char,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut Tokenizer;
    pub fn h3tok_destroy(t: *mut Tokenizer);
    pub fn h3tok_encode(
        t: *const Tokenizer,
        utf8: *const c_char,
        ids: *mut i32,
        capacity: usize,
    ) -> c_int;
}
