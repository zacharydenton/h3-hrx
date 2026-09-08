//! Safe wrappers over the C ABI: one owning handle each for the session and the tokenizer, and the
//! error-buffer convention (`0` is success; anything else fills `error` with a sentence) turned into
//! `Result`. Every raw pointer the library reads stays borrowed for the length of the call.
use crate::ffi;
use anyhow::{anyhow, bail, Result};
use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::time::Instant;

/// The library writes at most `capacity` bytes and always terminates; 4096 matches the C CLI.
fn err_buf() -> Vec<c_char> {
    vec![0; 4096]
}

fn err_text(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub struct Tokenizer(*mut ffi::Tokenizer);

impl Tokenizer {
    /// `None` takes the tokenizer compiled into `libh3pipe` (`H3_TOKENIZER` overrides it).
    pub fn new(path: Option<&str>) -> Result<Self> {
        let mut err = err_buf();
        let c_path = path.map(CString::new).transpose()?;
        let ptr = unsafe {
            ffi::h3tok_create(
                c_path.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if ptr.is_null() {
            bail!("tokenizer: {}", err_text(&err));
        }
        Ok(Self(ptr))
    }

    /// Appends the ids for `text`. `h3tok_encode` reports the count the text needs even when the buffer
    /// is smaller, so a short first call sizes the second exactly.
    pub fn encode_into(&self, text: &str, ids: &mut Vec<i32>) -> Result<()> {
        let c_text = CString::new(text)?;
        let mut buf = vec![0i32; 4096];
        let mut n =
            unsafe { ffi::h3tok_encode(self.0, c_text.as_ptr(), buf.as_mut_ptr(), buf.len()) };
        if n < 0 {
            bail!("cannot tokenize {:?}", text);
        }
        if n as usize > buf.len() {
            buf.resize(n as usize, 0);
            n = unsafe { ffi::h3tok_encode(self.0, c_text.as_ptr(), buf.as_mut_ptr(), buf.len()) };
            if n < 0 || n as usize > buf.len() {
                bail!("cannot tokenize {:?}", text);
            }
        }
        ids.extend_from_slice(&buf[..n as usize]);
        Ok(())
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe { ffi::h3tok_destroy(self.0) }
    }
}

/// Prints one line per denoising step with the running estimate. `user` is the start of the run.
unsafe extern "C" fn progress(
    user: *mut c_void,
    step: c_int,
    steps: c_int,
    sec: c_double,
) -> c_int {
    let t0 = &*(user as *const Instant);
    let wall = t0.elapsed().as_secs_f64();
    let left = if step > 0 {
        wall / step as f64 * (steps - step) as f64
    } else {
        0.0
    };
    eprintln!("  step {step}/{steps}  {sec:.1} s  ({left:.0} s left)");
    0
}

pub struct Session(*mut ffi::Session);

impl Session {
    pub fn new(config: &ffi::Config) -> Result<Self> {
        let mut err = err_buf();
        let mut s = std::ptr::null_mut();
        let rc = unsafe { ffi::h3pipe_create(config, &mut s, err.as_mut_ptr(), err.len()) };
        if rc != 0 {
            bail!("create: {}", err_text(&err));
        }
        Ok(Self(s))
    }

    pub fn shape_for(params: &ffi::Params) -> Result<ffi::Shape> {
        let mut shape = ffi::Shape::default();
        if unsafe { ffi::h3pipe_shape_for(params, &mut shape) } != 0 {
            bail!("invalid parameters");
        }
        Ok(shape)
    }

    /// One image or clip through the video VAE's encoder; returns the latent frame count it wrote.
    pub fn encode_video(
        &self,
        pixels: &[f32],
        frames: i32,
        height: i32,
        width: i32,
        latents: &mut [f32],
    ) -> Result<i32> {
        let mut err = err_buf();
        let mut latent_t = 0;
        let rc = unsafe {
            ffi::h3pipe_encode_video(
                self.0,
                pixels.as_ptr(),
                frames,
                height,
                width,
                latents.as_mut_ptr(),
                latents.len(),
                &mut latent_t,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(anyhow!("{}", err_text(&err)));
        }
        Ok(latent_t)
    }

    pub fn encode_audio(
        &self,
        samples: &[f32],
        n_samples: i32,
        latents: &mut [f32],
    ) -> Result<i32> {
        let mut err = err_buf();
        let mut audio_t = 0;
        let rc = unsafe {
            ffi::h3pipe_encode_audio(
                self.0,
                samples.as_ptr(),
                n_samples,
                latents.as_mut_ptr(),
                latents.len(),
                &mut audio_t,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(anyhow!("{}", err_text(&err)));
        }
        Ok(audio_t)
    }

    /// The plain path when there is nothing to condition on, `h3pipe_denoise_refs` otherwise; the two
    /// take the same buffers, so the caller does not care which ran.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        ids: &[i32],
        params: &ffi::Params,
        keyframes: &[ffi::Keyframe],
        refs: &[ffi::Ref],
        video: &mut [f32],
        audio: &mut [f32],
    ) -> Result<()> {
        let mut err = err_buf();
        let t0 = Instant::now();
        let user = &t0 as *const Instant as *mut c_void;
        let rc = if keyframes.is_empty() && refs.is_empty() {
            unsafe {
                ffi::h3pipe_denoise(
                    self.0,
                    ids.as_ptr(),
                    ids.len() as c_int,
                    params,
                    std::ptr::null(),
                    std::ptr::null(),
                    video.as_mut_ptr(),
                    video.len(),
                    audio.as_mut_ptr(),
                    audio.len(),
                    Some(progress),
                    user,
                    err.as_mut_ptr(),
                    err.len(),
                )
            }
        } else {
            unsafe {
                ffi::h3pipe_denoise_refs(
                    self.0,
                    ids.as_ptr(),
                    ids.len() as c_int,
                    params,
                    keyframes.as_ptr(),
                    keyframes.len() as c_int,
                    refs.as_ptr(),
                    refs.len() as c_int,
                    std::ptr::null(),
                    std::ptr::null(),
                    video.as_mut_ptr(),
                    video.len(),
                    audio.as_mut_ptr(),
                    audio.len(),
                    Some(progress),
                    user,
                    err.as_mut_ptr(),
                    err.len(),
                )
            }
        };
        if rc != 0 {
            bail!("denoise: {}", err_text(&err));
        }
        Ok(())
    }

    pub fn decode_video(
        &self,
        params: &ffi::Params,
        video: &[f32],
        frames: &mut [u8],
    ) -> Result<()> {
        let mut err = err_buf();
        let rc = unsafe {
            ffi::h3pipe_decode_video(
                self.0,
                params,
                video.as_ptr(),
                video.len(),
                frames.as_mut_ptr(),
                frames.len(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            bail!("decode video: {}", err_text(&err));
        }
        Ok(())
    }

    pub fn decode_audio(&self, audio: &[f32], audio_t: i32, samples: &mut [f32]) -> Result<()> {
        let mut err = err_buf();
        let rc = unsafe {
            ffi::h3pipe_decode_audio(
                self.0,
                audio.as_ptr(),
                audio.len(),
                audio_t,
                samples.as_mut_ptr(),
                samples.len(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            bail!("decode audio: {}", err_text(&err));
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { ffi::h3pipe_destroy(self.0) }
    }
}
