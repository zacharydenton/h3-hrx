//! MiniMax H3 inference against ComfyUI's checkpoints, with every GPU kernel in Loom.
//!
//! # The supported API
//!
//! [`Session`] is the whole of it: open one against the four checkpoints, then denoise, decode and
//! encode. Requests are described by [`DenoiseParams`], [`Reference`], [`Keyframe`] and [`Clip`],
//! which validate their own buffers against the extents they claim; [`Shape`] says what a request
//! produces before any work begins. Errors are [`Error`], and a failing call never leaves a partial
//! result behind.
//!
//! ```no_run
//! # fn main() -> h3_hrx::Result<()> {
//! use h3_hrx::{Config, DenoiseParams, Noise, Session, Tokenizer};
//! let ids = Tokenizer::new()?.encode("a red fox in snow")?;
//! // Safety: the checkpoints are not written while this session lives.
//! let mut session = unsafe { Session::new(Config::default()) }?;
//! let latents = session.denoise(&ids, &DenoiseParams::default(), Noise::default(), &[], &[], None)?;
//! # let _ = latents; Ok(()) }
//! ```
//!
//! # Low-level API
//!
//! The model, kernel, and checkpoint modules support diagnostics and custom pipelines.
//! Prefer [`Session`] for inference. This experimental crate's low-level API may change
//! between releases.
//!
//! # What this crate trusts
//!
//! A checkpoint is memory-mapped and read as the model it claims to be. The file must not be modified
//! or truncated while a [`Session`] holds it: the mapping is live, and a writer would change bytes the
//! library has already validated, or shrink the file under reads that are in flight. Point a session
//! at files you control.

pub mod adapter;
pub mod avae;
pub mod checkpoint;
pub mod compile;
pub mod dispatch;
pub mod dit;
pub mod layout;
pub mod model;
pub mod plan;
pub mod stack;
pub mod te;
pub mod vision;
pub mod vvae;
pub mod weights;

// Reached from nowhere outside this crate, and so never public.

mod cache;
mod conditioning;
pub mod error;
pub mod models;
mod noise;
mod pixels;
/// The reference resampler.
///
/// Reference and keyframe images are scaled with PIL's bilinear filter because
/// that is what the model was conditioned against; another resampler shifts the
/// conditioning it sees. In the library rather than the CLI so that an
/// application adapter scales its inputs the same way the CLI does.
pub mod resize;
mod rope;
mod sampler;
pub mod session;
mod tiles;
pub mod tokenizer;
mod trace;

pub use cache::{CachePolicy, CacheThresholds};
pub use dit::{
    Attention, DenoiseParams, Keyframe, LatentGrid, Latents, Noise, Presented, Reference, Sampler,
};
pub use error::{Error, Result};
pub use layout::{shape_for, Shape};
pub use session::{Config, ResidencyPolicy, Session, SessionOptions};
pub use tokenizer::Tokenizer;
pub use vvae::Clip;
