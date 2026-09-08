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
//! # fn main() -> h3::Result<()> {
//! use h3::{Config, DenoiseParams, Noise, Session, Tokenizer};
//! let ids = Tokenizer::new()?.encode("a red fox in snow")?;
//! // Safety: the checkpoints are not written while this session lives.
//! let mut session = unsafe { Session::new(Config::default()) }?;
//! let latents = session.denoise(&ids, &DenoiseParams::default(), Noise::default(), &[], &[], None)?;
//! # let _ = latents; Ok(()) }
//! ```
//!
//! # What is not supported
//!
//! Everything else — the transformer stack, the kernel compiler and cache, the packed layout, the
//! samplers, the checkpoint reader — is an implementation detail. Those modules are public, and
//! `#[doc(hidden)]`, so the diagnostic examples in `h3/examples/` can drive one stage at a time; that
//! is not a promise about them. They carry no stability guarantee and change with the model.
//!
//! # What this crate trusts
//!
//! A checkpoint is memory-mapped and read as the model it claims to be. The file must not be modified
//! or truncated while a [`Session`] holds it: the mapping is live, and a writer would change bytes the
//! library has already validated, or shrink the file under reads that are in flight. Point a session
//! at files you control.

// The implementation. Public so the diagnostic examples in `h3/examples/` can drive one stage at a
// time, and `#[doc(hidden)]` because it is not an interface anyone should build against. Written
// out as plain `mod` items rather than generated: rustfmt follows only literal module
// declarations, and a macro here would keep every file below out of `cargo fmt`.
#[doc(hidden)]
pub mod avae;
#[doc(hidden)]
pub mod cache;
#[doc(hidden)]
pub mod checkpoint;
#[doc(hidden)]
pub mod compile;
#[doc(hidden)]
pub mod conditioning;
#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dit;
#[doc(hidden)]
pub mod layout;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod noise;
#[doc(hidden)]
pub mod pixels;
#[doc(hidden)]
pub mod plan;
#[doc(hidden)]
pub mod rope;
#[doc(hidden)]
pub mod sampler;
#[doc(hidden)]
pub mod stack;
#[doc(hidden)]
pub mod te;
#[doc(hidden)]
pub mod tiles;
#[doc(hidden)]
pub mod vision;
#[doc(hidden)]
pub mod vvae;
#[doc(hidden)]
pub mod weights;

pub mod capi;
pub mod error;
pub mod models;
pub mod session;
pub mod tokenizer;

pub use dit::{
    Attention, DenoiseParams, Keyframe, LatentGrid, Latents, Noise, Presented, Reference, Sampler,
};
pub use error::{Error, Result};
pub use layout::{shape_for, Shape};
pub use session::{Config, Session};
pub use tokenizer::Tokenizer;
pub use vvae::Clip;
