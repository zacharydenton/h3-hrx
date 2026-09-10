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
//! samplers, the checkpoint reader — is an implementation detail, and private. The `internals`
//! feature opens it, so that `h3-dev` and the integration tests can drive one stage at a time; that
//! is not a promise about it. Nothing under that feature carries a stability guarantee, and all of
//! it changes with the model.
//!
//! # What this crate trusts
//!
//! A checkpoint is memory-mapped and read as the model it claims to be. The file must not be modified
//! or truncated while a [`Session`] holds it: the mapping is live, and a writer would change bytes the
//! library has already validated, or shrink the file under reads that are in flight. Point a session
//! at files you control.

// With the interior closed, the parts of it that only the diagnostics binary and the integration
// tests reach have no caller, and every one of them would be reported as dead. The `internals` build
// is where that report means something — it compiles those callers, and it is what `scripts/test.sh`
// runs — so the warning is silenced only in the build that cannot see them.
#![cfg_attr(not(feature = "internals"), allow(dead_code))]

// The implementation. `internals` opens it to the diagnostics binary and the integration
// tests, which drive one stage at a time; a normal build keeps it shut. Written out rather
// than generated, because rustfmt follows only literal module declarations and a macro here
// would keep every file below out of `cargo fmt`.

#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod avae;
#[cfg(not(feature = "internals"))]
mod avae;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod checkpoint;
#[cfg(not(feature = "internals"))]
mod checkpoint;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod compile;
#[cfg(not(feature = "internals"))]
mod compile;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod dispatch;
#[cfg(not(feature = "internals"))]
mod dispatch;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod dit;
#[cfg(not(feature = "internals"))]
mod dit;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod layout;
#[cfg(not(feature = "internals"))]
mod layout;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod model;
#[cfg(not(feature = "internals"))]
mod model;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod plan;
#[cfg(not(feature = "internals"))]
mod plan;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod stack;
#[cfg(not(feature = "internals"))]
mod stack;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod te;
#[cfg(not(feature = "internals"))]
mod te;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod vision;
#[cfg(not(feature = "internals"))]
mod vision;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod vvae;
#[cfg(not(feature = "internals"))]
mod vvae;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod weights;
#[cfg(not(feature = "internals"))]
mod weights;

// Reached from nowhere outside this crate, and so never public.

mod cache;
mod conditioning;
pub mod error;
pub mod models;
mod noise;
mod pixels;
mod rope;
mod sampler;
pub mod session;
mod tiles;
pub mod tokenizer;

pub use dit::{
    Attention, DenoiseParams, Keyframe, LatentGrid, Latents, Noise, Presented, Reference, Sampler,
};
pub use error::{Error, Result};
pub use layout::{shape_for, Shape};
pub use session::{Config, Session};
pub use tokenizer::Tokenizer;
pub use vvae::Clip;
