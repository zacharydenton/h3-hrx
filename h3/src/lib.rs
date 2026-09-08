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
//! let mut session = Session::new(Config::default())?;
//! let latents = session.denoise(&ids, &DenoiseParams::default(), Noise::default(), &[], &[], None)?;
//! # let _ = latents; Ok(()) }
//! ```
//!
//! # What is not supported
//!
//! Everything else — the transformer stack, the kernel compiler and cache, the packed layout, the
//! samplers, the checkpoint reader — is an implementation detail behind the `internals` feature. It
//! is public under that feature so the diagnostic examples in `h3/examples/` can drive one stage at a
//! time, not because it is a stable interface. Building against it means tracking this crate's
//! internals, which change with the model.
//!
//! # What this crate trusts
//!
//! A checkpoint is memory-mapped and read as the model it claims to be. The file must not be modified
//! or truncated while a [`Session`] holds it: the mapping is live, and a writer would change bytes the
//! library has already validated, or shrink the file under reads that are in flight. Point a session
//! at files you control.

// Some of the implementation's own API has its only callers in the diagnostic examples, which build
// only under `internals`. Without the feature those items are unreachable and warn, which says
// nothing useful; `cargo clippy --all-features` is the configuration where dead code is really dead,
// and scripts/test.sh runs it that way.
#![cfg_attr(not(feature = "internals"), allow(dead_code))]

/// The implementation. Public only under the `internals` feature; see the crate documentation.
macro_rules! internal {
    ($($m:ident),* $(,)?) => {
        $(
            #[cfg(feature = "internals")]
            pub mod $m;
            #[cfg(not(feature = "internals"))]
            mod $m;
        )*
    };
}

internal!(
    avae,
    cache,
    checkpoint,
    compile,
    conditioning,
    dispatch,
    dit,
    layout,
    model,
    noise,
    pixels,
    plan,
    rope,
    sampler,
    stack,
    te,
    tiles,
    vision,
    vvae,
    weights,
);

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
