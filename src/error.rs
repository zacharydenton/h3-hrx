//! The crate's error type.
//!
//! The distinction the C ABI draws is preserved here as variants rather than as a convention: a
//! caller's mistake and the pipeline's own failure map to different codes at the boundary
//! (`invalid_argument` to 64, a cancelled run to 2, everything else to 1), and the call sites choose
//! between them deliberately.
use crate::{checkpoint, compile, weights};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Checkpoint(#[from] checkpoint::Error),
    #[error(transparent)]
    Weights(#[from] weights::Error),
    #[error(transparent)]
    Compile(#[from] compile::Error),
    #[error(transparent)]
    Runtime(#[from] hrx::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] crate::tokenizer::Error),
    /// The caller asked for something it may not: a shape the model does not do, a missing checkpoint.
    #[error("{0}")]
    Invalid(String),
    /// A progress callback asked the run to stop.
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The code this error becomes at the C boundary.
pub fn code(e: &Error) -> i32 {
    match e {
        Error::Cancelled => 2,
        Error::Invalid(_) => 64,
        _ => 1,
    }
}

pub fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(message.into()))
}

pub fn other<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Other(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_boundary_codes_are_the_ones_the_abi_promises() {
        assert_eq!(code(&Error::Cancelled), 2);
        assert_eq!(code(&Error::Invalid("no such shape".into())), 64);
        assert_eq!(code(&Error::Other("a kernel failed".into())), 1);
        assert_eq!(
            code(&Error::Weights(weights::Error::Device(
                "out of memory".into()
            ))),
            1
        );
    }

    #[test]
    fn every_layer_converts_without_losing_its_message() {
        let e: Error = weights::Error::NoRecipe("h3.final.out.w".into()).into();
        assert_eq!(e.to_string(), "no recipe for tensor h3.final.out.w");
        let e: Error = compile::Error::Io("no space".into()).into();
        assert_eq!(e.to_string(), "no space");
    }
}
