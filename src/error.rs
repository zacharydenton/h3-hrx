//! The crate's error type.
use crate::{checkpoint, compile, weights};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Models(#[from] crate::models::Error),
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
    fn every_layer_converts_without_losing_its_message() {
        let e: Error = weights::Error::NoRecipe("h3.final.out.w".into()).into();
        assert_eq!(e.to_string(), "no recipe for tensor h3.final.out.w");
        let e: Error = compile::Error::Io("no space".into()).into();
        assert_eq!(e.to_string(), "no space");
    }
}
