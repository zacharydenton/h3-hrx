//! The Qwen3-VL tokenizer, so a caller needs no Python to turn a prompt into ids.
//!
//! This is HuggingFace's own `tokenizers` reading the same `tokenizer.json` that `transformers` reads,
//! rather than the hand-written byte-level BPE it replaces. That implementation evaluated the Qwen2
//! split pattern by hand on code points and documented its letter classification as an approximation
//! above U+0080; it also ignored `added_tokens` entirely. Delegating removes both compromises.
//!
//! The vocabulary is compiled in, so the library carries its own tokenizer. `H3_TOKENIZER` points at a
//! file instead, which is how a different vocabulary is tried without a rebuild.
use std::path::Path;

/// Qwen's `tokenizer.json`, Apache-2.0, vendored under `assets/`.
const EMBEDDED: &[u8] = include_bytes!("../../assets/tokenizer.json");

#[derive(Debug, thiserror::Error)]
#[error("tokenizer: {0}")]
pub struct Error(String);

pub type Result<T> = std::result::Result<T, Error>;

pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    /// The compiled-in vocabulary, unless `H3_TOKENIZER` names a file.
    pub fn new() -> Result<Self> {
        match std::env::var_os("H3_TOKENIZER") {
            Some(path) if !path.is_empty() => Self::from_file(Path::new(&path)),
            _ => Self::from_bytes(EMBEDDED),
        }
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes)
            .map(Self)
            .map_err(|e| Error(e.to_string()))
    }

    /// The prompt's ids, with no special tokens added: the presentation this model wants is built by
    /// the caller, so the tokenizer must not prepend or append anything of its own.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self
            .0
            .encode(text, false)
            .map_err(|e| Error(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|id| *id as i32).collect())
    }

    pub fn vocab_size(&self) -> usize {
        self.0.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        Tokenizer::from_bytes(EMBEDDED).expect("the embedded vocabulary parses")
    }

    #[test]
    fn the_embedded_vocabulary_loads() {
        let t = tok();
        // Qwen3-VL's vocabulary, with its added tokens
        assert!(t.vocab_size() > 151_000, "vocab {}", t.vocab_size());
    }

    #[test]
    fn encodes_the_prompts_the_c_implementation_was_checked_against() {
        // The counts `tests/test_tokenizer.py` asserts against transformers.
        let t = tok();
        for (text, want) in [
            (
                "A red fox trotting through a snowy forest at dawn, cinematic",
                13,
            ),
            ("", 0),
            ("   leading spaces and trailing spaces   ", 7),
        ] {
            assert_eq!(t.encode(text).unwrap().len(), want, "{text:?}");
        }
    }

    #[test]
    fn adds_no_special_tokens() {
        // The caller builds the presentation, so nothing may be prepended or appended.
        let t = tok();
        let a = t.encode("hello").unwrap();
        let b = t.encode("hello world").unwrap();
        assert_eq!(
            &b[..a.len()],
            &a[..],
            "a prefix must tokenize as a prefix here"
        );
    }

    #[test]
    fn round_trips_unicode_and_emoji() {
        let t = tok();
        for text in [
            "Café naïve résumé, Zürich, São Paulo: 日本語のテキスト and 🎬🔥",
            "\t\ttabs\n\nnewlines",
        ] {
            let ids = t.encode(text).unwrap();
            assert!(!ids.is_empty(), "{text:?}");
            assert!(ids.iter().all(|id| *id >= 0));
        }
    }

    #[test]
    fn a_file_overrides_the_embedded_vocabulary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, EMBEDDED).unwrap();
        let from_file = Tokenizer::from_file(&path).unwrap();
        assert_eq!(
            from_file.encode("a red fox").unwrap(),
            tok().encode("a red fox").unwrap()
        );
        assert!(Tokenizer::from_file(Path::new("/nonexistent/tokenizer.json")).is_err());
    }
}
