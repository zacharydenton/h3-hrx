//! The token sequence the model is addressed through.
//!
//! Before a prompt says what to do, the presentation says what is being looked
//! at and listened to: each image as `<Picture n>: ` followed by a span of
//! placeholder vision tokens the encoder fills in, each soundtrack as
//! `<Audio n>: `, and then the prompt. The numbering is what a prompt refers
//! back to, so the order things are announced in is meaning rather than
//! presentation.
//!
//! This is here rather than in a caller because all of it is the model's: the
//! sentinels are fixed by the compiled vocabulary, the span is one token to a
//! 32-pixel block, and the ceiling is the layout's. A caller that builds the
//! sequence itself and gets any of the three slightly wrong produces a prompt
//! the model reads differently, with nothing anywhere to say so.
//!
//! The budget is the real constraint on how many references a clip can take.
//! At 1344x768 an image span is 42x24 = 1008 tokens of the 4096 a request has,
//! so three references leave a little over a thousand for the prompt and four
//! leave almost none.

use crate::{Error, Result, Shape, Tokenizer};

/// The compiled vocabulary's sentinels for the start and end of a vision span.
const VISION_START: i32 = 151652;
const VISION_END: i32 = 151653;

/// The pixels one vision token covers, in each direction.
///
/// An image is announced as `(height / 32) * (width / 32)` placeholders, which
/// is why references are scaled onto a 32-pixel grid.
pub const VISION_BLOCK: i32 = 32;

/// The pixels one latent covers, in each direction.
pub const LATENT_BLOCK: i32 = 16;

/// A presentation under construction.
///
/// ```no_run
/// # use h3_hrx::{Presentation, Tokenizer, shape_for};
/// # fn main() -> h3_hrx::Result<()> {
/// let tokenizer = Tokenizer::new()?;
/// let shape = shape_for(480, 864, 124).unwrap();
/// let mut presentation = Presentation::new(&tokenizer);
/// presentation.picture(512, 288)?;
/// let ids = presentation.finish("a quiet room", &shape)?;
/// # Ok(())
/// # }
/// ```
pub struct Presentation<'a> {
    tokenizer: &'a Tokenizer,
    ids: Vec<i32>,
    pictures: usize,
    audios: usize,
}

impl<'a> Presentation<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            pictures: 0,
            audios: 0,
        }
    }

    /// Announce an image -- a keyframe or a reference -- as the next `<Picture n>`.
    ///
    /// Keyframes and references share one numbering, in the order they are
    /// announced. The dimensions are the scaled ones the encoder will see, not
    /// the original file's; [`crate::resize::fit`] produces them.
    pub fn picture(&mut self, width: i32, height: i32) -> Result<&mut Self> {
        if width <= 0 || height <= 0 || width % VISION_BLOCK != 0 || height % VISION_BLOCK != 0 {
            return Err(Error::Invalid(format!(
                "a picture is a positive multiple of {VISION_BLOCK} in each direction, not {width}x{height}"
            )));
        }

        self.pictures += 1;
        let label = format!("<Picture {}>: ", self.pictures);
        self.tokenizer.encode_into(&label, &mut self.ids)?;

        self.ids.push(VISION_START);
        self.ids.extend(std::iter::repeat_n(
            -1,
            (height / VISION_BLOCK) as usize * (width / VISION_BLOCK) as usize,
        ));
        self.ids.push(VISION_END);

        Ok(self)
    }

    /// Announce a soundtrack as the next `<Audio n>`.
    ///
    /// Audio carries no span: the samples reach the model as a reference rather
    /// than through the text encoder, and this only names it.
    pub fn audio(&mut self) -> Result<&mut Self> {
        self.audios += 1;
        let label = format!("<Audio {}>: ", self.audios);
        self.tokenizer.encode_into(&label, &mut self.ids)?;
        Ok(self)
    }

    /// The tokens announced so far, before the prompt.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Append the prompt and check it against the request's budget.
    ///
    /// Refused here rather than in the sampler: the sequence is built before
    /// any weights are mapped, so a presentation that cannot fit costs nothing
    /// to discover.
    pub fn finish(mut self, prompt: &str, shape: &Shape) -> Result<Vec<i32>> {
        self.tokenizer.encode_into(prompt, &mut self.ids)?;

        if self.ids.len() > shape.text_rows_max as usize {
            return Err(Error::Invalid(format!(
                "the presentation is {} tokens and the model takes at most {}; \
                 fewer or smaller references, or a shorter prompt",
                self.ids.len(),
                shape.text_rows_max
            )));
        }

        Ok(self.ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        crate::shape_for(480, 864, 124).expect("a canvas the model lays out")
    }

    #[test]
    fn a_picture_spans_one_token_to_a_block() {
        let tokenizer = Tokenizer::new().expect("the compiled vocabulary");
        let mut presentation = Presentation::new(&tokenizer);
        let label = presentation.len();

        presentation.picture(64, 32).expect("a 32-aligned picture");

        // Two blocks across and one down, plus the two sentinels, plus whatever
        // "<Picture 1>: " tokenized to.
        let blocks = ((64 / VISION_BLOCK) * (32 / VISION_BLOCK)) as usize;
        assert_eq!(
            presentation.len() - label,
            blocks + 2 + label_tokens(&tokenizer, 1)
        );
    }

    fn label_tokens(tokenizer: &Tokenizer, n: usize) -> usize {
        let mut ids = Vec::new();
        tokenizer
            .encode_into(&format!("<Picture {n}>: "), &mut ids)
            .expect("the label tokenizes");
        ids.len()
    }

    #[test]
    fn pictures_and_audio_are_numbered_in_the_order_they_are_announced() {
        let tokenizer = Tokenizer::new().expect("the compiled vocabulary");
        let mut presentation = Presentation::new(&tokenizer);

        presentation.picture(32, 32).unwrap();
        presentation.picture(32, 32).unwrap();
        presentation.audio().unwrap();

        // The numbering is what a prompt refers back to, so it has to follow
        // the announcements and not the kinds.
        let ids = presentation.finish("", &shape()).expect("it fits");
        let mut expected = Vec::new();
        for n in 1..=2 {
            tokenizer
                .encode_into(&format!("<Picture {n}>: "), &mut expected)
                .unwrap();
            expected.push(VISION_START);
            expected.push(-1);
            expected.push(VISION_END);
        }
        tokenizer.encode_into("<Audio 1>: ", &mut expected).unwrap();

        assert_eq!(ids, expected);
    }

    #[test]
    fn a_picture_off_the_grid_is_refused() {
        let tokenizer = Tokenizer::new().expect("the compiled vocabulary");
        let mut presentation = Presentation::new(&tokenizer);

        assert!(presentation.picture(33, 32).is_err());
        assert!(presentation.picture(32, 0).is_err());
    }

    #[test]
    fn a_presentation_past_the_budget_is_refused_before_anything_loads() {
        let tokenizer = Tokenizer::new().expect("the compiled vocabulary");
        let shape = shape();
        let mut presentation = Presentation::new(&tokenizer);

        // 4096 tokens of span on its own, before the label or the prompt.
        let side = VISION_BLOCK * 64;
        presentation.picture(side, side).unwrap();

        let error = presentation
            .finish("a quiet room", &shape)
            .expect_err("the budget is the limit a caller actually meets");

        assert!(format!("{error}").contains("at most 4096"));
    }
}
