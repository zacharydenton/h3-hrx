//! The recipe tables for each checkpoint: which tensor feeds which kernel operand, at what pitch.
//!
//! A plan validates every source tensor's dtype and shape as it goes, so a checkpoint that does not
//! match the build fails when it is opened, naming the tensor, rather than part-way through a run.
pub mod avae;
pub mod dit;
pub mod te;
pub mod vvae;
