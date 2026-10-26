//! Cooperative checkpoints for Session's single private stream.
use hrx::execution::NativeCooperator;
use std::cell::RefCell;

#[derive(Clone)]
struct Active {
    stream: usize,
    cooperator: NativeCooperator,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

// The owned handle expires in HRX when the stage exits. Restore the previous
// scope even when a progress callback or model operation panics.
pub(crate) struct Scope(Option<Active>);
impl Scope {
    pub(crate) fn enter(stream: &hrx::Stream, cooperator: &NativeCooperator) -> Self {
        Self(ACTIVE.with(|slot| {
            slot.replace(Some(Active {
                stream: stream.id(),
                cooperator: cooperator.clone(),
            }))
        }))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| slot.replace(self.0.take()));
    }
}

fn checkpoint(stream: &mut hrx::Stream, boundary: bool) -> hrx::Result<()> {
    let active = ACTIVE.with(|slot| slot.borrow().clone());
    if let Some(active) = active {
        if active.stream != stream.id() {
            return Err(hrx::Error::Message(
                "H3 stage checkpoint used a different stream".into(),
            ));
        }
        // SAFETY: Session's state owns one stream; every model uses that stream.
        // Scope checks its identity, and synchronize drains every prior use.
        unsafe {
            if boundary {
                active.cooperator.checkpoint(|| stream.synchronize())?;
            } else {
                active
                    .cooperator
                    .yield_if_waiting(|| stream.synchronize())?;
            }
        }
    }
    Ok(())
}

/// Bound queued work to one transformer block, even when no job is waiting yet.
pub(crate) fn boundary(stream: &mut hrx::Stream) -> hrx::Result<()> {
    checkpoint(stream, true)
}

/// Eager kernels and lazy loading can hand off sooner when another job arrives.
pub(crate) fn yield_if_waiting(stream: &mut hrx::Stream) -> hrx::Result<()> {
    checkpoint(stream, false)
}
