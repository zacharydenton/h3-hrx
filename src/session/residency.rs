//! Cache owners are session-local because native graphs bind their original stream.
//! A checked-out unit remains pinned until the session fences and returns it.
use hrx::residency::{ModelLease, ResidencyManager};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

pub(super) struct Unit<T: Send + 'static> {
    manager: ResidencyManager,
    key: String,
    active: Option<ModelLease<Mutex<Option<T>>>>,
}

impl<T: Send + 'static> Unit<T> {
    fn new(manager: &ResidencyManager, session: u64, name: &str) -> Self {
        Self {
            manager: manager.clone(),
            key: format!("h3:{session}:{name}"),
            active: None,
        }
    }

    pub fn checkout(&mut self, value: &mut Option<T>) -> hrx::Result<()> {
        if self.active.is_none() {
            let lease =
                self.manager
                    .load_budgeted(&self.key, |_| true, |_| Ok(Mutex::new(None)))?;
            *value = lease.lock().unwrap_or_else(|e| e.into_inner()).take();
            self.active = Some(lease);
        }
        Ok(())
    }

    /// Caller has drained every native use, including cancellation/error paths.
    pub fn checkin(&mut self, value: &mut Option<T>) {
        if let Some(lease) = self.active.take() {
            *lease.lock().unwrap_or_else(|e| e.into_inner()) = value.take();
        }
    }
}

impl<T: Send + 'static> Drop for Unit<T> {
    fn drop(&mut self) {
        self.active = None;
        // Keys never escape this session. Remove mappings before its checkpoint
        // immutability contract ends, even when the shared manager lives longer.
        let _ = self.manager.evict(&self.key);
    }
}

pub(super) struct Units {
    pub dit: Unit<crate::dit::Dit>,
    pub te: Unit<crate::te::TextEncoder>,
    pub video: Unit<crate::vvae::VideoVae>,
    pub audio: Unit<crate::avae::AudioVae>,
}

impl Units {
    pub fn new(manager: &ResidencyManager) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        Self {
            dit: Unit::new(manager, id, "dit"),
            te: Unit::new(manager, id, "text"),
            video: Unit::new(manager, id, "video"),
            audio: Unit::new(manager, id, "audio"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_units_reuse_then_evict_and_unregister_on_drop() {
        let manager = ResidencyManager::new(100).unwrap();
        let mut first = Unit::new(&manager, u64::MAX, "test-first");
        let mut second = Unit::new(&manager, u64::MAX, "test-second");
        let (mut a, mut b) = (None, None);
        first.checkout(&mut a).unwrap();
        a = Some(manager.budget().reserve(60).unwrap());
        second.checkout(&mut b).unwrap();
        assert!(manager.budget().reserve(50).is_err());
        first.checkin(&mut a);
        first.checkout(&mut a).unwrap();
        assert_eq!(a.as_ref().unwrap().bytes(), 60);
        first.checkin(&mut a);
        b = Some(manager.budget().reserve(50).unwrap());
        assert_eq!(manager.statistics().evictions, 1);
        first.checkout(&mut a).unwrap();
        assert!(a.is_none(), "evicted unit must reload");
        first.checkin(&mut a);
        second.checkin(&mut b);
        drop((first, second));
        assert_eq!(manager.statistics().resources, 0);
        assert_eq!(manager.statistics().reserved_bytes, 0);
    }
}
