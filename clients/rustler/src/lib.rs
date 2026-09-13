//! App-owned Rustler adapter, calling H3's Rust API directly. A worker owns each
//! model session so GC, inference and model destruction never occupy a normal
//! BEAM scheduler. Jobs contain owned data, never a borrowed NIF environment.
use rustler::{Atom, Binary, Encoder, Env, LocalPid, NifResult, OwnedEnv, ResourceArc};
use std::sync::mpsc::{self, SyncSender};
mod atoms {
    rustler::atoms! { ok, h3_result }
}
enum Job {
    Ping(LocalPid, u64),
    Encode(LocalPid, u64, Vec<f32>, usize),
}
struct Model {
    jobs: SyncSender<Job>,
}
#[rustler::resource_impl]
impl rustler::Resource for Model {}

/// Checkpoint files supplied by the application must remain unchanged for the
/// resource's lifetime, as required by the model's mmap-based Rust API.
#[rustler::nif(schedule = "DirtyIo")]
fn open(audio_checkpoint: Option<String>) -> NifResult<ResourceArc<Model>> {
    let config = h3_hrx::Config {
        dit: None,
        te: None,
        video_vae: None,
        audio_vae: audio_checkpoint.map(Into::into),
        ..Default::default()
    };
    // The application owns and keeps the configured checkpoint immutable.
    let mut session = unsafe { h3_hrx::Session::new(config) }
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    let (jobs, receiver) = mpsc::sync_channel(2);
    std::thread::Builder::new()
        .name("h3-model".into())
        .spawn(move || {
            while let Ok(job) = receiver.recv() {
                let (pid, id, result) = match job {
                    Job::Ping(pid, id) => (pid, id, Ok((Vec::new(), 0usize))),
                    Job::Encode(pid, id, samples, frames) => (
                        pid,
                        id,
                        session
                            .encode_audio(&samples, frames)
                            .map_err(|e| e.to_string()),
                    ),
                };
                let mut env = OwnedEnv::new();
                // A dead caller does not invalidate model state or keep a NIF borrow.
                let _ =
                    env.send_and_clear(&pid, |env| (atoms::h3_result(), id, result).encode(env));
            }
            // Last resource is gone: close the channel and drop the GPU session here.
        })
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(ResourceArc::new(Model { jobs }))
}
#[rustler::nif]
fn ping(env: Env<'_>, model: ResourceArc<Model>, id: u64) -> NifResult<Atom> {
    model
        .jobs
        .try_send(Job::Ping(env.pid(), id))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(atoms::ok())
}
#[rustler::nif(schedule = "DirtyCpu")]
fn encode_audio(
    env: Env<'_>,
    model: ResourceArc<Model>,
    samples: Binary<'_>,
    frames: usize,
    id: u64,
) -> NifResult<Atom> {
    if !samples.len().is_multiple_of(4) {
        return Err(rustler::Error::BadArg);
    }
    let samples = samples
        .as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    model
        .jobs
        .try_send(Job::Encode(env.pid(), id, samples, frames))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(atoms::ok())
}
rustler::init!("Elixir.H3.Native");
