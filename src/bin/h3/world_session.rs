//! Bounded-memory rollout driver. All inference lives in Session::step_world.
use super::{media, resize};
use anyhow::{bail, Context, Result};
use h3_hrx::{Action, ActionSchedule, Config, Session, Tokenizer, WorldRequest, WorldState};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub struct Args {
    /// Initial observation for a new world
    #[arg(long, required_unless_present = "resume", conflicts_with = "resume")]
    first_frame: Option<PathBuf>,
    /// Static scene description for a new world
    #[arg(
        short = 'p',
        required_unless_present = "resume",
        conflicts_with = "resume"
    )]
    prompt: Option<String>,
    /// Prepare the initial static scene through the optional multimodal endpoint
    #[arg(long, conflicts_with = "resume")]
    generate_prompt: bool,
    /// Chat API prefix (otherwise H3_PROMPT_BASE_URL)
    #[arg(long, requires = "generate_prompt")]
    prompt_base_url: Option<String>,
    /// Endpoint model (otherwise H3_PROMPT_MODEL)
    #[arg(long, requires = "generate_prompt")]
    prompt_model: Option<String>,
    /// Endpoint supports image_url inputs; permits sending the initial observation
    #[arg(long, requires = "generate_prompt")]
    prompt_images: bool,
    /// Checkpoint to resume (may also be branched with a different --state)
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Save after each completed step; defaults to --resume when continuing
    #[arg(long, required_unless_present = "resume")]
    state: Option<PathBuf>,
    /// Read controls between steps: presets, held keys, JSON frame arrays, status, quit
    #[arg(long)]
    interactive: bool,
    /// Generate one initial segment with this preset
    #[arg(long, conflicts_with = "actions", required_unless_present_any = ["actions", "interactive"])]
    action_preset: Option<String>,
    /// Generate one initial segment with a per-frame JSON schedule
    #[arg(long)]
    actions: Option<PathBuf>,
    /// Directory for numbered segment clips, action records and recovery states
    #[arg(long, default_value = "world-rollout")]
    out: PathBuf,
    /// Canvas width (default 832, multiple of 32)
    #[arg(long, conflicts_with = "resume")]
    width: Option<i32>,
    /// Canvas height (default 480, multiple of 32)
    #[arg(long, conflicts_with = "resume")]
    height: Option<i32>,
    /// Frames per segment (default 124; must be 17k+5)
    #[arg(long, conflicts_with = "resume")]
    frames: Option<i32>,
    /// Sigma grid points (51 gives 50 Euler evaluations)
    #[arg(long, conflicts_with = "resume")]
    steps: Option<usize>,
    /// Initial seed (default 0); increments after each completed segment
    #[arg(long, conflicts_with = "resume")]
    seed: Option<u64>,
    /// Native allocation ceiling; host buffers need additional RAM
    #[arg(long, default_value_t = 28672, value_parser = clap::value_parser!(u64).range(1..))]
    memory_budget_mib: u64,
    /// Use only model checkpoints already cached on disk
    #[arg(long)]
    offline: bool,
    /// Released adapter checkpoint (otherwise resolved through the Hub cache)
    #[arg(long)]
    world_adapter: Option<PathBuf>,
    /// FL2VA checkpoint override
    #[arg(long)]
    dit: Option<PathBuf>,
    /// Text encoder checkpoint override
    #[arg(long)]
    te: Option<PathBuf>,
    /// Video VAE checkpoint override
    #[arg(long)]
    video_vae: Option<PathBuf>,
    /// Audio VAE checkpoint override
    #[arg(long)]
    audio_vae: Option<PathBuf>,
}

enum Input {
    Step(ActionSchedule),
    Status,
    Quit,
}
fn control(line: &str, frames: usize) -> Result<Input> {
    let line = line.trim();
    if line == "quit" || line == "exit" {
        return Ok(Input::Quit);
    }
    if line == "status" || line.is_empty() {
        return Ok(Input::Status);
    }
    let actions = if line.starts_with('[') {
        ActionSchedule::from_json(line)?
    } else if let Ok(preset) = ActionSchedule::preset(line, frames) {
        preset
    } else {
        let keys: Vec<_> = line.split_whitespace().map(str::to_uppercase).collect();
        ActionSchedule(vec![Action::from_keys(&keys)?; frames])
    };
    actions.sentences(frames)?;
    Ok(Input::Step(actions))
}

// Advisory locks are released by the OS even if the process is killed. Keep
// the lock file's inode in place: deleting it would let a second writer lock
// a new inode while a previous writer still holds the old one.
struct StateLock {
    _file: std::fs::File,
}
impl StateLock {
    fn acquire(path: &Path) -> Result<Self> {
        let mut name = path.as_os_str().to_os_string();
        name.push(".lock");
        let path = PathBuf::from(name);
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        file.try_lock()
            .with_context(|| format!("{} is locked by another world session", path.display()))?;
        file.set_len(0)?;
        writeln!(file, "{}", std::process::id())?;
        Ok(Self { _file: file })
    }
}

fn status(state: &WorldState) {
    eprintln!(
        "world: {} segments, frame {} ({:.2}s), next seed {}",
        state.segments(),
        state.frame(),
        state.frame() as f64 / 24.0,
        state.parameters().seed
    );
}

pub fn run(args: Args) -> Result<()> {
    let save = args
        .state
        .as_ref()
        .or(args.resume.as_ref())
        .context("--state is required for a new world")?;
    let _lock = StateLock::acquire(save)?;
    if save.exists() && args.resume.as_ref() != Some(save) {
        bail!(
            "{} already exists; use --resume to continue it or choose a new --state",
            save.display()
        );
    }
    std::fs::create_dir_all(&args.out)?;
    let _output_lock = StateLock::acquire(&args.out.join("rollout"))?;
    let generator = super::prompting::endpoint_generator(
        args.generate_prompt,
        args.prompt_base_url.as_deref(),
        args.prompt_model.as_deref(),
        args.prompt_images,
        false,
    )?;
    let record_path = args.out.join("scene-prompt.json");
    if generator.is_some() && record_path.exists() {
        bail!(
            "{} already exists; choose a new --out directory",
            record_path.display()
        );
    }
    let mut generated_record = None;
    let mut state = if let Some(path) = &args.resume {
        WorldState::load(path)?
    } else {
        let mut p = WorldRequest::parameters();
        p.width = args.width.unwrap_or(p.width);
        p.height = args.height.unwrap_or(p.height);
        p.frames = args.frames.unwrap_or(p.frames);
        p.steps = args.steps.unwrap_or(p.steps);
        p.seed = args.seed.unwrap_or(p.seed);
        // Validate canvas and output bounds before resizing or allocating it.
        h3_hrx::world::validate_rollout_parameters(&p)?;
        let (rgb, w, h) =
            media::decode_image(args.first_frame.as_ref().context("missing first frame")?)?;
        let pixels: Vec<u8> = resize::world_first_frame(&rgb, w, h, p.width, p.height)
            .into_iter()
            .map(|v| (v * 255.).round() as u8)
            .collect();
        let instruction = args.prompt.as_deref().context("missing scene")?;
        let scene = if let Some(generator) = generator {
            let (text, record) = generate_scene(generator, instruction, &p, &pixels)?;
            generated_record = Some(record);
            text
        } else {
            instruction.into()
        };
        WorldState::new(scene, p, pixels)?
    };
    let initial = if let Some(path) = &args.actions {
        Some(ActionSchedule::from_json(&std::fs::read_to_string(path)?)?)
    } else {
        args.action_preset
            .as_deref()
            .map(|name| ActionSchedule::preset(name, state.parameters().frames as usize))
            .transpose()?
    };
    if let Some(actions) = &initial {
        actions.sentences(state.parameters().frames as usize)?;
    }
    // Preserve scene preparation before any heavyweight model resolution. A
    // failed download or device open can then be retried with --resume, without
    // another endpoint call. Adapter identity is bound before generation below.
    state.save(save)?;
    if let Some(record) = generated_record {
        let mut file = tempfile::NamedTempFile::new_in(&args.out)?;
        file.write_all(&serde_json::to_vec_pretty(&record)?)?;
        file.as_file().sync_all()?;
        file.persist_noclobber(&record_path).map_err(|e| e.error)?;
    }
    let resolver = h3_hrx::models::Resolver::new().offline(args.offline);
    let resolve = |path: &Option<PathBuf>, name: &str| -> Result<PathBuf> {
        Ok(match path {
            Some(p) => p.clone(),
            None => resolver.find(name)?,
        })
    };
    let adapter = match &args.world_adapter {
        Some(p) => p.clone(),
        None => h3_hrx::models::Resolver::new()
            .repository("DANNY621", "H3-World")
            .revision(Some(h3_hrx::world::ADAPTER_REVISION.into()))
            .offline(args.offline)
            .find(h3_hrx::world::ADAPTER_FILE)?,
    };
    // SAFETY: CLI-owned checkpoints are not mutated during inference.
    unsafe { h3_hrx::adapter::validate_world(&adapter) }?;
    let adapter_digest = hrx::bundle::file_digest(&adapter)?;
    state.bind_adapter(&adapter_digest)?;
    let config = Config {
        dit: Some(resolve(&args.dit, h3_hrx::models::DIT_FL2VA)?),
        te: Some(resolve(&args.te, h3_hrx::models::TE)?),
        video_vae: Some(resolve(&args.video_vae, h3_hrx::models::VIDEO_VAE)?),
        audio_vae: Some(resolve(&args.audio_vae, h3_hrx::models::AUDIO_VAE)?),
        loras: vec![h3_hrx::adapter::Lora::new(adapter, 1.)],
        attention: h3_hrx::Attention::F16,
        loom_library: std::env::var_os("HRX_LOOM_LIBRARY").map(PathBuf::from),
        ..Default::default()
    };
    let model_paths = serde_json::json!({ "dit": config.dit, "text_encoder": config.te,
        "video_vae": config.video_vae, "audio_vae": config.audio_vae,
        "adapter_sha256": adapter_digest });
    let bytes = usize::try_from(
        args.memory_budget_mib
            .checked_mul(1024 * 1024)
            .context("memory budget overflow")?,
    )?;
    let manager = hrx::residency::ResidencyManager::new(bytes)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    // SAFETY: this command never modifies the resolved checkpoint files.
    let mut session = unsafe {
        Session::new_in(
            config,
            h3_hrx::SessionOptions {
                residency: h3_hrx::ResidencyPolicy::StageScoped,
                ..Default::default()
            },
            &context,
        )
    }?;
    let tokenizer = Tokenizer::new()?;
    // Persist the bound adapter before an interrupted first step can be retried.
    state.save(save)?;
    let mut driver = Driver {
        session: &mut session,
        tokenizer: &tokenizer,
        state: &mut state,
        save,
        out: &args.out,
        models: &model_paths,
    };
    if let Some(actions) = initial {
        driver.step(actions)?;
    }
    if args.interactive {
        eprintln!("Controls: forward, pan-left, W J, per-frame JSON; status; quit. Each input generates a complete segment.");
        let stdin = std::io::stdin();
        driver.interactive(&mut stdin.lock())?;
    }
    Ok(())
}

fn generate_scene(
    generator: super::prompting::Generator,
    instruction: &str,
    p: &h3_hrx::DenoiseParams,
    pixels: &[u8],
) -> Result<(String, serde_json::Value)> {
    #[cfg(feature = "prompt-generation")]
    {
        use h3_hrx::media_context::{Frame, Media, MediaEntry};
        let tokenizer = Tokenizer::new()?;
        let mut shape = Session::shape_for(p.height, p.width, p.frames).context("invalid shape")?;
        // Future controls are not known yet. Leave room for the longest trained
        // action at every interval so a new command cannot exhaust the scene's budget.
        let mut max_tokens = 0;
        for bits in 0..512 {
            max_tokens = max_tokens.max(tokenizer.encode(&Action(bits).sentence())?.len());
        }
        shape.text_rows_max -= (max_tokens * shape.latent_t as usize) as i32;
        if shape.text_rows_max <= 0 {
            bail!("actions exhaust the text token budget");
        }
        let entries = [MediaEntry {
            media: Media::Picture(Frame {
                pixels: pixels
                    .iter()
                    .map(|b| *b as f32 / 255.)
                    .collect::<Vec<_>>()
                    .into(),
                width: p.width as usize,
                height: p.height as usize,
            }),
            role: "first_frame".into(),
            metadata: serde_json::Value::Null,
        }];
        eprintln!("prompt: preparing the initial static scene");
        let result = generator.generate_world_scene(h3_hrx::prompt::PromptRequest {
            instruction,
            entries: &entries,
            shape: &shape,
        })?;
        Ok((result.text, result.record))
    }
    #[cfg(not(feature = "prompt-generation"))]
    {
        let _ = (generator, instruction, p, pixels);
        bail!("rebuild h3 with the prompt-generation feature")
    }
}

struct Driver<'a> {
    session: &'a mut Session,
    tokenizer: &'a Tokenizer,
    state: &'a mut WorldState,
    save: &'a Path,
    out: &'a Path,
    models: &'a serde_json::Value,
}
impl Driver<'_> {
    fn step(&mut self, actions: ActionSchedule) -> Result<()> {
        let state = &mut *self.state;
        let save = self.save;
        let destination = self.out.join(format!("segment-{:06}", state.segments()));
        if destination.exists() {
            bail!(
                "{} already exists; resume its state.h3world or choose a new --out directory",
                destination.display()
            );
        }
        let staged = tempfile::Builder::new()
            .prefix(".world-step-")
            .tempdir_in(self.out)?;
        let mut next = state.clone();
        let p = state.parameters();
        let input_digest = hrx::bundle::file_digest(save)?;
        let mut progress = |done, total, _| {
            eprint!("\rsegment {}: {done}/{total} evaluations", state.segments());
            let _ = std::io::stderr().flush();
            false
        };
        let segment =
            self.session
                .step_world(&mut next, self.tokenizer, &actions, Some(&mut progress))?;
        eprintln!();
        let wav = staged.path().join("audio.wav");
        std::fs::write(
            &wav,
            media::wav_bytes(&segment.audio, segment.shape.audio_t as u32 * 800),
        )?;
        media::mux(
            &staged.path().join("video.mp4"),
            &wav,
            &segment.video,
            p.width,
            p.height,
        )?;
        media::write_still(
            &staged.path().join("observation.png"),
            next.observation(),
            p.width,
            p.height,
        )?;
        next.save(staged.path().join("state.h3world"))?;
        std::fs::write(
            staged.path().join("actions.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "upstream_revision": h3_hrx::world::UPSTREAM_REVISION, "mode": "observation_rollout",
                "input_state_sha256": input_digest, "models": self.models,
                "segment": segment.index, "start_frame": segment.start_frame, "end_frame": next.frame(),
                "seed": segment.seed, "scene_prompt": state.scene(), "width": p.width, "height": p.height,
                "frames": p.frames, "fps": 24, "evaluations": p.steps-1,
                "video_shift": p.video_shift, "audio_shift": p.audio_shift, "sampler": "euler", "attention": "f16",
                "frame_keys_bits": actions.0.iter().map(|a| a.0).collect::<Vec<_>>(),
                "action_sentences": segment.action_sentences,
                "shared_boundary_frames": 1
            }))?,
        )?;
        // Publish artifacts before advancing the top-level cursor. If saving
        // that cursor fails, the segment's own state is a recovery checkpoint.
        std::fs::rename(staged.path(), &destination)?;
        next.save(save).with_context(|| {
            format!(
                "segment saved; recover with --resume {}",
                destination.join("state.h3world").display()
            )
        })?;
        *state = next;
        println!("{}", destination.join("video.mp4").display());
        status(state);
        Ok(())
    }
    fn interactive(&mut self, input: &mut impl BufRead) -> Result<()> {
        let mut line = String::new();
        loop {
            eprint!("world> ");
            std::io::stderr().flush()?;
            line.clear();
            if input.read_line(&mut line)? == 0 {
                break;
            }
            match control(&line, self.state.parameters().frames as usize) {
                Ok(Input::Step(actions)) => self.step(actions)?,
                Ok(Input::Status) => status(self.state),
                Ok(Input::Quit) => break,
                Err(error) => eprintln!("{error}"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_accept_combinations_and_exact_frame_schedules() {
        let Input::Step(s) = control("w j", 5).unwrap() else {
            panic!()
        };
        assert_eq!(s.0, vec![Action::from_keys(&["W", "J"]).unwrap(); 5]);
        let Input::Step(s) = control("pan-right-fast", 5).unwrap() else {
            panic!()
        };
        assert!(s.sentences(5).unwrap()[0].contains("pans right sharply"));
        assert!(matches!(
            control("[[\"W\"],[],[],[],[]]", 5).unwrap(),
            Input::Step(_)
        ));
        assert!(control("[[\"W\"]]", 5).is_err());
        assert!(control("Q", 5).is_err());
        assert!(matches!(control("quit", 5).unwrap(), Input::Quit));
    }
    #[test]
    fn saved_cursor_cannot_have_two_writers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.state");
        let lock = StateLock::acquire(&path).unwrap();
        assert!(StateLock::acquire(&path).is_err());
        drop(lock);
        assert!(StateLock::acquire(&path).is_ok());
    }

    #[test]
    #[cfg(feature = "prompt-generation")]
    fn endpoint_scene_reserves_tokens_for_future_controls_and_survives_resume() {
        use std::{
            io::Read,
            net::TcpListener,
            time::{Duration, Instant},
        };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let start = Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(s) => break s,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            && start.elapsed() < Duration::from_secs(5) =>
                    {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(e) => panic!("endpoint accept: {e}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut header = Vec::new();
            let mut byte = [0];
            while !header.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                header.push(byte[0]);
            }
            let header = String::from_utf8(header).unwrap();
            let len = header
                .lines()
                .find_map(|l| {
                    l.to_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|n| n.parse::<usize>().unwrap())
                })
                .unwrap();
            let mut body = vec![0; len];
            stream.read_exact(&mut body).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let response = serde_json::json!({"choices":[{"finish_reason":"stop","message":{"content":"A man in a concrete garage."}}]}).to_string();
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).unwrap();
            request
        });
        let mut config = h3_hrx::prompt::EndpointConfig::new(url, "mock".into());
        config.images = true;
        config.timeout = Duration::from_secs(5);
        let generator = h3_hrx::prompt::PromptGenerator::new(config).unwrap();
        let p = h3_hrx::DenoiseParams {
            width: 64,
            height: 64,
            ..WorldRequest::parameters()
        };
        let pixels = vec![128; 64 * 64 * 3];
        let (text, record) =
            generate_scene(generator, "Describe this garage.", &p, &pixels).unwrap();
        let request = worker.join().unwrap();
        assert_eq!(record["template_version"], "h3-world-scene-v1");
        let supplied: serde_json::Value = serde_json::from_str(
            request["messages"][1]["content"][0]["text"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert!(supplied["h3_prompt_token_budget"].as_u64().unwrap() < 4096 - 500);
        assert_eq!(request["messages"][1]["content"][1]["type"], "image_url");
        let state = WorldState::new(text, p, pixels).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.h3world");
        state.save(&path).unwrap();
        assert_eq!(
            WorldState::load(path).unwrap().scene(),
            "A man in a concrete garage."
        );
        assert!(request["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("only the static"));
    }
}
