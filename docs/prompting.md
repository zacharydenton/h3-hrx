# Writing prompts for H3

H3 was trained on structured prompts, not free-form sentences. MiniMax publishes the format it expects as a
skill in the model repository, [`skills/h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing);
`references/base-en.txt` covers the text and keyframe modes and `references/ref-en.txt` the full-reference one.
This page is the short version plus the prompt behind the clip at the top of the README. Without `--generate-prompt`, the prompt is passed to the text encoder verbatim,
on stdin or after `-p`. Optional endpoint-based rewriting validates the generated format.

## Optional prompt generation

The default CLI includes a custom processor that follows MiniMax's published
[base](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
guides. It is not the proprietary MiniMax Context-IR service. It first analyzes
reference evidence, then rewrites your instruction into the appropriate H3 format.

Configure an OpenAI-compatible Chat Completions endpoint. The URL includes its
API prefix; the client appends `/chat/completions`:

```sh
export H3_PROMPT_BASE_URL=http://localhost:8000/v1
export H3_PROMPT_MODEL=your-multimodal-model
# For an authenticated endpoint, also set H3_PROMPT_API_KEY.
h3 prompt -p 'A red fox crosses a snowy clearing; no music' > fox.txt
h3 --generate-prompt -p 'A red fox crosses a snowy clearing; no music' \
  --save-prompt fox.txt --out fox.mp4
```

`h3 prompt` shares generation's input and shape arguments, writes only prompt
text to stdout, and does not run video generation. `--prompt-base-url` and
`--prompt-model` override the environment. `--save-prompt FILE` writes the final
text plus `FILE.json` with model, template version, effective duration, reference
mapping, and validation status. Reuse the text without `--generate-prompt` to
avoid another rewrite. Keep the same reference ordering and presentation options.

For media, explicitly declare the endpoint's supported input types:

```sh
h3 prompt --prompt-images --first-frame frame.png \
  -p 'The subject slowly turns toward the camera' > turn.txt
h3 prompt --prompt-images --prompt-audio person.png voice.wav \
  -p 'Use Picture 1 for appearance and Audio 1 for the voice' > character.txt
h3 prompt --prompt-images --prompt-audio movement.mp4 --video-audio 1 \
  -p 'Continue this scene with its original atmosphere' > continuation.txt
h3 prompt --prompt-images --prompt-audio --refmod character.safetensors \
  -p 'The referenced character greets the viewer' > greeting.txt
```

Images use `image_url` PNG data URLs. Video uses timestamped images sampled at
2 fps, preserving its timeline instead of relying on a provider-specific video
field. Audio uses `input_audio` with WAV data. A video soundtrack is included only
when selected with `--video-audio INDEX` (one-based among positional videos).
The endpoint must support these payloads; generic chat compatibility alone does
not imply image or audio support. Unsupported media fails explicitly.

Enabling rewriting sends your instruction and supplied/reconstructed media to
the configured endpoint. Ordinary generation makes no LLM request. `--offline`
continues to mean **no checkpoint downloads**; it does not disable the explicitly
selected endpoint. Prompt-only raw-media requests need no H3 checkpoints or GPU.
RefMods require the corresponding local VAE and GPU for reconstruction; see
[RefMod presentation](refmods.md#upstream-presentation).

The processor uses the actual rounded frame count at 24 fps, not an integer API
duration. Validation checks section order, available media labels, cut timing,
quoted literal text and the complete H3 token budget, including visual spans.
One corrective rewrite is allowed. Endpoint failures or an invalid final prompt
stop before denoising, without silently truncating references or reverting to the
original instruction. Long video presentations can exhaust the 4096-token budget:
use shorter/smaller references or a smaller canvas, then rerun.

The endpoint has a 300-second request timeout and no automatic HTTP retries.
Generation uses two successful endpoint requests, or three when repair is needed.
There is no built-in model download, model choice, or claim of official Context-IR
quality. Assess instruction preservation and video quality with your chosen model.

### Fully local shared-Qwen backend

This backend is experimental. Native text decoding and basic image recognition
have been exercised; full prompt/RefMod/World qualification is still pending
because concurrent GPU workloads exhausted available RAM during testing. See
[validation status](testing.md#shared-qwen-local-prompting).

Build with `cargo build --release --features local-prompt-generation`, then select
`--prompt-backend local`. The endpoint remains the default backend. Local prompting
makes no chat API requests; model provisioning can contact Hugging Face unless
`--offline` is set.

```sh
h3 prompt --prompt-backend local --refmod character.safetensors \
  -p 'The referenced character walks through a forest' > forest.txt
h3 --generate-prompt --prompt-backend local --refmod character-with-voice.safetensors \
  --prompt-audio-note '1=Use this reference for the character voice; say Hello.' \
  -p 'The character greets the viewer' --save-prompt greeting.txt --out greeting.mp4
h3 world --generate-prompt --prompt-backend local --first-frame scene.png \
  --action-preset pan-left -p 'Describe this room' --out room.mp4
```

This backend shares H3's existing INT8 Qwen3-VL-32B encoder and vision weights,
then adds BF16 language layers 50–63, final normalization and the output head.
The continuation comes from `Qwen/Qwen3-VL-32B-Instruct` revision
`0cfaf48183f594c314753d30a4c4974bc75f3ccb`; only shards 11–14 are downloaded.
`--prompt-local-model-dir DIR` uses those same four files locally, with their
pinned SHA-256 identities checked. This is not the Qwen3.6 checkpoint used by
qwen-hrx. The embedded H3 tokenizer is required.

Visual analysis and rewriting run locally with greedy decoding and at most one
corrective rewrite. Reference labels, strengths and synthetic stack timing are
preserved. Audio is **not listened to**: provide a nonempty `--prompt-audio-note
INDEX=TEXT` for every active Audio label, including copies. Notes describe intended
use and are recorded as user statements, not observed evidence. Audio RefMod
latents still condition H3 normally. Missing notes fail before reconstruction.

The resident model has a default **48 GiB native allocation ceiling** and needs
additional system headroom. The loader estimates weights, KV storage and workspace
before uploading, and requires another **8 GiB of available RAM**. It fails if the
model does not fit; there is no offloading or endpoint fallback. Override the
ceiling with `--memory-budget-mib`. Host media has its separate
`--refmod-media-budget-mib` limit.
Decoding rechecks available RAM between tokens and stops below the 8 GiB reserve.
Failed completions release both the generation weights and shared encoder. Other processes
can still allocate RAM between checks; the reserve is not a system-wide reservation.

`--prompt-context-tokens` defaults to 8192 and `--prompt-max-tokens` to 1024 per
completion. Oversized contexts and completions that hit the output limit fail
explicitly. The generation-only weights and KV caches are released after writing
the prompt; the shared encoder and bounded vision cache remain available for H3
conditioning. Raw-reference VAE encoding runs before local prompt generation.
`H3_STAGE_TRACE=1` reports prefill and decoding progress without printing content.
Saved prompt provenance includes model paths/revision, precision, audio notes,
token limits, validation and completion timings.

For Rust, use `local_prompt::{LocalPromptConfig, LocalPromptGenerator, AudioNote}`.
Construct the generator with `unsafe { LocalPromptGenerator::new(config) }`, keeping
its checkpoint files immutable while used by a session. Call `generate`,
`generate_refmods`, or `generate_world_scene` with `&mut Session`; the RefMod result
has the same `prompt` and `presentation` fields as the endpoint path. The session
must use a live `ResidencyManager` allocation budget no larger than the configured
local ceiling. Keep the manager alive and prefer stage-scoped residency. Set the
config's optional atomic cancellation flag to stop between language blocks/tokens.

### Native Rust RefMod prompting

With the `prompt-generation` feature, `PromptGenerator::generate_refmods` accepts
prepared RefMods directly. It decodes their effective image/video/audio latents
through the local VAEs, calls the configured endpoint, and returns the final prompt
plus its matching `PreparedPresentation`. Original source files are unnecessary.
The CLI uses the same native reconstruction method, `Session::refmod_entries`.

```rust,no_run
use h3_hrx::{Config, DenoiseParams, ResidencyPolicy, Session, SessionOptions, shape_for};
use h3_hrx::prompt::{EndpointConfig, PromptGenerator, RefModPromptRequest};
use h3_hrx::refmod::{ApplyOptions, RefMod, RefModPresentationOptions};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let file = RefMod::load("character-with-voice.safetensors")?;
let mods = [file.prepare(ApplyOptions::default())?];
let mut endpoint = EndpointConfig::new(
    "http://localhost:8000/v1".into(), "your-multimodal-model".into(),
);
endpoint.images = true;
endpoint.audio = true;
let generator = PromptGenerator::new(endpoint)?;
let budget = hrx::residency::ResidencyManager::new(8 * 1024 * 1024 * 1024)?;
let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
    memory_budget: Some(budget.budget()),
    ..Default::default()
})?;
// Safety: checkpoint files remain unchanged while the session exists.
let mut session = unsafe { Session::new_in(Config::default(), SessionOptions {
    residency: ResidencyPolicy::StageScoped,
    ..Default::default()
}, &context) }?;
let params = DenoiseParams::default();
let shape = shape_for(params.height, params.width, params.frames).ok_or("invalid output shape")?;
let generated = generator.generate_refmods(&mut session, RefModPromptRequest {
    instruction: "The referenced character greets the viewer",
    entries: &[], // Optional raw media goes before RefMod members.
    refmods: &mods,
    shape: &shape,
    presentation: RefModPresentationOptions::default(),
})?;
println!("{}", generated.prompt.text);
let references: Vec<_> = mods.iter().flat_map(|m| m.references()).collect();
// In an inference session with enough memory for H3, pass generated.presentation
// and these same references to Session::denoise_presented.
# let _ = references;
# Ok(()) }
```

The 8 GiB model allocation cap above is for small-reference prompt preparation;
full H3 inference needs a larger budget. Reconstruction defaults to a conservative
1 GiB host float-media budget and synthetic 24 fps playback. Set
`RefModPresentationOptions::{max_media_bytes, fps}` explicitly to change those
limits (CLI: `--refmod-media-budget-mib` and `--reference-fps`). The host media budget excludes model weights, GPU scratch, encoded HTTP
payloads and the later vision presentation. Copies share decoded buffers; disabled
members need neither decoding nor endpoint media support. Capability and media
budget failures stop before reconstruction. Keep the prepared RefMods and their
ordering unchanged when using the returned presentation for inference.

## The three fields

Text-to-video-with-audio (T2VA) prompts are three labelled fields, in this order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- **`integrated_multimodal_description`** is the body: style, composition, subjects, actions, camera, and any
  dialogue or diegetic sound, developed along the timeline. Open `[Shot 1]` with the style
  (`Live-action, cinematic`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, `vintage film`).
- **`overall_soundscape`** is one paragraph, 1 to 4 sentences, of ambient and physical sound across the whole
  clip: wind, surf, footsteps, fabric, impacts, breathing. `N/A` only for deliberate silence.
- **`non_diegetic_music`** is the score the characters cannot hear, in 1 to 3 sentences of instrumentation,
  tempo, rhythm and dynamics. `N/A` when there is none.

The split between the last two is the easy one to get wrong: a swelling orchestral score belongs in
`non_diegetic_music`, never alongside the hooves and surf in `overall_soundscape`. Music a character can hear —
a radio, a busker, singing — is diegetic and goes in the description instead.

The keyframe modes prepend one instruction line, then a blank line, then the same three fields. `--first-frame`
is I2VA; a first and last image is FL2VA (the `fl2va` checkpoint); reference images and audio are Ref2VA (the
`ref2va` checkpoint, a different set of six sections — see `ref-en.txt`).

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

## Details that matter

- **Duration.** Describe 4 to 15 seconds, and match what you actually render: 124 frames at 24 fps is 5.17 s.
- **Cuts.** Number shots and give each later one a strictly increasing time inside the duration
  (`[Shot 2] At 00:03.500, the camera cuts to ...`). No timestamp on `[Shot 1]`. A cut should introduce new
  information; if only the distance or angle changes, move the camera instead. At 5 seconds a single shot is
  usually the better trade.
- **Camera.** Motion type, then amplitude and speed only when they matter, written as English rather than
  stacked labels: `Zoom In/Out`, `Push In`/`Pull Out`, `Pan`, `Truck`, `Tilt`, `Pedestal Up/Down`, `Arc Shot`,
  `Tracking Shot`, `Static Shot`, `Shake Slightly/Strongly`, `POV`, `Roll Clockwise/Counterclockwise`, with
  `with small/large amplitude` and `at slow/fast speed`.
- **Speakers.** Stable IDs `(S1)`, `(S2)`, `(S1,S2)` for anyone who vocalises, and nobody else. The spoken text
  goes inside `<d>[English] ...</d>` verbatim, with the delivery described outside it. Voiceover uses the exact
  phrase `says in an off-screen voiceover` and then states that the character's lips stay closed.
- **On-screen text** goes in double quotes, in its own language, unchanged: `a sign reading "营业中"`.
- **Concrete over abstract.** MiniMax's guidance is to prefer things that can be seen or heard over words like
  "beautiful" or "epic". `Cinematic` is the exception: it is one of the named styles and belongs in the opening
  clause of `[Shot 1]`.

## The README clip

The prompt is [cliff_rider_768p.txt](prompts/cliff_rider_768p.txt), and this reproduces the clip:

```sh
h3 --width 1344 --height 768 --frames 124 --steps 31 --seed 7 \
   --out cliff_rider.mp4 < docs/prompts/cliff_rider_768p.txt
```

![cliff rider](media/cliff_rider_768p_strip.jpg)

It is a single 5-second shot with no dialogue, so it carries no speaker IDs; the wind and hooves are diegetic
and the brass and timpani are not. Measured 2026-09-07 on the Radeon 8060S with `H3_PROFILE=1`: 101.8 s per
evaluation, steady across the last ten of the 30 evaluations, and 82.8 s to decode. That is one run, and the
decode figure has no second sample; [the decoder report](archive/vae-30s.md) has the decoder's own measurements and method.
