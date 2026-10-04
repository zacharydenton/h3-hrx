# H3-World action-controlled generation

`h3 world` runs the [H3-World method](https://arxiv.org/html/2609.01560v1) in
Rust/Loom using the released rank-32 adapter and the existing FL2VA base model.
It generates a complete fixed-length clip from an initial image, a static scene
description, and a recorded action schedule. `h3 world-session` adds continued
interaction: generate a segment, observe its result, choose the next controls,
and resume from the last decoded frame.

The paper itself uses fixed-horizon, bidirectional generation and identifies
persistent world state and real-time interaction as future work (section 5).
Our continuation layer is an application extension of that released model.
It retains the current RGB observation, not a recurrent hidden state, 3D scene,
physics engine, motion history, or memory of objects outside the view. Continuity
can drift across segments; each step still requires full diffusion inference.

```sh
h3 world --first-frame garage.png \
  -p "A man in a yellow floral shirt stands in a dim concrete parking garage." \
  --action-preset forward --seed 2 --out forward.mp4
```

Defaults are 832×480, 124 frames at 24 fps, Euler with 50 evaluations, video/audio
shifts 12/3, and no CFG amplification. As elsewhere in this CLI, `--steps` counts
sigma grid points: use `--steps 51` for 50 evaluations. World mode selects FP16
attention and stage-scoped residency by default. Existing memory-budget options
apply. An output `.world.json` records the scene, action clauses, frame key bits,
seed, sampling settings, source revisions, adapter path and SHA-256.

The pinned adapter downloads through the standard Hugging Face cache. Use
`--world-adapter PATH` for a local copy or `--offline` to require cached files.
Use FL2VA weights for any explicit `--dit` override. Exactly one complete
104-pair rank-32 world adapter is required. Additional references, last-frame
conditioning, Turbo, custom adapter composition, and step caching are rejected.
The adapter remains subject to the MiniMax H3 Community License.

## Controls

Presets: `still`, `forward`, `back`, `strafe-left`, `strafe-right`, `tilt-up`,
`tilt-down`, `pan-left`, `pan-right`, `pan-left-fast`, `pan-right-fast`.

For changing controls, pass `--actions schedule.json` instead of a preset.
The file is a JSON array with exactly one held-key array per output frame:

```json
[["W"], ["W", "L"], ["W", "L"], ["W", "L"], []]
```

That example is a five-frame clip, so pass `--frames 5`. Supported frame counts
are `17k+5`, including 5, 22 and 124. `W/A/S/D` move the character; `J/L` pan
left/right; `I` tilts down and `K` tilts up, matching the training recordings.
`F` changes pan wording from slowly to sharply. Combine keys for diagonal
movement or simultaneous character/camera control. Opposing keys cancel after
OR-pooling each native latent interval (repeating widths 1,4,4,4,4 frames).

The learned clauses use “the man”, exactly as upstream. Control generalization
to other subjects and scenes is empirical. `--generate-prompt` can prepare the
static scene description through the existing optional endpoint using a dedicated
scene-only template; it never receives or rewrites the action clauses.

## Continued world sessions

Start a smaller interactive session on a shared machine:

```sh
h3 world-session --first-frame garage.png \
  -p "A man in a yellow floral shirt in a dim concrete parking garage." \
  --state garage.h3world --out garage-rollout \
  --width 320 --height 192 --frames 22 --seed 2 --interactive
```

At `world>`, enter a preset such as `forward` or `pan-right`, combined held keys
such as `W J`, or a complete per-frame JSON array. Each command generates one
segment before accepting the next command. `status` reports the cursor and next
seed; `quit` or EOF ends the session. Controls are also readable from piped stdin.
The default sampler uses 50 evaluations, so this is turn-by-turn interaction,
not real-time keyboard control. Without shape overrides, paper defaults apply.

To prepare the initial description through the optional endpoint, add
`--generate-prompt --prompt-images` and configure `--prompt-base-url` and
`--prompt-model` (or `H3_PROMPT_BASE_URL` / `H3_PROMPT_MODEL`). Authentication uses
`H3_PROMPT_API_KEY`. This makes one scene-only request with the initial picture;
controls never pass through the LLM. The command reserves text space for future
action combinations, stores provenance in `scene-prompt.json`, and saves the
final scene in the world checkpoint. Resume reuses that scene without another
endpoint call. The feature is disabled unless explicitly requested.

Continue later, or generate just one next segment:

```sh
h3 world-session --resume garage.h3world --out garage-rollout --interactive
h3 world-session --resume garage.h3world --out garage-rollout --action-preset forward
```

Branch an existing observation into another future without changing its source:

```sh
h3 world-session --resume garage.h3world --state alternate.h3world \
  --out alternate-rollout --action-preset pan-left
```

`--actions schedule.json` accepts the same per-frame schedule as `h3 world`.
Resume restores the scene, geometry, duration, sampling settings and next seed;
those settings cannot be overridden on that command. Use the same base model
checkpoints when continuing. The saved adapter SHA-256 is checked on resume.
`--offline` and explicit `--dit`, `--te`, `--video-vae`, `--audio-vae` and
`--world-adapter` paths work as in the single-segment command.

Each `segment-000000/` directory contains `video.mp4`, `audio.wav`, a lossless
`observation.png`, `actions.json` with inputs/provenance, and `state.h3world`.
The top-level `--state` is atomically replaced after those outputs succeed.
Generation, decoding and output failures leave the previous saved cursor intact.
If only the final cursor update fails, resume the completed segment's own
`state.h3world`. Existing segment directories are never deliberately overwritten;
use a new output directory for a branch.

The initial checkpoint is saved before the first generation. Advisory locks
beside the active state and in the output directory prevent concurrent writers.
The OS releases those locks even after a forcibly killed process; their small
`.lock` files remain in place and should not be deleted while a session runs.

Frame zero in each segment shares the previous observation's timestamp. The
global cursor therefore advances by `frames - 1`, and the seed advances by one
(wrapping at `u64::MAX`). When assembling a video trajectory, omit frame zero
from later segments. Reconstructed boundary frames need not match pixel-for-pixel;
the lossless final decoded RGB frame, rather than an MP4 frame, becomes the next
input. Audio is generated independently for each segment, with no audio history.

The CLI uses stage-scoped residency and a **28 GiB native allocation cap** by
default. `--memory-budget-mib` changes that ceiling. Host images, decoded clips,
ffmpeg and other processes require additional RAM. Only the current observation
and one output segment remain in memory; previous outputs live on disk. The Rust
rollout API also rejects decoded video above 512 MiB per segment and observations
above 64 MiB before allocating them.

## Rust API and routing

Build an `ActionSchedule` from frame states, JSON, or a preset, then create a
`WorldRequest` with `WorldRequest::new`. `WorldRequest::parameters()` supplies the
world defaults. Configure one `adapter::Lora` at strength 1 in `Config`, create a
`PreparedPresentation` containing one first-frame picture and the scene text,
encode that image with `Session::encode_video`, and call
`Session::denoise_world` with the corresponding `Keyframe`. Use
`SessionOptions { residency: ResidencyPolicy::StageScoped, ..Default::default() }`
on shared machines, and a budgeted `ModelContext` when using `Session::new_in`.
Decode video and audio through the ordinary session methods.

For continued interaction, `WorldState` owns the current RGB8 observation and
cursor. `Session::step_world` performs presentation, image encoding, action
conditioning, denoising and video/audio decoding, then advances the state only
after success. Cancellation or failure leaves it unchanged. `WorldState::clone`
branches the future, while `save`/`load` persist it without loading a model:

```rust,no_run
use h3_hrx::{ActionSchedule, Session, Tokenizer, WorldState};

fn advance(session: &mut Session) -> h3_hrx::Result<()> {
    let tokenizer = Tokenizer::new()?;
    let mut state = WorldState::load("garage.h3world")?;
    let actions = ActionSchedule::preset("forward", state.parameters().frames as usize)?;
    let segment = session.step_world(&mut state, &tokenizer, &actions, None)?;
    state.save("garage.h3world")?;
    // Consume segment.video (RGB8) and segment.audio (planar stereo, 32 kHz).
    Ok(())
}
```

Each distinct action sentence is encoded and refined independently, then reused
for repeated intervals. Static text/vision is refined separately. Mirrored
positions align action spans with video intervals. A dedicated native attention
kernel allows each action span to be read only by itself and its matched video
interval. Action queries can read the static context, first frame, audio, and
matched video, but not other actions or unmatched video. Video-to-video attention
remains bidirectional. No dense attention mask is allocated.

## Validation

CPU fixtures enumerate all keyboard states, check action tokens, temporal
positions, and cover resizing. The GPU kernel test checks directed routing
against an independent scaled-dot-product oracle, including ragged rows and
changed schedules. The integration test changes actions and returns to the
original schedule in one stage-scoped session:

```sh
scripts/test.sh --cpu
cargo test --test kernels world_attention_matches_directed_cpu_oracle -- --ignored
cargo test --test world world_changes_actions_and_reuses_a_session -- --ignored
```

The integration test needs cached base models and the pinned adapter. Numerical kernel agreement and action-sensitive latents do not establish
paper-level visual quality or exact parity with the authors' BF16 CUDA pipeline;
the base weights here are quantized. For visual qualification, hold the image,
seed and sampler fixed and compare still/forward, opposing pans, slow/fast pans,
and a reversal after latent interval 15. Preserve the `.world.json` records.

On a shared machine, inspect available RAM before starting. For example,
`--memory-budget-mib 28672` caps this session's native allocations at 28 GiB;
leave additional space for host staging and other processes. The integration
test uses the same native allocation ceiling and runs its requests serially.

### Observed validation, 2026-10-03

On Linux/gfx1151 (AMD Strix Halo, 128 GiB shared RAM), HRX 0.8.7 and
Rust 1.95.0-nightly, the CPU suite passed 204 tests including doctests. The
native masked-attention oracle passed both launch sizes. The full-model
64×64, five-frame, one-evaluation left/right/left regression passed, including
exact repeatability, in 848 seconds with stage-scoped residency.

Two visual smoke runs used the authors' `examples/first_frame.png`, seed 2,
320×192, 22 frames, 50 Euler evaluations, default 12/3 shifts, the quantized
FL2VA base, and the released adapter. Both produced video with sound:

| Control | Wall time | Minimum system RAM available | Cumulative background flow x |
| --- | ---: | ---: | ---: |
| pan-left | 528 s | 11.15 GiB | +55.66 px |
| pan-right | 528 s | 11.47 GiB | −17.37 px |

These runs were serial with a 28 GiB native allocation ceiling. The garage
remained visually coherent and the opposite flow signs matched the requested
camera directions; pan magnitudes were asymmetric. Flow used OpenCV 5 Farneback
on successive grayscale frames (`0.5,3,15,3,5,1.2,0`), averaging the upper
background crop `[12:85,12:308]`, then summing across transitions. Positive
background motion corresponds to a left camera pan.

An 832×480, 124-frame run successfully loaded and sampled under the same cap,
with at least 10.26 GiB available, but was interrupted after the initial
144-second evaluation because of runtime on the shared GPU. It did not produce
a completed clip. Full-size quality, still/forward, speed changes,
and upstream BF16 numerical parity remain unqualified. The short opposing-pan
checks establish native action response, not reproduction of the paper's results.

Local smoke artifacts and provenance are under `target/world-smoke/`; they are
build artifacts, not checked-in media. Reproduce the small run by adding
`--width 320 --height 192 --frames 22 --memory-budget-mib 28672` to the example
above, using `pan-left` and `pan-right` with seed 2.

### Continued-session validation, 2026-10-03

The extended CPU suite passes 210 tests, including state persistence, branching,
failure rollback, writer locking and optional endpoint scene preparation. Builds without default features, both with
and without the CLI, also pass.

A real `world-session` run generated a 320×192, 22-frame left pan, exited, then
resumed in a new process and generated a right pan from its saved observation.
Both used 50 Euler evaluations and the same scene description; the seeds were
2 then 3. The scene and character remained visually consistent across the handoff.

| Segment | Wall time | Minimum system RAM available | Cumulative background flow x |
| --- | ---: | ---: | ---: |
| Initial left pan | 580 s | 11.33 GiB | +55.56 px |
| Resumed right pan | 586 s | 9.74 GiB | −54.81 px |

Both ran serially with a 28 GiB native ceiling. A monitor would stop only our
process if available system RAM stayed below 6 GiB; no stop was triggered. The
second segment's recorded input SHA-256 matches the first segment's checkpoint.
The saved observations match their lossless PNGs exactly. The final cursor is
frame 42 with next seed 4; the checkpoint is about 185 KB. The resumed clip's
first compressed video frame differs from its input observation by a mean of
4.82 RGB8 levels, so the model handoff is not pixel-exact. Flow used the same
settings and crop described above.

Artifacts, contact sheets, provenance and resource records are in
`target/world-rollout-smoke/`. This checks continued action response and
save/resume behavior over two segments, not persistent off-screen state or
long-horizon consistency. The separate ignored `world_rollout_decodes_and_resumes`
regression uses a 64×64, five-frame case and is included in `scripts/test.sh --full`.

### Temporal reversal validation, 2026-10-03

A 320×192, 124-frame clip used the paper's 37 action intervals: left-fast for
the first 15 intervals (RGB frames 0–50), then right-fast for the remaining 22
(frames 51–123). With seed 2 and 50 Euler evaluations, cumulative background
flow was **+85.27 px before the switch and −60.76 px afterward**, using the
same Farneback settings and background crop. The transition into frame 51 is
included in the second sum. The requested reversal is visible in the video.

The run completed in 630 seconds with a 28 GiB native ceiling and at least
10.17 GiB of system RAM available. An earlier 26 GiB attempt was rejected by
the allocation budget during conditioning; it did not sample a clip. The
successful retry began after available RAM recovered, with the 6 GiB guard
unchanged. Artifacts and both resource records are in
`target/world-temporal-smoke/`.

This demonstrates within-segment action timing on the native path. The reduced
resolution clip also shows subject ghosting and lighting artifacts. It is not
a visual-quality reproduction of the authors' 832×480 BF16 results, and does not
remove the full-size or numerical-parity qualification limits above.
