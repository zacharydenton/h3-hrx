# Generate structured prompts with an endpoint

Turn a short instruction and reference media into a structured H3 prompt.
Configure a multimodal endpoint, review the saved prompt, then reuse it for
repeatable renders.

## Configure and run

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

## Reference media

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
h3 prompt --prompt-images --refmod portrait.safetensors \
  --refmod-source '1:1=original portrait.png' \
  -p 'The referenced character greets the viewer' > greeting.txt
```

Images use `image_url` PNG data URLs. Video uses timestamped images sampled at
2 fps, preserving its timeline instead of relying on a provider-specific video
field. Audio uses `input_audio` with WAV data. A video soundtrack is included only
when selected with `--video-audio INDEX` (one-based among positional videos).
The endpoint must support these payloads; generic chat compatibility alone does
not imply image or audio support. Unsupported media fails explicitly.

## Requests and validation

Enabling rewriting sends your instruction and supplied/reconstructed media to
the configured endpoint. Ordinary generation makes no LLM request. `--offline`
continues to mean **no checkpoint downloads**; it does not disable the explicitly
selected endpoint. Prompt-only raw-media requests need no H3 checkpoints or GPU.
RefMod members without `--refmod-source SLOT:MEMBER=PATH` require the corresponding
local VAE and GPU for reconstruction. Supplying originals for every active member
also makes endpoint RefMod prompting independent of H3 checkpoints and GPU; see
[RefMod presentation](refmods.md#upstream-presentation).

The processor uses the actual rounded frame count at 24 fps, not an integer API
duration. Validation checks section order, available media labels, cut timing,
quoted literal text and the complete H3 token budget, including visual spans.
One corrective rewrite is allowed. Endpoint failures or an invalid final prompt
stop before denoising, without silently truncating references or reverting to the
original instruction. Long video presentations can exhaust the 4096-token budget:
use shorter/smaller references or a smaller canvas, then rerun.

Requests have a 300-second timeout and no automatic HTTP retries. Prompt generation
uses two endpoint calls, or three if format repair is needed. Evaluate instruction
preservation and output quality with your chosen model.

## Native Rust RefMod prompting

Prompt generation uses the configured endpoint. With originals, call
`PromptGenerator::generate_refmods_with_sources(None, request, &sources)`.
This returns the prompt and its matching H3 presentation without opening a
session, loading model weights, or initializing a GPU. The application decodes
its source files into `Media`; each `RefModSource` selects a one-based RefMod slot
and original member number. For example, for a single image member:

```rust,no_run
use h3_hrx::{media_context::{Frame, Media}, Shape};
use h3_hrx::prompt::{PromptGenerator, RefModPromptRequest, RefModPromptResult, PromptError};
use h3_hrx::refmod::{ApplyOptions, RefMod, RefModSource};

fn prompt_from_original(
    generator: &PromptGenerator, // Configure EndpointConfig.images = true.
    refmod: &RefMod,
    original: Frame, // Decoded RGB floats, dimensions aligned to 32 pixels.
    shape: &Shape,
) -> Result<RefModPromptResult, PromptError> {
    let mods = [refmod.prepare(ApplyOptions::default())?];
    let sources = [RefModSource {
        slot: 1,
        member: 1,
        media: Media::Picture(original),
        provenance: serde_json::json!({"path":"original.png"}),
        synthetic_timing: false,
    }];
    generator.generate_refmods_with_sources(None, RefModPromptRequest {
        instruction: "The referenced character greets the viewer",
        entries: &[],
        refmods: &mods,
        shape,
        presentation: Default::default(),
    }, &sources)
}
```

Every active member needs a source when the session is `None`; missing sources
fail before contacting the endpoint. Pass `Some(&mut session)` to reconstruct
unmapped members through the VAEs. Original media changes presentation only;
use the same RefMods, ordering and preparation options for latent conditioning.

`PromptGenerator::generate_refmods(&mut session, request)` reconstructs every
active member through the local VAEs before calling the endpoint. Both methods
return the prompt and matching `PreparedPresentation`; pass that presentation and
the same prepared references to `Session::denoise_presented`.

`RefModPresentationOptions::{max_media_bytes, fps}` controls the host float-media
budget (default 1 GiB) and reconstructed playback rate (default 24 fps). Copies
share decoded buffers; disabled members are skipped. The host budget excludes
GPU allocations and serialized endpoint payloads. See
[RefMod presentation](refmods.md#upstream-presentation) and
[session memory budgets](runtime-options.md#shared-allocation-budgets).
