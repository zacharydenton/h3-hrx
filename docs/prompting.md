# H3 prompts and references

Describe the picture, action and sound in H3's structured format. Pass the prompt
on stdin or with `-p`; generation uses it verbatim.

Start with a [showcase prompt](showcase.md), or write the fields below.
[Endpoint prompt generation](prompt-generation.md) can turn a short instruction
and reference media into this format.

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
`ref2va` checkpoint, a different set of six sections — see the [format reference](#format-reference)).

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

## Keyframes and references

```sh
h3 --first-frame opening.png --out animated.mp4 < keyframe-prompt.txt
h3 person.jpg voice.wav --out referenced.mp4 < reference-prompt.txt
h3 --refmod character.safetensors --out character.mp4 < reference-prompt.txt
```

`--first-frame` anchors the opening composition. `--last-frame` adds a final
keyframe and requires a first frame. Positional media supplies image, audio or
video references; `--video-audio INDEX` includes a positional video's soundtrack.
Use [RefMods](refmods.md) to encode references once and reuse them across renders.

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

## Cliff rider example

This single-shot example separates wind and hooves from its orchestral score:

```sh
h3 --width 1344 --height 768 --frames 124 --steps 31 --seed 7 \
   --out cliff_rider.mp4 < docs/prompts/cliff_rider_768p.txt
```

![cliff rider](media/cliff_rider_768p_strip.jpg)

## Optional prompt generation

Use `h3 prompt` to write a prompt before rendering, or `--generate-prompt` to
combine both steps. See [endpoint configuration and media inputs](prompt-generation.md)
and the [Rust RefMod example](prompt-generation.md#native-rust-refmod-prompting).

## Format reference

MiniMax's [prompt-writing guide](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)
defines text/keyframe modes in `references/base-en.txt` and reference mode in
`references/ref-en.txt`.
