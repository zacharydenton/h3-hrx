# Audio, stills, and reference modes

See [setup](setup.md) and [prompt format](prompting.md) first.

## Audio only

`--audio-only` skips video decoding and writes a WAV. The model still denoises
both streams; a 32×32 canvas minimizes video work.

```sh
h3 --width 32 --height 32 --frames 124 --audio-only --out sound \
  < sound-prompt.txt
h3 voice.wav --width 32 --height 32 --frames 124 --audio-only --out voice \
  < reference-prompt.txt
```

A retaining library `Session` reuses loaded weights and compiled kernels across
requests. See [residency options](runtime-options.md).

## Still images

`--still frame.png` saves one decoded frame; `--still-frame N` selects it.
The model still generates a clip. Describe a static camera and still photograph.

```sh
h3 --width 1344 --height 768 --frames 22 --steps 21 --still fox.png \
  < still-prompt.txt
h3 scene.jpg --width 864 --height 480 --frames 22 --steps 21 --still night.png \
  < edit-prompt.txt
```

[Fox still](media/fox_still_1344x768.jpg) · [Night edit](media/edit_night_neon.jpg)

## Conditioning

- `--first-frame image.png` anchors frame zero using FL2VA.
- `--last-frame image.png` adds a final keyframe and requires a first frame.
  See the [Orbit example](loras.md#360-orbit-example).
- Positional images, audio and videos select Ref2VA. `--video-audio INDEX`
  explicitly includes a positional video's soundtrack.
- `--refmod FILE` reuses encoded references. See [RefMods](refmods.md) for strengths,
  original-media presentation and combining references with keyframes.
