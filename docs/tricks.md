# Audio, stills, and reference modes

These modes use the same model and library as video generation. Start with
[setup](setup.md) and write a [prompt for the chosen mode](prompting.md).
Reference images and audio require the ref2va checkpoint.

## Audio only

A 32×32 canvas minimizes the video stream; `--audio-only` skips video decoding
and writes a WAV. The model still denoises both streams.

```sh
h3 --width 32 --height 32 --frames 124 --audio-only --out sound \
  < sound-prompt.txt
h3 voice.wav --width 32 --height 32 --frames 124 --audio-only --out voice \
  < reference-prompt.txt
```

Recorded on the Radeon 8060S on 2026-09-07: five seconds of audio at 32×32
cost about 0.6 s per evaluation from text alone, or 1.1 s with a five-second
voice reference. At 30 evaluations that is approximately 18 s or 33 s of
**denoising**, excluding weight loading, text encoding, and decoding. A
resident `Session` reuses loaded weights and compiled kernels across requests.

## Still images

`--still frame.png` extracts one decoded frame; `--still-frame N` chooses
which. Describe a static camera and still photograph in the prompt.

```sh
h3 --width 1344 --height 768 --frames 22 --steps 21 --still fox.png \
  < still-prompt.txt
h3 scene.jpg --width 864 --height 480 --frames 22 --steps 21 --still night.png \
  < edit-prompt.txt
```

Recorded 20-evaluation examples took 5.3 minutes for a 1344×768 text-to-image
run and 2.4 minutes for an 864×480 reference edit, including setup and decoding.
See the [fox still](media/fox_still_1344x768.jpg) and
[night edit](media/edit_night_neon.jpg). These are historical samples; the host
still generates a clip before selecting the frame.

## Conditioning

- `--first-frame image.png` pins an initial keyframe using the fl2va model.
- Positional images and audio files select ref2va. Images become `<Picture i>`
  and audio becomes `<Audio j>` in the prompt presentation.
- Video references and keyframes beyond frame zero are exposed by the
  Rust API, but not the CLI. Last-frame generation is unvalidated here.

For reusable kernels and numerical results, see the
[repository map](../CONTRIBUTING.md#repository-layout) and
[performance notes](performance.md).
