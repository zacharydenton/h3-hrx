# ComfyUI spatial compositor oracle

`comfy.safetensors` contains float32 output samples from the unmodified
`split_tiles`, `blend`, and `tiled_decode` methods in
`comfy/ldm/minimax/vae.py` at ComfyUI commit
`a7169322485d0049380fb207fa17e9fb3ec40486`. This includes the composited-neighbour
fix from [`fc584aaa`](https://github.com/comfyanonymous/ComfyUI/commit/fc584aaa226560ccdfe70c2bcfe9424af1adeb04).

Only `_decode_tile_row` was replaced. It yields synthetic `[1,3,2,H,W]` tiles:

```text
value(c, frame, y, x) = 100*c + 10*frame + 3*tile_row + 7*tile_column
                       + y/256 + x/512
```

The methods ran on CPU with PyTorch 2.14.0, in float32. The test reproduces the
input formula; it does not reproduce ComfyUI's compositing algorithm.

Cases cover a single tile, horizontal and vertical triple overlaps, and overlap
intersections: 64×96, 96×480, 480×96, 320×320, and 480×864. Samples include every
16th pixel, the final pixel, and the pixels immediately before, at, and after
each tile start. Tensor axes are `[channel,frame,sampled_y,sampled_x]`.

Safetensors metadata contains the dimensions, sample coordinates, ComfyUI commit,
and SHA-256 of the source file. No model weights or real media are included.
