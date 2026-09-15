# Native ComfyUI on the benchmark machine

The checkout is `~/code/ComfyUI`, updated to upstream commit
`b00584967778871f329c2d31190a5e9b767b8366` (ComfyUI 0.35.0). It runs directly on
Arch Linux with the existing `python-pytorch-opt-rocm 2.14.0-1` and
`hip-runtime-amd 7.2.4-1`. PyTorch reports HIP `7.2.53211` and the Strix Halo GPU.
The Python environment inherits those system packages; no container is involved.

## Python environment

The local overlay supplies the ComfyUI packages missing from the system Python.
Its exact additions are in [comfy-overlay.txt](comfy-overlay.txt). On this host,
the setup is:

```sh
uv venv --python /usr/bin/python3 --system-site-packages ~/code/ComfyUI/.venv-rocm
uv pip install --python ~/code/ComfyUI/.venv-rocm/bin/python --no-deps \
  --prerelease allow --index https://rocm.nightlies.amd.com/v2/gfx1151/ \
  -r docs/benchmarks/20260914/comfy-overlay.txt
```

The system already provides PyTorch, torchvision, NumPy, transformers,
safetensors, Hugging Face Hub, and the other dependencies recorded with the run.
`--no-deps` keeps this overlay from replacing the installed ROCm PyTorch with
another wheel. It is a recipe for the recorded host, not a complete Python
environment for a machine without those system packages.

Both the default and Comfy Kitchen INT8 paths passed a small full video/audio
generation before the 768p runs. ComfyUI's HTTP server and H3 model selectors
were also checked. FFmpeg is the system 9.0.1 binary with H.264 and AAC support,
shared by h3 and ComfyUI.

## Launch the UI

From the h3-hrx checkout:

```sh
~/code/ComfyUI/.venv-rocm/bin/python docs/benchmarks/20260914/serve_comfy.py
```

On the benchmark machine, `~/.local/bin/comfyui` runs that command. Open
`http://127.0.0.1:8188`. Extra ComfyUI arguments are forwarded, for example
`comfyui --port 8189`.

The launcher discovers the existing quantized H3 checkpoints in the standard
Hugging Face cache and supplies their snapshot directories through ComfyUI's
extra-model-paths configuration. It writes that configuration under
`~/.config/h3-hrx/` (or `$XDG_CONFIG_HOME/h3-hrx/`). No model copies or local
`ComfyUI/models` layout are needed. Cloud API nodes are disabled.

ComfyUI's **Model Attention Backend** node can select **comfy kitchen attention**
for the diffusion model. The benchmark's `--attention-backend comfy-kitchen-int8`
uses the same public model-patcher API and registered backend. It changes only
DiT attention; text encoding and VAE decoding retain their defaults.

Open [comfy-workflow-api.json](comfy-workflow-api.json) in ComfyUI to load the
alien-fjord scene with the cached quantized models, 768p, 124 frames, seed 1618,
and 20 evaluations. The workflow uses built-in nodes and defaults to Comfy
Kitchen attention. Select **pytorch attention** in **Model Attention Backend**
for the dense BF16 attention configuration. The workflow passed ComfyUI's prompt
validation without loading model weights.

The UI workflow saves through ComfyUI's standard video node. Timed comparisons
use [comfy_dump.py](../../../scripts/comfy_dump.py) in a fresh process with the
same sampling settings, explicit unloading between stages, and the shared
system FFmpeg encoder. A reused UI session can have different loading times and
memory residency because it caches models between requests.

## Run the benchmark queue

```sh
cargo build --release --locked --bin h3
python3 docs/benchmarks/20260914/run_day.py
```

Provision the model and HRX caches first using the project's normal setup. The
measurement queue runs offline and excludes downloads. Kernel caches are kept
between runs; first use of an uncached shape can still compile kernels.

Each process launches separately, after 30 seconds of GPU idleness and with at
least 80 GiB of system memory available. The wrapper samples memory every second
and stops its own workload if available memory falls below 12 GiB. The queue
stops on failure and refuses to overwrite existing logs. To continue after
investigating a failure, select only jobs that have not started with repeated
`--job NAME` arguments.

The default queue runs the six full renders. Profiling jobs require explicit
`--job` selection and their timings are excluded from generation speedups. The
ComfyUI PyTorch/HIP profiling job coincided with a host freeze on this machine;
see [the incident record](incident.json). It was not repeated. While benchmarking,
keep the UI idle and avoid competing GPU work or CPU compilation.
