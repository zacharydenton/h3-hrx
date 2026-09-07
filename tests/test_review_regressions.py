"""CPU-only binding/cache/runner regressions; no model weights or GPU contexts."""
import contextlib
import ctypes
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from h3pipe_loom import H3Pipe, Shape
import kernel_cache
import kernel_test


def spy_pipe(shape=Shape(22, 7, 4, 6, 37, 4096)):
    """An H3Pipe whose native calls record their arguments and succeed, with a fixed shape; no library is loaded."""
    pipe = H3Pipe.__new__(H3Pipe); pipe.shape = lambda p: shape; calls = []
    class Native:
        def __getattr__(self, name):
            def call(*args): calls.append((name, args)); return 0
            return call
    pipe._native, pipe._handle = Native(), None
    return pipe, calls


class ReviewRegressions(unittest.TestCase):
    def test_prompt_encoder_defaults_to_transformers(self):
        """Run the uncached CLI path with fake dependencies; no torch import or GPU context."""
        torch = MagicMock(); torch.nn.Module = object
        transformers = MagicMock()
        ids, embeds = MagicMock(), MagicMock()
        embeds.shape = (3, 5120); embeds.pow.return_value.mean.return_value.sqrt.return_value = 1.0
        transformers.AutoTokenizer.from_pretrained.return_value.return_value = {"input_ids": ids}
        model = MagicMock()
        model.model.language_model.return_value.hidden_states = [None] * 50 + [[embeds]]
        embeds.float.return_value.cpu.return_value = embeds
        spec = importlib.util.spec_from_file_location("review_encode_prompt", ROOT / "tools/encode_prompt.py")
        encoder = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            spec.loader.exec_module(encoder)
            for flags in ([], ["--torch"]):
                with tempfile.TemporaryDirectory() as tmp, patch.object(sys, "argv", ["encode_prompt.py", "a fox", "--out", tmp] + flags), \
                     patch.object(encoder, "load_encoder", return_value=model) as load, patch.object(encoder, "rotate_inputs") as rotate, \
                     contextlib.redirect_stdout(io.StringIO()):
                    encoder.main()
                    load.assert_called_once_with("cuda"); rotate.assert_called_once_with(model, "cuda")
                    saved, path = torch.save.call_args.args
                    self.assertIs(saved["embeds"], embeds)
                    self.assertEqual(path, encoder.prompt_path("a fox", Path(tmp)))

    def test_keyframe_shapes_are_rejected_before_native_call(self):
        pipe = H3Pipe.__new__(H3Pipe)
        pipe.shape = lambda p: Shape(22, 7, 4, 6, 37, 4096)
        p = H3Pipe.params(height=64, width=96, frames=22)
        # _native intentionally does not exist: a bad buffer must fail before ctypes.
        for shape in [(24, 1, 2, 2), (24, 1, 6, 4), (24, 2, 4, 6)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "keyframe video"):
                pipe.denoise([1], p, keyframes=[{"frame_index": 0, "video": np.zeros(shape, np.float32)}])
        valid = np.zeros((24, 1, 4, 6), np.float32)
        for field, value in [("pixels", np.zeros((32, 32))), ("audio", np.zeros((1, 32, 5)))]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"keyframe {field}"):
                pipe.denoise([1], p, keyframes=[{"frame_index": 0, "video": valid, field: value}])

    def test_valid_keyframe_reaches_native_with_correct_data(self):
        pipe = H3Pipe.__new__(H3Pipe)
        pipe.shape = lambda p: Shape(22, 7, 4, 6, 37, 4096)
        seen = []
        class Native:
            def h3pipe_denoise_refs(self, handle, ids, n, params, keyframes, count, *args):
                seen.append(np.ctypeslib.as_array(keyframes[0].video_latent, shape=(24 * 4 * 6,)).copy())
                return 0
        pipe._native, pipe._handle = Native(), None
        video = np.arange(24 * 4 * 6, dtype=np.float32).reshape(24, 1, 4, 6)
        pipe.denoise([1], H3Pipe.params(height=64, width=96, frames=22), keyframes=[{"frame_index": 0, "video": video}])
        np.testing.assert_array_equal(seen[0], video.ravel())

    def test_undersized_arrays_never_reach_native(self):
        """Every array crosses to C as a raw pointer with a size C derives from other arguments; the binding checks the shape first (also under python -O)."""
        pipe, calls = spy_pipe()
        p = H3Pipe.params(height=64, width=96, frames=22)
        bad = [
            ("encode_video", (np.zeros((1, 32, 32, 1)),), "channels"),        # grayscale: 1024 floats where C reads 3072
            ("encode_video", (np.zeros((32, 32)),), "dimensions"),
            ("encode_video", (np.zeros((1, 32, 40, 3)),), "multiples of 32"),
            ("encode_video", (np.zeros((0, 32, 32, 3)),), "empty"),
            ("vision_embed", (np.zeros((32, 32)),), "dimensions"),
            ("vision_embed", (np.zeros((32, 32, 4)),), "channels"),
            ("encode_audio", (np.zeros((1, 800)),), "shape"),                  # mono: 800 floats where C reads 1600
            ("encode_audio", (np.zeros(800),), "dimensions"),
            ("decode_audio", (np.zeros((2, 31, 5)),), "shape"),
            ("decode_video", (p, np.zeros((24, 7, 4, 5))), "shape"),
            ("denoise", ([1], p), "noise_video", {"noise_video": np.zeros((24, 7, 4, 5))}),
            ("denoise", ([1], p), "ref 0 image latents", {"refs": [{"kind": "image", "video": np.zeros((24, 2, 4, 6))}]}),   # an image reference is one latent frame
            ("denoise", ([1], p), "even lat_h", {"refs": [{"kind": "image", "video": np.zeros((24, 1, 3, 6))}]}),
            ("denoise", ([1], p), "ref 0 video latents", {"refs": [{"kind": "video", "video": np.zeros((23, 2, 4, 6))}]}),
            ("denoise", ([1], p), "ref 0 pixels", {"refs": [{"kind": "image", "video": np.zeros((24, 1, 4, 6)), "pixels": np.zeros((64, 96))}]}),
            ("denoise", ([1], p), "do not match", {"refs": [{"kind": "image", "video": np.zeros((24, 1, 4, 6)), "pixels": np.zeros((32, 32, 3))}]}),
            ("denoise", ([1], p), "ref 0 audio latents", {"refs": [{"kind": "audio", "audio": np.zeros((1, 32, 5))}]}),
            ("denoise", ([1], p), "audio needs", {"refs": [{"kind": "audio"}]}),
            ("denoise", ([1], p), "kind", {"refs": [{"kind": "picture", "video": np.zeros((24, 1, 4, 6))}]}),
            ("denoise", ([1], p), "keyframe pixels", {"keyframes": [{"frame_index": 0, "video": np.zeros((24, 1, 4, 6)), "pixels": np.zeros((32, 32, 3))}]}),
            ("denoise", ([], p), "at least one", {}),
        ]
        for method, args, message, *kw in bad:
            with self.subTest(method=method, message=message), self.assertRaisesRegex(ValueError, message):
                getattr(pipe, method)(*args, **(kw[0] if kw else {}))
        self.assertEqual(calls, [])
        # the well-formed calls do reach C with the sizes C will read
        pipe.encode_video(np.zeros((32, 32, 3))); pipe.vision_embed(np.zeros((32, 64, 3))); pipe.encode_audio(np.zeros((2, 801))); pipe.decode_audio(np.zeros((2, 32, 3)))
        pipe.denoise([1], p, refs=[{"kind": "image", "video": np.zeros((24, 1, 4, 6)), "pixels": np.zeros((64, 96, 3))}, {"kind": "audio", "audio": np.zeros((2, 32, 5))}])
        self.assertEqual([c[0] for c in calls], ["h3pipe_encode_video", "h3pipe_vision_embed", "h3pipe_encode_audio", "h3pipe_decode_audio", "h3pipe_denoise_refs"])
        self.assertEqual(calls[0][1][2:5], (1, 32, 32)); self.assertEqual(calls[2][1][2], 801)
        refs = calls[4][1][6]; self.assertEqual((refs[0].latent_t, refs[0].lat_h, refs[0].lat_w, refs[0].height, refs[0].width, refs[1].audio_t), (1, 4, 6, 64, 96, 5))

    def test_shape_failure_is_an_error(self):
        pipe = H3Pipe.__new__(H3Pipe)
        class Native:
            def h3pipe_shape_for(self, p, s): return 64
        pipe._native = Native()
        with self.assertRaisesRegex(ValueError, "invalid parameters"): pipe.shape(H3Pipe.params(height=31))

    def test_parity_gate_fails_on_broken_comparisons(self):
        """The ComfyUI parity gate: a comparison that exits nonzero, or omits an expected block or the trajectory line, is a failure; missing fixtures skip unless required."""
        spec = importlib.util.spec_from_file_location("parity", ROOT / "tests/test_comfy_parity.py"); parity = importlib.util.module_from_spec(spec); spec.loader.exec_module(parity)
        blocks = " ".join(f"blk_{b}: [video 0.9995]" for b in parity.REQUIRED_BLOCKS)
        import h3pipe_loom
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "tools").mkdir()
            for f in ("build/comfy_t2va_blocks/blocks/blk_49.npy", "build/comfy_fl2va/x_19.npy", "checkpoint.safetensors"):
                (root / f).parent.mkdir(parents=True, exist_ok=True); (root / f).write_text("")
            compare = root / "tools/compare_comfy.py"
            def fake(body): compare.write_text("import sys\n" + body)
            with patch.object(parity, "ROOT", root), patch.object(h3pipe_loom, "DIT", root / "checkpoint.safetensors"), contextlib.redirect_stdout(io.StringIO()):
                fake("sys.exit(42)"); self.assertEqual(parity.main(), 1)                                            # every comparison crashed
                fake("print('')"); self.assertEqual(parity.main(), 1)                                               # no results at all
                fake(f"print('{blocks}'.replace(' blk_', '\\nblk_'))"); self.assertEqual(parity.main(), 1)          # blocks but no trajectory
                fake(f"print('{blocks}'.replace(' blk_', '\\nblk_')); print('x_05: rel err 0.01')"); self.assertEqual(parity.main(), 0)
                fake(f"print('{blocks}'.replace(' blk_', '\\nblk_').replace('blk_40: [video 0.9995]', 'blk_40: [video 0.98]')); print('x_05: rel err 0.01')"); self.assertEqual(parity.main(), 1)
                fake(f"print('{blocks}'.replace(' blk_', '\\nblk_').replace('blk_40: [video 0.9995]\\n', '')); print('x_05: rel err 0.01')"); self.assertEqual(parity.main(), 1)   # block 40 missing
                fake(f"print('{blocks}'.replace(' blk_', '\\nblk_')); print('x_05: rel err 0.03')"); self.assertEqual(parity.main(), 1)
                (root / "checkpoint.safetensors").unlink()
                self.assertEqual(parity.main(), 0)                                                                  # a skip without the checkpoint
                with patch.dict(os.environ, {"H3_REQUIRE_PARITY": "1"}): self.assertEqual(parity.main(), 1)         # never for the release gate

    def test_clients_agree_on_the_checkpoint_layout(self):
        """h3, the binding and the examples must name the same four ComfyUI files under the same models directory."""
        import h3pipe_loom
        files = ["diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                 "vae/minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"]
        for name, want in zip(("DIT", "TE", "VIDEO_VAE", "AUDIO_VAE"), files):
            self.assertTrue(str(getattr(h3pipe_loom, name)).endswith(want), name)
        for path in ("host/h3_cli.cpp", "examples/c/minimal.c", "examples/rust/src/main.rs", "examples/go/main.go", "README.md"):
            text = (ROOT / path).read_text()
            for want in files: self.assertIn(want, text, f"{path} does not name {want}")
            self.assertIn("H3_MODELS", text, path)

    def test_cache_invalidation_and_failed_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output, compiler = (Path(tmp) / name for name in ("source.loom", "kernel.hsaco", "compiler"))
            source.write_text("v1"); compiler.write_text("compiler v1")
            calls = []
            def compile_stub(src, symbol, config, dest):
                calls.append((src.read_text(), symbol, dict(config)))
                dest.write_bytes(b"complete kernel")
            with patch.object(kernel_test, "LOOM_COMPILE", compiler), patch.object(kernel_test, "compile_kernel", compile_stub):
                def build(config=None, symbol="entry"):
                    kernel_cache.compile_cached(source, symbol, config or {"width": 32}, output)
                build(); build(); self.assertEqual(len(calls), 1)
                source.write_text("v2"); build(); self.assertEqual(len(calls), 2)
                build({"width": 64}); self.assertEqual(len(calls), 3)
                build({"width": 64}, "other"); self.assertEqual(len(calls), 4)
                output.unlink(); build(); self.assertEqual(len(calls), 5)
                compiler.write_text("compiler version two"); build(); self.assertEqual(len(calls), 6)
                source.write_text("v3")
                def fail(src, symbol, config, dest):
                    dest.write_bytes(b"incomplete")
                    raise RuntimeError("compile failed")
                with patch.object(kernel_test, "compile_kernel", fail), self.assertRaisesRegex(RuntimeError, "compile failed"):
                    build()
                self.assertEqual(output.read_bytes(), b"complete kernel")
                build(); self.assertEqual(len(calls), 7)

    def test_comfy_runner_preserves_process_status(self):
        script = (ROOT / "scripts/test.sh").read_text().splitlines()
        step = next(line for line in script if line.startswith("step()"))
        command = next(line for line in script if line.startswith('step "reference vs'))
        with tempfile.TemporaryDirectory() as tmp:
            podman = Path(tmp) / "podman"
            env = {**os.environ, "PATH": tmp + os.pathsep + os.environ["PATH"]}
            for rc, message in [(42, "RuntimeError: failed"), (137, ""), (0, "PASS")]:
                podman.write_text(f"#!/bin/sh\nprintf '%s\\n' '{message}'\nexit {rc}\n"); podman.chmod(0o755)
                result = subprocess.run(["bash", "-c", f'status=0\n{step}\n{command}\nexit "$status"'], env=env, capture_output=True)
                self.assertEqual(result.returncode, int(rc != 0))


if __name__ == "__main__":
    unittest.main()
