"""CPU-only binding/cache/runner regressions; no model weights or GPU contexts."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from h3pipe_loom import H3Pipe, Shape
import kernel_cache
import kernel_test


class ReviewRegressions(unittest.TestCase):
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
