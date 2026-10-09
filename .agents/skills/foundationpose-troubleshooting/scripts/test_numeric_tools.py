#!/usr/bin/env python3
"""CPU regression checks; no CAD, CUDA context, or model weights required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from make_trace_patch import instrument, make_patch
from numeric import compare_arrays, inspect_frame, load_frame, pose_error, validate_frame, write_json
from replay_capture import read_manifest, run_sequence


def example_frame():
    return {"rgb": np.zeros((3, 4, 3), np.uint8),
            "depth_m": np.full((3, 4), 0.75, np.float32),
            "mask": np.full((3, 4), 255, np.uint8),
            "K": np.array([[100, 0, 1.5], [0, 100, 1], [0, 0, 1]], np.float32)}


class Measurements(unittest.TestCase):
    def test_depth_holes_and_mask(self):
        frame = example_frame()
        frame["depth_m"][0] = [0, np.nan, np.inf, 100]
        report = inspect_frame(frame)
        self.assertEqual(report["valid_mask_depth_pixels"], 8)
        self.assertEqual(report["depth_m"]["nonfinite"], 2)
        self.assertEqual(report["mask_bbox_xyxy_inclusive"], [0, 0, 3, 2])
        np.testing.assert_allclose(report["raw_mask_center_proxy_m"], [0, 0, 0.75])
        json.dumps(report, allow_nan=False)

    def test_empty_mask_and_no_valid_depth(self):
        frame = example_frame()
        frame["mask"][:] = 0
        report = inspect_frame(frame)
        self.assertIsNone(report["valid_mask_depth_fraction"])
        with self.assertRaises(ValueError):
            validate_frame(frame, require_mask=True)
        frame["mask"][:] = 1
        frame["depth_m"][:] = 0
        self.assertNotIn("raw_mask_center_proxy_m", inspect_frame(frame))

    def test_bad_shape_dtype_and_calibration(self):
        for field, value in (("rgb", np.zeros((3, 4), np.uint8)),
                             ("depth_m", np.zeros((2, 4), np.float32)),
                             ("depth_m", np.zeros((3, 4), np.uint16)),
                             ("mask", np.zeros((3, 4), bool)),
                             ("K", np.zeros((3, 3)))):
            frame = example_frame()
            frame[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_frame(frame)

    def test_known_translation_rotation_and_grasp(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[2, 3] = 0.03
        result = pose_error(pose, np.eye(4), [0.1, 0, 0])
        self.assertAlmostEqual(result["translation_error_m"], 0.03)
        self.assertAlmostEqual(result["rotation_error_deg"], 90)
        self.assertAlmostEqual(result["grasp_point_error_m"], np.sqrt(0.0209))
        pose[0, 0] = 2
        with self.assertRaises(ValueError):
            pose_error(pose, np.eye(4))

    def test_array_difference_and_nonfinite_rejection(self):
        a = np.array([1, 2, 3], np.float32)
        self.assertTrue(compare_arrays(a, a)["passes"])
        result = compare_arrays(a, a + 0.25, atol=0, rtol=0)
        self.assertFalse(result["passes"])
        self.assertEqual(result["rmse"], 0.25)
        self.assertFalse(compare_arrays(a, a[:2])["passes"])
        self.assertFalse(compare_arrays(np.array([np.nan]), np.array([np.nan]))["passes"])


class Replay(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        np.savez(self.root / "frame.npz", **example_frame())
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"frames": [
            {"id": str(i), "path": "frame.npz", "action": action}
            for i, action in enumerate(("register", "track", "register", "track"))]}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_lossless_capture_and_no_clobber(self):
        frame = load_frame(self.root / "frame.npz")
        np.testing.assert_array_equal(frame["depth_m"], example_frame()["depth_m"])
        path = self.root / "report.json"
        write_json(path, {"ok": True})
        with self.assertRaises(FileExistsError):
            write_json(path, {"ok": False})
        self.assertTrue(json.loads(path.read_text())["ok"])

    def test_manifest_rejects_uninitialized_tracking(self):
        self.manifest.write_text(json.dumps({"frames": [
            {"id": "1", "path": "frame.npz", "action": "track"}]}))
        with self.assertRaises(ValueError):
            read_manifest(self.manifest)

    def test_replay_actions_masks_and_failure(self):
        entries = read_manifest(self.manifest)["frames"]
        calls = []

        class Estimator:
            def register(self, frame):
                calls.append("register")
                assert frame[3] is not None
                return SimpleNamespace(pose=np.eye(4, dtype=np.float32), score=2, elapsed_s=0.01)

            def track(self, frame):
                calls.append("track")
                assert frame[3] is None
                return SimpleNamespace(pose=np.eye(4, dtype=np.float32), score=1, elapsed_s=0.001)

        output = self.root / "success"
        output.mkdir()
        run_sequence(Estimator(), lambda *args: args, entries, output)
        self.assertEqual(calls, ["register", "track", "register", "track"])
        self.assertEqual(len(list(output.glob("*_pose.npy"))), 4)
        self.assertIn("placeholder", json.loads((output / "records.json").read_text())[1]["score_meaning"])

        class BrokenEstimator(Estimator):
            def track(self, frame):
                raise RuntimeError("lost GPU")

        output = self.root / "failure"
        output.mkdir()
        with self.assertRaises(RuntimeError):
            run_sequence(BrokenEstimator(), lambda *args: args, entries, output)
        records = json.loads((output / "records.json").read_text())
        self.assertEqual(len(records), 2)
        self.assertFalse(records[-1]["ok"])

    def test_validate_only_cli_and_repeat_output_refusal(self):
        output = self.root / "cli"
        argv = [sys.executable, str(Path(__file__).with_name("replay_capture.py")),
                "--manifest", str(self.manifest), "--output", str(output), "--validate-only"]
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(read_manifest(output / "manifest.json")["frames"]), 4)
        before = (output / "run.json").read_bytes()
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, (output / "run.json").read_bytes())
        self.assertFalse((output / "failure.json").exists())


class Instrumentation(unittest.TestCase):
    def test_patch_applies_and_source_drift_fails(self):
        repo = Path(__file__).resolve().parents[4]
        source = repo / "src/foundation_pose.cpp"
        if not source.exists():
            self.skipTest("run from a library checkout to check patch compatibility")
        patch = make_patch(repo)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/foundation_pose.cpp").write_bytes(source.read_bytes())
            patch_path = root / "trace.patch"
            patch_path.write_text(patch)
            # A standalone directory avoids touching the actual checkout.
            subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=root, check=True)
            subprocess.run(["git", "apply", str(patch_path)], cwd=root, check=True)
            with self.assertRaises(ValueError):
                make_patch(root)
        with self.assertRaises(ValueError):
            instrument(source.read_text().replace("  return estimate;", "  return changed_estimate;"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
