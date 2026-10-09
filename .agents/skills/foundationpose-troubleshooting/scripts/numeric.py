#!/usr/bin/env python3
"""CPU-only FoundationPose input, pose, and tensor measurements (NumPy only)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def default_repo():
    """Find a containing library checkout; allow copied scripts with explicit --repo."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src/foundation_pose.cpp').is_file():
            return parent
    return Path.cwd()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_frame(path):
    with np.load(path, allow_pickle=False) as archive:
        frame = {key: archive[key] for key in archive.files}
    validate_frame(frame)
    return frame


def validate_frame(frame, require_mask=False):
    for key in ("rgb", "depth_m", "K"):
        if key not in frame:
            raise ValueError(f"capture is missing {key}")
    rgb, depth, k = (frame[key] for key in ("rgb", "depth_m", "K"))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be HWC uint8 RGB")
    if min(rgb.shape[:2]) < 1 or depth.shape != rgb.shape[:2] or depth.dtype != np.float32:
        raise ValueError("depth_m must be HW float32, matching nonempty RGB, in meters")
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("K must be a finite 3x3 matrix with positive focal lengths")
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("library intrinsics support fx, fy, cx, cy only (no skew)")
    mask = frame.get("mask")
    if mask is not None and (mask.shape != depth.shape or mask.dtype != np.uint8):
        raise ValueError("mask must be HW uint8, matching RGB and depth")
    if require_mask and (mask is None or not np.any(mask)):
        raise ValueError("registration requires a nonempty target mask")


def statistics(array):
    array = np.asarray(array)
    finite = array[np.isfinite(array)].astype(np.float64)
    result = {"shape": list(array.shape), "dtype": str(array.dtype),
              "elements": int(array.size), "nonfinite": int(array.size - finite.size)}
    if finite.size:
        result.update(zip(("min", "p05", "median", "p95", "max"),
                          np.percentile(finite, [0, 5, 50, 95, 100]).tolist()))
        result["mean"] = float(finite.mean())
    return result


def inspect_frame(frame, depth_min=0.001, depth_max=100.0):
    validate_frame(frame)
    if not 0 < depth_min < depth_max:
        raise ValueError("require 0 < depth_min < depth_max")
    depth, k = frame["depth_m"], frame["K"]
    valid = np.isfinite(depth) & (depth >= depth_min) & (depth < depth_max)
    result = {"depth_m": statistics(depth), "valid_depth_pixels": int(valid.sum()),
              "valid_depth_fraction": float(valid.mean()), "K": k.tolist(),
              "depth_valid_interval_m": [depth_min, depth_max], "warnings": []}
    if np.any(~np.isfinite(depth)):
        result["warnings"].append("Nonfinite input depths are preserved; inspect sensor encoding.")
    if "mask" not in frame:
        return result
    mask = frame["mask"] != 0
    count = int(mask.sum())
    masked_valid = mask & valid
    result.update(mask_pixels=count, mask_fraction=float(mask.mean()),
                  valid_mask_depth_pixels=int(masked_valid.sum()),
                  valid_mask_depth_fraction=float(masked_valid.sum() / count) if count else None,
                  masked_valid_depth_m=statistics(depth[masked_valid]))
    if not count:
        result["warnings"].append("Empty target mask.")
        return result
    y, x = np.where(mask)
    u, v = (float(x.min() + x.max()) / 2, float(y.min() + y.max()) / 2)
    result["mask_bbox_xyxy_inclusive"] = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    result["mask_bbox_midpoint_uv"] = [u, v]
    if np.any(masked_valid):
        z = float(np.median(depth[masked_valid]))
        result["raw_mask_center_proxy_m"] = [(u - float(k[0, 2])) * z / float(k[0, 0]),
                                            (v - float(k[1, 2])) * z / float(k[1, 1]), z]
        result["center_proxy_note"] = "Raw mask + exact median; not CUDA's filtered histogram initialization."
    else:
        result["warnings"].append("No valid depth under target mask.")
    return result


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite 4x4 object-to-camera matrix")
    r = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-3, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-3, rtol=0)):
        raise ValueError("pose is not a rigid SE(3) transform")
    return pose


def pose_error(pose, reference, point=None):
    pose, reference = validate_pose(pose), validate_pose(reference)
    cosine = (np.trace(pose[:3, :3] @ reference[:3, :3].T) - 1) / 2
    delta = pose[:3, 3] - reference[:3, 3]
    result = {"translation_error_m": float(np.linalg.norm(delta)),
              "translation_delta_camera_m": delta.tolist(),
              "rotation_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
              "symmetry_handling": "none; compare application-equivalent reference poses separately"}
    if point is not None:
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("grasp point must be three finite coordinates in the original mesh, meters")
        displacement = (pose @ np.r_[point, 1] - reference @ np.r_[point, 1])[:3]
        result["grasp_point_error_m"] = float(np.linalg.norm(displacement))
        result["grasp_point_delta_camera_m"] = displacement.tolist()
    return result


def compare_arrays(a, b, atol=1e-6, rtol=1e-5):
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("tolerances must be finite and nonnegative")
    result = {"a": statistics(a), "b": statistics(b), "atol": atol, "rtol": rtol}
    if a.shape != b.shape:
        return {**result, "passes": False, "reason": "shape mismatch"}
    a, b = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = a[finite] - b[finite]
    result.update(finite_pairs=int(finite.sum()),
                  passes=bool(finite.all() and np.allclose(a, b, atol=atol, rtol=rtol)))
    if delta.size:
        result.update(max_abs=float(np.max(np.abs(delta))),
                      mean_abs=float(np.mean(np.abs(delta))),
                      rmse=float(np.sqrt(np.mean(delta ** 2))))
    return result


def load_array(path):
    path = Path(path)
    return np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="measure an exact RGB-D NPZ capture")
    inspect.add_argument("capture", type=Path)
    inspect.add_argument("--depth-min", type=float, default=0.001)
    inspect.add_argument("--depth-max", type=float, default=100.0)
    pose = sub.add_parser("pose", help="SE(3) error; both poses use original mesh coordinates/meters")
    pose.add_argument("estimate", type=Path)
    pose.add_argument("reference", type=Path)
    pose.add_argument("--grasp-point", nargs=3, type=float, metavar=("X", "Y", "Z"))
    compare = sub.add_parser("compare", help="compare aligned NPY/text arrays; exit 1 outside tolerance")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--atol", type=float, default=1e-6)
    compare.add_argument("--rtol", type=float, default=1e-5)
    for command in (inspect, pose, compare):
        command.add_argument("--output", type=Path, help="new JSON file (never overwritten)")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            result = inspect_frame(load_frame(args.capture), args.depth_min, args.depth_max)
        elif args.command == "pose":
            result = pose_error(load_array(args.estimate), load_array(args.reference), args.grasp_point)
        else:
            result = compare_arrays(load_array(args.a), load_array(args.b), args.atol, args.rtol)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(result.get("passes") is False)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
