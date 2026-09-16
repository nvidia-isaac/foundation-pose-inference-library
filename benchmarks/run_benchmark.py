#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the NVIDIA wide CUDA/nvdiffrast/TensorRT library on BOP YCB-V.

The script owns dataset I/O in Python and calls the C++ shared library through
ctypes. That keeps the core estimator linkable from C++ while avoiding a hard
OpenCV dependency in the library itself.
"""

from __future__ import annotations

import argparse
import csv
import ctypes as C
import json
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


DATASET = "ycbv"
CSV_NAME_REGISTER = "foundationpose_ycbv-test.csv"
CSV_NAME_TRACKING = "foundationpose_ycbv-test_tracking.csv"
TIMING_NAME_REGISTER = "timing_ycbv_register.json"
TIMING_NAME_TRACKING = "timing_ycbv_tracking.json"
SELECTED_TARGETS_NAME = "selected_targets_ycbv.json"
MODE_OUTPUTS = {
    "register": (CSV_NAME_REGISTER, TIMING_NAME_REGISTER),
    "tracking": (CSV_NAME_TRACKING, TIMING_NAME_TRACKING),
}


class CreateOptions(C.Structure):
    _fields_ = [
        ("cad_path", C.c_char_p),
        ("device_id", C.c_int),
        ("mesh_unit_scale", C.c_float),
        ("refine_model_path", C.c_char_p),
        ("score_model_path", C.c_char_p),
        ("engine_cache_dir", C.c_char_p),
        ("refine_model_name", C.c_char_p),
        ("score_model_name", C.c_char_p),
        ("rendered_input_name", C.c_char_p),
        ("observed_input_name", C.c_char_p),
        ("refine_translation_output_name", C.c_char_p),
        ("refine_rotation_output_name", C.c_char_p),
        ("score_output_name", C.c_char_p),
    ]


class Config(C.Structure):
    """Mirrors fp_config_t. Fields are appended, never inserted: keep this in
    the same order as include/foundation_pose_nvidia/c_api.h."""

    _fields_ = [
        ("n_hypotheses", C.c_int),
        ("n_refine_iters", C.c_int),
        ("n_track_iters", C.c_int),
        ("crop_ratio", C.c_float),
        ("input_width", C.c_int),
        ("input_height", C.c_int),
        ("max_image_width", C.c_int),
        ("max_image_height", C.c_int),
        ("capture_cuda_graph", C.c_int),
        ("model_free_sample_stride", C.c_int),
        ("model_free_max_vertices", C.c_int),
        ("model_free_depth_edge_threshold", C.c_float),
        ("tensorrt_precision", C.c_int),
        ("batch_size", C.c_int),
    ]


class ImageView(C.Structure):
    _fields_ = [
        ("width", C.c_int),
        ("height", C.c_int),
        ("rgb_u8", C.POINTER(C.c_uint8)),
        ("depth_m", C.POINTER(C.c_float)),
        ("mask_u8", C.POINTER(C.c_uint8)),
        ("k_row_major", C.POINTER(C.c_float)),
    ]


class ReferenceImageView(C.Structure):
    _fields_ = [
        ("width", C.c_int),
        ("height", C.c_int),
        ("rgb_u8", C.POINTER(C.c_uint8)),
        ("depth_m", C.POINTER(C.c_float)),
        ("mask_u8", C.POINTER(C.c_uint8)),
        ("k_row_major", C.POINTER(C.c_float)),
        ("camera_to_world_row_major", C.POINTER(C.c_float)),
    ]


class PoseResult(C.Structure):
    _fields_ = [
        ("pose_row_major", C.c_float * 16),
        ("score", C.c_float),
    ]


@dataclass
class DatasetLayout:
    bop_root: Path
    dataset_root: Path
    models_dir: Path
    targets_path: Path
    target_count: int
    image_count: int
    object_ids: list[int]
    scene_ids: list[int]
    warnings: list[str]
    errors: list[str]


def encode_optional(value: str | Path | None) -> bytes | None:
    if value is None:
        return None
    text = str(value)
    return text.encode("utf-8") if text else None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_id_filter(value: str | None) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(piece))
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else v


def sample_icosphere(min_vertices: int) -> np.ndarray:
    t = (1.0 + 5.0**0.5) * 0.5
    vertices = [
        _normalize(np.array(v, dtype=np.float64))
        for v in [
            (-1.0, t, 0.0),
            (1.0, t, 0.0),
            (-1.0, -t, 0.0),
            (1.0, -t, 0.0),
            (0.0, -1.0, t),
            (0.0, 1.0, t),
            (0.0, -1.0, -t),
            (0.0, 1.0, -t),
            (t, 0.0, -1.0),
            (t, 0.0, 1.0),
            (-t, 0.0, -1.0),
            (-t, 0.0, 1.0),
        ]
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    def midpoint(a: int, b: int, cache: dict[tuple[int, int], int]) -> int:
        key = tuple(sorted((a, b)))
        if key not in cache:
            vertices.append(_normalize((vertices[a] + vertices[b]) * 0.5))
            cache[key] = len(vertices) - 1
        return cache[key]

    while len(vertices) < min_vertices:
        cache: dict[tuple[int, int], int] = {}
        next_faces: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab = midpoint(a, b, cache)
            bc = midpoint(b, c, cache)
            ca = midpoint(c, a, cache)
            next_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = next_faces
    return np.asarray(vertices, dtype=np.float64)


def _rotation_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_in_object_from_view(cam_pos: np.ndarray) -> np.ndarray:
    z_axis = _normalize(-cam_pos)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), z_axis)
    if np.linalg.norm(x_axis) < 0.001:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    out = np.eye(4, dtype=np.float64)
    out[:3, 0] = x_axis
    out[:3, 1] = y_axis
    out[:3, 2] = z_axis
    out[:3, 3] = cam_pos
    return out


def _geodesic_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    value = (float(np.trace(r1 @ r2.T)) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _symmetry_transforms(model_info: dict[str, Any]) -> list[np.ndarray]:
    transforms = [np.eye(4, dtype=np.float64)]
    for sym in model_info.get("symmetries_discrete", []):
        tf = np.asarray(sym, dtype=np.float64).reshape(4, 4)
        tf[:3, 3] /= 1000.0
        transforms.append(tf)
    for sym in model_info.get("symmetries_continuous", []):
        axis = _normalize(np.asarray(sym["axis"], dtype=np.float64).reshape(3))
        offset = np.asarray(sym.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3) / 1000.0
        k = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        for angle_deg in range(0, 360, 5):
            theta = np.deg2rad(angle_deg)
            r = np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
            tf = np.eye(4, dtype=np.float64)
            tf[:3, :3] = r
            tf[:3, 3] = offset - r @ offset
            transforms.append(tf)
    return transforms


def symmetry_clustered_rotation_count(model_info: dict[str, Any]) -> int:
    rotations: list[np.ndarray] = []
    for view in sample_icosphere(40):
        cam_in_obj = _camera_in_object_from_view(view)
        for j in range(int(360.0 / 60.0)):
            cam_rot = cam_in_obj[:3, :3] @ _rotation_z(j * np.deg2rad(60.0))
            cam_tf = np.eye(4, dtype=np.float64)
            cam_tf[:3, :3] = cam_rot
            cam_tf[:3, 3] = cam_in_obj[:3, 3]
            rotations.append(np.linalg.inv(cam_tf)[:3, :3].copy())

    threshold = np.deg2rad(30.0)
    symmetries = _symmetry_transforms(model_info)
    clustered: list[np.ndarray] = []
    for candidate in rotations:
        is_new = True
        for existing in clustered:
            for sym in symmetries:
                if _geodesic_distance(candidate @ sym[:3, :3], existing) < threshold:
                    is_new = False
                    break
            if not is_new:
                break
        if is_new:
            clustered.append(candidate)
    return len(clustered)


def build_hypothesis_counts(layout: DatasetLayout, object_ids: set[int], max_hypotheses: int) -> dict[int, int]:
    models_info = read_json(layout.models_dir / "models_info.json")
    return {
        obj_id: min(max_hypotheses, symmetry_clustered_rotation_count(models_info.get(str(obj_id), {})))
        for obj_id in sorted(object_ids)
    }


def hypothesis_count_for(args: argparse.Namespace, obj_id: int) -> int:
    counts = getattr(args, "hypothesis_counts_by_obj", {})
    return int(counts.get(int(obj_id), args.n_hypotheses))


def validate_dataset(bop_root: Path, models_dir_override: Path | None, output_dir: Path) -> DatasetLayout:
    warnings: list[str] = []
    errors: list[str] = []
    if "BOP" not in str(bop_root):
        warnings.append("BOP root path does not contain the case-sensitive string 'BOP'.")

    dataset_root = bop_root / DATASET
    targets_path = dataset_root / "test_targets_bop19.json"
    if not dataset_root.is_dir():
        errors.append(f"Missing dataset directory: {dataset_root}")
    if not targets_path.is_file():
        errors.append(f"Missing target file: {targets_path}")

    candidates = [
        models_dir_override,
        dataset_root / "models",
        bop_root / f"{DATASET}_models" / "models",
        dataset_root / f"{DATASET}_models" / "models",
    ]
    models_dir = next((p for p in candidates if p is not None and p.is_dir()), dataset_root / "models")
    if not models_dir.is_dir():
        errors.append(f"Missing models directory: {models_dir}")
    if not (models_dir / "models_info.json").is_file():
        errors.append(f"Missing models_info.json under: {models_dir}")

    targets: list[dict[str, Any]] = read_json(targets_path) if targets_path.is_file() else []
    object_ids = sorted({int(t["obj_id"]) for t in targets})
    scene_ids = sorted({int(t["scene_id"]) for t in targets})
    image_count = len({(int(t["scene_id"]), int(t["im_id"])) for t in targets})

    output_dir.mkdir(parents=True, exist_ok=True)
    layout = DatasetLayout(
        bop_root=bop_root,
        dataset_root=dataset_root,
        models_dir=models_dir,
        targets_path=targets_path,
        target_count=len(targets),
        image_count=image_count,
        object_ids=object_ids,
        scene_ids=scene_ids,
        warnings=warnings,
        errors=errors,
    )
    (output_dir / "dataset_validation_ycbv.json").write_text(
        json.dumps(layout.__dict__ | {
            "bop_root": str(layout.bop_root),
            "dataset_root": str(layout.dataset_root),
            "models_dir": str(layout.models_dir),
            "targets_path": str(layout.targets_path),
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return layout


def filter_targets(
    targets: list[dict[str, Any]],
    scene_ids: set[int] | None,
    obj_ids: set[int] | None,
    image_ids: set[int] | None,
    max_targets: int | None,
) -> list[dict[str, Any]]:
    out = []
    for target in targets:
        if scene_ids is not None and int(target["scene_id"]) not in scene_ids:
            continue
        if obj_ids is not None and int(target["obj_id"]) not in obj_ids:
            continue
        if image_ids is not None and int(target["im_id"]) not in image_ids:
            continue
        out.append(target)
        if max_targets is not None and len(out) >= max_targets:
            break
    return out


def candidate_libraries(root: Path) -> list[Path]:
    names = [
        "libfoundation_pose_nvidia.so",
        "libfoundation_pose_nvidia.dylib",
        "foundation_pose_nvidia.dll",
    ]
    dirs = [
        root / "build",
        root / "build" / "Release",
        root / "build" / "Debug",
        root / "out" / "build",
    ]
    return [d / name for d in dirs for name in names]


def load_library(path: Path | None) -> C.CDLL:
    root = Path(__file__).resolve().parents[1]
    selected = path
    if selected is None:
        selected = next((p for p in candidate_libraries(root) if p.exists()), None)
    if selected is None or not selected.exists():
        searched = "\n  ".join(str(p) for p in candidate_libraries(root))
        raise FileNotFoundError(f"Could not find shared library. Searched:\n  {searched}")

    lib = C.CDLL(str(selected))
    lib.fp_default_config.argtypes = [C.POINTER(Config)]
    lib.fp_default_config.restype = None
    lib.fp_create.argtypes = [C.POINTER(CreateOptions), C.POINTER(Config), C.c_char_p, C.c_size_t]
    lib.fp_create.restype = C.c_void_p
    lib.fp_create_model_free.argtypes = [
        C.POINTER(ReferenceImageView),
        C.c_size_t,
        C.POINTER(CreateOptions),
        C.POINTER(Config),
        C.c_char_p,
        C.c_size_t,
    ]
    lib.fp_create_model_free.restype = C.c_void_p
    lib.fp_prepare.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.c_size_t]
    lib.fp_prepare.restype = C.c_int
    lib.fp_destroy.argtypes = [C.c_void_p]
    lib.fp_destroy.restype = None
    lib.fp_register_frame.argtypes = [
        C.c_void_p,
        C.POINTER(ImageView),
        C.c_int,
        C.c_int,
        C.POINTER(PoseResult),
        C.c_char_p,
        C.c_size_t,
    ]
    lib.fp_register_frame.restype = C.c_int
    lib.fp_track_frame.argtypes = [
        C.c_void_p,
        C.POINTER(ImageView),
        C.c_int,
        C.POINTER(PoseResult),
        C.c_char_p,
        C.c_size_t,
    ]
    lib.fp_track_frame.restype = C.c_int
    lib.fp_synchronize_device.argtypes = [C.c_char_p, C.c_size_t]
    lib.fp_synchronize_device.restype = C.c_int
    lib.fp_build_info.argtypes = []
    lib.fp_build_info.restype = C.c_char_p
    print(lib.fp_build_info().decode("utf-8"))
    return lib


class Estimator:
    def __init__(
        self,
        lib: C.CDLL,
        args: argparse.Namespace,
        cad_path: Path,
        config: Config,
        obj_id: int,
        prepare_batch: int,
    ):
        self.lib = lib
        self.obj_id = int(obj_id)
        self.setup_timings: list[dict[str, Any]] = []
        self._strings = [
            encode_optional(cad_path),
            encode_optional(args.refine_model_path),
            encode_optional(args.score_model_path),
            encode_optional(args.engine_cache_dir),
            encode_optional(args.refine_model_name),
            encode_optional(args.score_model_name),
            encode_optional(args.rendered_input_name),
            encode_optional(args.observed_input_name),
            encode_optional(args.refine_translation_output_name),
            encode_optional(args.refine_rotation_output_name),
            encode_optional(args.score_output_name),
        ]
        options = CreateOptions(
            C.c_char_p(self._strings[0]),
            int(args.device_id),
            C.c_float(args.mesh_unit_scale),
            C.c_char_p(self._strings[1]),
            C.c_char_p(self._strings[2]),
            C.c_char_p(self._strings[3]),
            C.c_char_p(self._strings[4]),
            C.c_char_p(self._strings[5]),
            C.c_char_p(self._strings[6]),
            C.c_char_p(self._strings[7]),
            C.c_char_p(self._strings[8]),
            C.c_char_p(self._strings[9]),
            C.c_char_p(self._strings[10]),
        )
        err = C.create_string_buffer(4096)
        start = time.perf_counter()
        self.handle = lib.fp_create(C.byref(options), C.byref(config), err, len(err))
        create_elapsed = time.perf_counter() - start
        self.setup_timings.append(
            {
                "obj_id": self.obj_id,
                "type": "create_estimator",
                "time_s": create_elapsed,
                "cad_path": str(cad_path),
            }
        )
        if not self.handle:
            raise RuntimeError(err.value.decode("utf-8"))
        if args.prepare_estimators:
            start = time.perf_counter()
            status = lib.fp_prepare(self.handle, int(prepare_batch), err, len(err))
            self.synchronize()
            prepare_elapsed = time.perf_counter() - start
            self.setup_timings.append(
                {
                    "obj_id": self.obj_id,
                    "type": "prepare_estimator",
                    "time_s": prepare_elapsed,
                    "batch_size": int(prepare_batch),
                }
            )
            if status != 0:
                raise RuntimeError(err.value.decode("utf-8"))

    def close(self) -> None:
        if self.handle:
            self.lib.fp_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "Estimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register(self, rgb: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, k: np.ndarray, n_refine: int, n_hypotheses: int) -> tuple[np.ndarray, float]:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        k = np.ascontiguousarray(k.reshape(9), dtype=np.float32)
        view = ImageView(
            rgb.shape[1],
            rgb.shape[0],
            rgb.ctypes.data_as(C.POINTER(C.c_uint8)),
            depth_m.ctypes.data_as(C.POINTER(C.c_float)),
            mask.ctypes.data_as(C.POINTER(C.c_uint8)),
            k.ctypes.data_as(C.POINTER(C.c_float)),
        )
        result = PoseResult()
        err = C.create_string_buffer(4096)
        self.synchronize()
        start = time.perf_counter()
        status = self.lib.fp_register_frame(
            self.handle,
            C.byref(view),
            n_refine,
            n_hypotheses,
            C.byref(result),
            err,
            len(err),
        )
        self.synchronize()
        elapsed = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(err.value.decode("utf-8"))
        pose = np.asarray(result.pose_row_major, dtype=np.float32).reshape(4, 4)
        return (pose, float(result.score)), elapsed

    def track(self, rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray, n_refine: int) -> tuple[np.ndarray, float]:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        k = np.ascontiguousarray(k.reshape(9), dtype=np.float32)
        view = ImageView(
            rgb.shape[1],
            rgb.shape[0],
            rgb.ctypes.data_as(C.POINTER(C.c_uint8)),
            depth_m.ctypes.data_as(C.POINTER(C.c_float)),
            C.POINTER(C.c_uint8)(),
            k.ctypes.data_as(C.POINTER(C.c_float)),
        )
        result = PoseResult()
        err = C.create_string_buffer(4096)
        self.synchronize()
        start = time.perf_counter()
        status = self.lib.fp_track_frame(
            self.handle,
            C.byref(view),
            n_refine,
            C.byref(result),
            err,
            len(err),
        )
        self.synchronize()
        elapsed = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(err.value.decode("utf-8"))
        pose = np.asarray(result.pose_row_major, dtype=np.float32).reshape(4, 4)
        return pose, elapsed

    def synchronize(self) -> None:
        err = C.create_string_buffer(1024)
        status = self.lib.fp_synchronize_device(err, len(err))
        if status != 0:
            raise RuntimeError(err.value.decode("utf-8"))


def find_gt_id(scene_gt: dict[str, Any], im_id: int, obj_id: int) -> int | None:
    for index, entry in enumerate(scene_gt.get(str(im_id), [])):
        if int(entry["obj_id"]) == obj_id:
            return index
    return None


def load_frame(layout: DatasetLayout, scene_id: int, im_id: int, obj_id: int, need_mask: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    scene_dir = layout.dataset_root / "test" / f"{scene_id:06d}"
    rgb = cv2.imread(str(scene_dir / "rgb" / f"{im_id:06d}.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Missing RGB image for scene {scene_id} image {im_id}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    scene_camera = read_json(scene_dir / "scene_camera.json")
    cam = scene_camera[str(im_id)]
    k = np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3)
    depth_scale = float(cam.get("depth_scale", 1.0))
    depth_raw = cv2.imread(str(scene_dir / "depth" / f"{im_id:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Missing depth image for scene {scene_id} image {im_id}")
    depth_m = depth_raw.astype(np.float32) * depth_scale / 1000.0

    if not need_mask:
        return rgb, depth_m, None, k

    scene_gt = read_json(scene_dir / "scene_gt.json")
    gt_id = find_gt_id(scene_gt, im_id, obj_id)
    if gt_id is None:
        raise FileNotFoundError(f"No GT entry for scene {scene_id} image {im_id} obj {obj_id}")
    mask = cv2.imread(
        str(scene_dir / "mask_visib" / f"{im_id:06d}_{gt_id:06d}.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if mask is None:
        raise FileNotFoundError(f"Missing mask for scene {scene_id} image {im_id} obj {obj_id}")
    return rgb, depth_m, (mask > 0).astype(np.uint8), k


def format_rotation(pose: np.ndarray) -> str:
    return " ".join(f"{v:.6f}" for v in pose[:3, :3].reshape(-1))


def format_translation(pose: np.ndarray) -> str:
    return " ".join(f"{v:.4f}" for v in (pose[:3, 3] * 1000.0).reshape(-1))


def make_row(
    scene_id: int,
    im_id: int,
    obj_id: int,
    score: float,
    pose: np.ndarray,
    elapsed: float,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "scene_id": scene_id,
            "im_id": im_id,
            "obj_id": obj_id,
            "score": f"{score:.6f}",
            "R": format_rotation(pose),
            "t": format_translation(pose),
            "time": f"{elapsed:.6f}",
        },
        {
            "scene_id": scene_id,
            "im_id": im_id,
            "obj_id": obj_id,
            "time_s": elapsed,
            "type": kind,
        },
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def eta_text(start_wall: float, completed: int, total: int, initial_completed: int) -> str:
    completed_this_run = completed - initial_completed
    if completed_this_run <= 0 or completed >= total:
        return "eta=0s" if completed >= total else "eta=unknown"
    elapsed = time.perf_counter() - start_wall
    remaining = (elapsed / completed_this_run) * (total - completed)
    finish_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining))
    return f"eta={format_duration(remaining)} finish_local={finish_at}"


def summarize_values(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(t["time_s"]) for t in records]
    by_type: dict[str, dict[str, Any]] = {}
    for kind in sorted({str(t.get("type", "unknown")) for t in records}):
        kind_values = np.asarray([float(t["time_s"]) for t in records if str(t.get("type", "unknown")) == kind], dtype=np.float64)
        if kind_values.size:
            by_type[kind] = {
                "count": int(kind_values.size),
                "total_time_s": float(kind_values.sum()),
                "mean_time_s": float(kind_values.mean()),
                "median_time_s": float(np.median(kind_values)),
                "p95_time_s": float(np.percentile(kind_values, 95)),
                "min_time_s": float(kind_values.min()),
                "max_time_s": float(kind_values.max()),
            }
    if not values:
        return {
            "count": 0,
            "total_time_s": 0.0,
            "mean_time_s": 0.0,
            "median_time_s": 0.0,
            "p95_time_s": 0.0,
            "p99_time_s": 0.0,
            "min_time_s": 0.0,
            "max_time_s": 0.0,
            "by_type": by_type,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "total_time_s": float(arr.sum()),
        "mean_time_s": float(arr.mean()),
        "median_time_s": float(statistics.median(values)),
        "p95_time_s": float(np.percentile(arr, 95)),
        "p99_time_s": float(np.percentile(arr, 99)),
        "min_time_s": float(arr.min()),
        "max_time_s": float(arr.max()),
        "by_type": by_type,
    }


def summarize_timings(
    timings: list[dict[str, Any]],
    setup_timings: list[dict[str, Any]],
    total_targets: int,
    wall: float,
    mode: str,
) -> dict[str, Any]:
    values = [float(t["time_s"]) for t in timings]
    by_type: dict[str, dict[str, Any]] = {}
    for kind in sorted({str(t.get("type", "unknown")) for t in timings}):
        kind_values = np.asarray([float(t["time_s"]) for t in timings if str(t.get("type", "unknown")) == kind], dtype=np.float64)
        if kind_values.size:
            by_type[kind] = {
                "count": int(kind_values.size),
                "mean_time_s": float(kind_values.mean()),
                "median_time_s": float(np.median(kind_values)),
                "p95_time_s": float(np.percentile(kind_values, 95)),
                "min_time_s": float(kind_values.min()),
                "max_time_s": float(kind_values.max()),
            }
    if not values:
        return {
            "mode": mode,
            "total_targets": total_targets,
            "completed": 0,
            "total_wall_time_s": wall,
            "mean_time_s": 0.0,
            "median_time_s": 0.0,
            "p95_time_s": 0.0,
            "p99_time_s": 0.0,
            "min_time_s": 0.0,
            "max_time_s": 0.0,
            "by_type": by_type,
            "setup": summarize_values(setup_timings),
            "per_setup": setup_timings,
            "per_frame": timings,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mode": mode,
        "total_targets": total_targets,
        "completed": len(values),
        "total_wall_time_s": wall,
        "mean_time_s": float(arr.mean()),
        "median_time_s": float(statistics.median(values)),
        "p95_time_s": float(np.percentile(arr, 95)),
        "p99_time_s": float(np.percentile(arr, 99)),
        "min_time_s": float(arr.min()),
        "max_time_s": float(arr.max()),
        "by_type": by_type,
        "setup": summarize_values(setup_timings),
        "per_setup": setup_timings,
        "per_frame": timings,
    }


def write_outputs(
    output_dir: Path,
    csv_name: str,
    timing_name: str,
    rows: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    setup_timings: list[dict[str, Any]],
    total_targets: int,
    wall: float,
    mode: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        writer.writeheader()
        writer.writerows(rows)
    timing_path = output_dir / timing_name
    timing_path.write_text(
        json.dumps(summarize_timings(timings, setup_timings, total_targets, wall, mode), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {timing_path}")
    return csv_path, timing_path


def selected_targets_path(output_dir: Path, value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else output_dir / path
    return output_dir / SELECTED_TARGETS_NAME


def print_latency_summary(summary: dict[str, Any], mode: str) -> None:
    by_type_key = "track" if mode == "tracking" else "register"
    stats = summary.get("by_type", {}).get(by_type_key)
    if stats is None:
        return
    mean_ms = stats["mean_time_s"] * 1000
    median_ms = stats["median_time_s"] * 1000
    print(
        f"[{mode}] latency per object"
        f" — mean={mean_ms:.2f}ms  median={median_ms:.2f}ms"
        f"  (by_type['{by_type_key}'], n={stats['count']})"
    )


def write_selected_targets(output_dir: Path, value: str | None, targets: list[dict[str, Any]]) -> Path:
    path = selected_targets_path(output_dir, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return path


def apply_image_times(rows: list[dict[str, Any]], image_times: dict[tuple[int, int], float]) -> None:
    for row in rows:
        key = (int(row["scene_id"]), int(row["im_id"]))
        row["time"] = f"{image_times.get(key, 0.0):.6f}"


def target_key(target: dict[str, Any]) -> tuple[int, int, int]:
    return (int(target["scene_id"]), int(target["im_id"]), int(target["obj_id"]))


def load_resume_state(
    output_dir: Path,
    csv_name: str,
    timing_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], float], set[tuple[int, int, int]]]:
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    setup_timings: list[dict[str, Any]] = []
    image_times: dict[tuple[int, int], float] = {}
    completed: set[tuple[int, int, int]] = set()

    csv_path = output_dir / csv_name
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                completed.add(target_key(row))

    timing_path = output_dir / timing_name
    if timing_path.is_file():
        data = read_json(timing_path)
        for item in data.get("per_frame", []):
            timings.append(dict(item))
            key = (int(item["scene_id"]), int(item["im_id"]))
            image_times[key] = image_times.get(key, 0.0) + float(item["time_s"])
        for item in data.get("per_setup", []):
            setup_timings.append(dict(item))

    return rows, timings, setup_timings, image_times, completed


def run_register_mode(
    args: argparse.Namespace,
    lib: C.CDLL,
    layout: DatasetLayout,
    targets: list[dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Path:
    csv_name, timing_name = MODE_OUTPUTS["register"]
    if args.resume:
        rows, timings, setup_timings, image_times, completed = load_resume_state(output_dir, csv_name, timing_name)
        if completed:
            print(f"[register] resuming with {len(completed)}/{len(targets)} completed targets")
    else:
        rows, timings, setup_timings, image_times, completed = [], [], [], {}, set()

    initial_completed = len(completed)
    estimators: OrderedDict[int, Estimator] = OrderedDict()
    ordered_targets = (
        sorted(targets, key=lambda t: (int(t["obj_id"]), int(t["scene_id"]), int(t["im_id"])))
        if args.group_register_by_object
        else targets
    )
    start_wall = time.perf_counter()
    try:
        for index, target in enumerate(ordered_targets, start=1):
            scene_id, im_id, obj_id = target_key(target)
            key = (scene_id, im_id, obj_id)
            if key in completed:
                continue
            if obj_id not in estimators:
                if args.max_register_estimators > 0 and len(estimators) >= args.max_register_estimators:
                    _, evicted = estimators.popitem(last=False)
                    evicted.close()
                cad = layout.models_dir / f"obj_{obj_id:06d}.ply"
                estimators[obj_id] = Estimator(
                    lib,
                    args,
                    cad,
                    config,
                    obj_id,
                    hypothesis_count_for(args, obj_id),
                )
                setup_timings.extend(estimators[obj_id].setup_timings)
            else:
                estimators.move_to_end(obj_id)
            rgb, depth_m, mask, k = load_frame(layout, scene_id, im_id, obj_id)
            if mask is None:
                raise RuntimeError(f"Missing register mask for scene {scene_id} image {im_id} obj {obj_id}")
            (pose, score), elapsed = estimators[obj_id].register(
                rgb, depth_m, mask, k, args.n_refine, hypothesis_count_for(args, obj_id)
            )
            row, timing = make_row(scene_id, im_id, obj_id, score, pose, elapsed, "register")
            rows.append(row)
            image_times[(scene_id, im_id)] = image_times.get((scene_id, im_id), 0.0) + elapsed
            timings.append(timing)
            completed.add(key)

            completed_count = len(completed)
            if args.checkpoint_interval > 0 and completed_count % args.checkpoint_interval == 0:
                apply_image_times(rows, image_times)
                write_outputs(
                    output_dir,
                    csv_name,
                    timing_name,
                    rows,
                    timings,
                    setup_timings,
                    len(targets),
                    time.perf_counter() - start_wall,
                    "register",
                )
                print(f"[register] checkpoint {completed_count}/{len(targets)} {eta_text(start_wall, completed_count, len(targets), initial_completed)}")
            if index % 25 == 0 or completed_count == len(targets):
                print(
                    f"[register] {completed_count}/{len(targets)} scene={scene_id} "
                    f"im={im_id} obj={obj_id} time={elapsed:.4f}s "
                    f"{eta_text(start_wall, completed_count, len(targets), initial_completed)}"
                )
    finally:
        for estimator in estimators.values():
            estimator.close()

    apply_image_times(rows, image_times)
    csv_path, _ = write_outputs(
        output_dir,
        csv_name,
        timing_name,
        rows,
        timings,
        setup_timings,
        len(targets),
        time.perf_counter() - start_wall,
        "register",
    )
    print_latency_summary(summarize_timings(timings, setup_timings, len(targets), 0.0, "register"), "register")
    return csv_path


def grouped_tracking_targets(targets: list[dict[str, Any]]) -> list[tuple[tuple[int, int], list[dict[str, Any]]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for target in targets:
        obj_id = int(target["obj_id"])
        scene_id = int(target["scene_id"])
        groups.setdefault((obj_id, scene_id), []).append(target)
    return [
        (key, sorted(group, key=lambda t: int(t["im_id"])))
        for key, group in sorted(groups.items())
    ]


def enumerate_scene_frames(layout: DatasetLayout, scene_id: int) -> list[int]:
    """All image ids of a test scene in ascending (video) order.

    Reads scene_gt.json keys, so it returns every consecutive frame when the
    full-frame split is installed (scripts/download_bop_ycbv.sh test_all) and
    only the sparse BOP19 keyframes otherwise.
    """
    scene_dir = layout.dataset_root / "test" / f"{scene_id:06d}"
    scene_gt = read_json(scene_dir / "scene_gt.json")
    return sorted(int(k) for k in scene_gt)


def run_tracking_mode(
    args: argparse.Namespace,
    lib: C.CDLL,
    layout: DatasetLayout,
    targets: list[dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Path:
    csv_name, timing_name = MODE_OUTPUTS["tracking"]
    if args.resume:
        rows, timings, setup_timings, image_times, completed = load_resume_state(output_dir, csv_name, timing_name)
        if completed:
            print(f"[tracking] resuming with {len(completed)}/{len(targets)} completed targets")
    else:
        rows, timings, setup_timings, image_times, completed = [], [], [], {}, set()

    total = len(targets)
    initial_completed = len(completed)
    start_wall = time.perf_counter()
    groups = grouped_tracking_targets(targets)
    for group_index, ((obj_id, scene_id), group_targets) in enumerate(groups, start=1):
        print(
            f"[tracking] group {group_index}/{len(groups)} object={obj_id:02d} "
            f"scene={scene_id:06d} targets={len(group_targets)}"
        )
        cad = layout.models_dir / f"obj_{obj_id:06d}.ply"
        estimator = Estimator(
            lib,
            args,
            cad,
            config,
            obj_id,
            hypothesis_count_for(args, obj_id),
        )
        setup_timings.extend(estimator.setup_timings)
        tracking_ready = False
        try:
            for target in group_targets:
                scene_id, im_id, obj_id = target_key(target)
                key = (scene_id, im_id, obj_id)
                already_completed = key in completed
                timing_type: str
                need_seed = not tracking_ready
                rgb, depth_m, mask, k = load_frame(layout, scene_id, im_id, obj_id, need_mask=need_seed)
                if need_seed:
                    if mask is None:
                        raise RuntimeError(f"Missing tracking seed mask for scene {scene_id} image {im_id} obj {obj_id}")
                    (pose, score), elapsed = estimator.register(
                        rgb, depth_m, mask, k, args.n_refine, hypothesis_count_for(args, obj_id)
                    )
                    tracking_ready = True
                    timing_type = "tracking_seed_register"
                else:
                    pose, elapsed = estimator.track(rgb, depth_m, k, args.n_track_refine)
                    score = 1.0
                    timing_type = "track"

                if already_completed:
                    continue

                row, timing = make_row(scene_id, im_id, obj_id, score, pose, elapsed, timing_type)
                rows.append(row)
                image_times[(scene_id, im_id)] = image_times.get((scene_id, im_id), 0.0) + elapsed
                timings.append(timing)
                completed.add(key)

                completed_count = len(completed)
                if args.checkpoint_interval > 0 and completed_count % args.checkpoint_interval == 0:
                    apply_image_times(rows, image_times)
                    write_outputs(
                        output_dir,
                        csv_name,
                        timing_name,
                        rows,
                        timings,
                        setup_timings,
                        total,
                        time.perf_counter() - start_wall,
                        "tracking",
                    )
                    print(f"[tracking] checkpoint {completed_count}/{total} {eta_text(start_wall, completed_count, total, initial_completed)}")
                if completed_count % 25 == 0 or completed_count == total:
                    print(
                        f"[tracking] {completed_count}/{total} scene={scene_id} "
                        f"im={im_id} obj={obj_id} type={timing_type} time={elapsed:.4f}s "
                        f"{eta_text(start_wall, completed_count, total, initial_completed)}"
                    )
        finally:
            estimator.close()

    apply_image_times(rows, image_times)
    csv_path, _ = write_outputs(
        output_dir,
        csv_name,
        timing_name,
        rows,
        timings,
        setup_timings,
        total,
        time.perf_counter() - start_wall,
        "tracking",
    )
    print_latency_summary(summarize_timings(timings, setup_timings, total, 0.0, "tracking"), "tracking")
    return csv_path


def run_dense_tracking_mode(
    args: argparse.Namespace,
    lib: C.CDLL,
    layout: DatasetLayout,
    targets: list[dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Path:
    """Video (frame-to-frame) tracking on the full test sequences.

    For each (object, scene) the estimator registers the FIRST frame of the
    sequence, then tracks EVERY consecutive frame (gap = 1) so the pose
    propagates through real video motion. Poses are recorded and scored only at
    the BOP19 keyframe targets (`targets`), so results are directly comparable to
    register / sparse tracking at the same eval points and the BOP evaluation
    stays fast. Requires the full-frame split (scripts/download_bop_ycbv.sh
    test_all); with only the sparse split this degrades to sparse tracking.
    """
    csv_name, timing_name = MODE_OUTPUTS["tracking"]
    if args.resume:
        rows, timings, setup_timings, image_times, completed = load_resume_state(output_dir, csv_name, timing_name)
        if completed:
            print(f"[dense-tracking] resuming with {len(completed)}/{len(targets)} completed targets")
    else:
        rows, timings, setup_timings, image_times, completed = [], [], [], {}, set()

    total = len(targets)
    initial_completed = len(completed)
    start_wall = time.perf_counter()
    groups = grouped_tracking_targets(targets)
    for group_index, ((obj_id, scene_id), group_targets) in enumerate(groups, start=1):
        eval_ims = {int(t["im_id"]) for t in group_targets}
        all_frames = enumerate_scene_frames(layout, scene_id)
        last_eval = max(eval_ims)
        frames = [im for im in all_frames if im <= last_eval]
        if not frames:
            continue
        if len(frames) <= len(eval_ims) + 1:
            print(
                f"[dense-tracking] WARNING scene={scene_id:06d}: only {len(frames)} frames available "
                f"(<= {len(eval_ims)} eval keyframes) — install the full-frame split "
                f"(scripts/download_bop_ycbv.sh test_all) for true video tracking."
            )
        print(
            f"[dense-tracking] group {group_index}/{len(groups)} object={obj_id:02d} "
            f"scene={scene_id:06d} track_frames={len(frames)} eval_frames={len(eval_ims)}"
        )
        cad = layout.models_dir / f"obj_{obj_id:06d}.ply"
        estimator = Estimator(
            lib,
            args,
            cad,
            config,
            obj_id,
            hypothesis_count_for(args, obj_id),
        )
        setup_timings.extend(estimator.setup_timings)
        try:
            for idx, im_id in enumerate(frames):
                need_seed = idx == 0
                is_eval = im_id in eval_ims
                rgb, depth_m, mask, k = load_frame(layout, scene_id, im_id, obj_id, need_mask=need_seed)
                if need_seed:
                    if mask is None:
                        raise RuntimeError(f"Missing dense-tracking seed mask for scene {scene_id} image {im_id} obj {obj_id}")
                    (pose, score), elapsed = estimator.register(
                        rgb, depth_m, mask, k, args.n_refine, hypothesis_count_for(args, obj_id)
                    )
                    timing_type = "tracking_seed_register"
                else:
                    # Every intermediate frame updates the tracker state (dense
                    # propagation) even though only keyframes are scored.
                    pose, elapsed = estimator.track(rgb, depth_m, k, args.n_track_refine)
                    score = 1.0
                    timing_type = "track"

                if not is_eval:
                    continue
                key = (scene_id, im_id, obj_id)
                if key in completed:
                    continue
                row, timing = make_row(scene_id, im_id, obj_id, score, pose, elapsed, timing_type)
                rows.append(row)
                image_times[(scene_id, im_id)] = image_times.get((scene_id, im_id), 0.0) + elapsed
                timings.append(timing)
                completed.add(key)

                completed_count = len(completed)
                if args.checkpoint_interval > 0 and completed_count % args.checkpoint_interval == 0:
                    apply_image_times(rows, image_times)
                    write_outputs(
                        output_dir, csv_name, timing_name, rows, timings, setup_timings,
                        total, time.perf_counter() - start_wall, "tracking",
                    )
                    print(f"[dense-tracking] checkpoint {completed_count}/{total} {eta_text(start_wall, completed_count, total, initial_completed)}")
                if completed_count % 25 == 0 or completed_count == total:
                    print(
                        f"[dense-tracking] {completed_count}/{total} scene={scene_id} "
                        f"im={im_id} obj={obj_id} type={timing_type} time={elapsed:.4f}s "
                        f"{eta_text(start_wall, completed_count, total, initial_completed)}"
                    )
        finally:
            estimator.close()

    apply_image_times(rows, image_times)
    csv_path, _ = write_outputs(
        output_dir,
        csv_name,
        timing_name,
        rows,
        timings,
        setup_timings,
        total,
        time.perf_counter() - start_wall,
        "tracking",
    )
    print_latency_summary(summarize_timings(timings, setup_timings, total, 0.0, "tracking"), "tracking")
    return csv_path


def run_bop_evaluations(
    args: argparse.Namespace,
    layout: DatasetLayout,
    output_dir: Path,
    targets_path: Path,
    csv_paths: list[Path],
) -> None:
    if not args.bop_toolkit_path:
        raise RuntimeError("--evaluate requires --bop-toolkit-path")
    env = os.environ.copy()
    env["BOP_PATH"] = str(layout.bop_root)
    env["BOP_RESULTS_PATH"] = str(output_dir)
    env["BOP_EVAL_PATH"] = str(output_dir / "eval")
    toolkit_path = Path(args.bop_toolkit_path).expanduser().resolve()
    exe_dir = str(Path(sys.executable).parent)
    env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(toolkit_path) + os.pathsep + env.get("PYTHONPATH", "")
    for csv_path in csv_paths:
        cmd = [
            sys.executable,
            str(toolkit_path / "scripts" / "eval_bop19_pose.py"),
            f"--result_filenames={csv_path.name}",
            f"--results_path={output_dir}",
            f"--eval_path={output_dir / 'eval_official' / csv_path.stem}",
            f"--targets_filename={targets_path}",
            f"--renderer_type={args.eval_renderer_type}",
            f"--num_workers={args.eval_num_workers}",
            f"--device={args.eval_device}",
        ]
        if args.eval_use_gpu:
            cmd.append("--use_gpu")
        if args.cleanup_eval:
            cmd.append("--cleanup_eval")
        log_path = output_dir / f"eval_{csv_path.stem}.log"
        print(f"Running BOP eval for {csv_path.name}; log={log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_benchmark(args: argparse.Namespace) -> None:
    if args.keep_register_estimators_resident:
        args.max_register_estimators = 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    layout = validate_dataset(Path(args.bop_path).expanduser().resolve(), Path(args.models_dir).expanduser().resolve() if args.models_dir else None, output_dir)
    if layout.warnings:
        for warning in layout.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    if layout.errors:
        raise RuntimeError("Dataset validation failed:\n  " + "\n  ".join(layout.errors))

    targets = filter_targets(
        read_json(layout.targets_path),
        parse_id_filter(args.scene_ids),
        parse_id_filter(args.obj_ids),
        parse_id_filter(args.image_ids),
        args.max_targets,
    )
    if args.validate_only:
        print(f"Validated {layout.target_count} targets; selected {len(targets)}.")
        return
    if not args.benchmark:
        raise RuntimeError("Pass --benchmark to run inference, or --validate-only to only validate the dataset.")
    targets_path = write_selected_targets(output_dir, args.selected_targets_path, targets)
    if args.use_symmetry_hypothesis_counts:
        args.hypothesis_counts_by_obj = build_hypothesis_counts(
            layout, {int(t["obj_id"]) for t in targets}, args.n_hypotheses
        )
        reduced = {
            obj_id: count
            for obj_id, count in args.hypothesis_counts_by_obj.items()
            if count < args.n_hypotheses
        }
        if reduced:
            summary = ", ".join(f"{obj_id}:{count}" for obj_id, count in sorted(reduced.items()))
            print(f"Using BOP symmetry-aware hypothesis caps: {summary}")
    else:
        args.hypothesis_counts_by_obj = {}

    lib = load_library(Path(args.library).expanduser().resolve() if args.library else None)
    config = Config()
    lib.fp_default_config(C.byref(config))
    config.n_refine_iters = args.n_refine
    config.n_track_iters = args.n_track_refine
    config.n_hypotheses = args.n_hypotheses
    config.input_width = args.input_size
    config.input_height = args.input_size
    config.max_image_width = args.max_image_width
    config.max_image_height = args.max_image_height
    config.capture_cuda_graph = int(args.capture_cuda_graph)
    config.tensorrt_precision = {"fp32": 0, "tf32": 1, "fp16": 2, "bf16": 3}[args.precision]

    modes = ["register", "tracking"] if args.mode == "both" else [args.mode]
    csv_paths: list[Path] = []
    for mode in modes:
        if mode == "register":
            csv_paths.append(run_register_mode(args, lib, layout, targets, config, output_dir))
        elif mode == "tracking":
            track_fn = run_dense_tracking_mode if args.track_all_frames else run_tracking_mode
            csv_paths.append(track_fn(args, lib, layout, targets, config, output_dir))
        else:
            raise RuntimeError(f"Unsupported benchmark mode: {mode}")

    if args.evaluate:
        run_bop_evaluations(args, layout, output_dir, targets_path, csv_paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
        epilog="""Examples:
  Validate dataset and model paths:
    python benchmarks/run_benchmark.py --validate-only --bop_path /data/BOP \\
      --refine-model-path /models/refiner_net.onnx \\
      --score-model-path /models/score_net.onnx \\
      --library ./build/libfoundation_pose_nvidia.so

  Run a 75-target register smoke evaluation:
    python benchmarks/run_benchmark.py --benchmark --mode register --evaluate \\
      --bop_path /data/BOP --bop-toolkit-path /src/bop_toolkit \\
      --refine-model-path /models/refiner_net.onnx \\
      --score-model-path /models/score_net.onnx \\
      --library ./build/libfoundation_pose_nvidia.so \\
      --engine-cache-dir ./engine_cache --max-targets 75 \\
      --eval-renderer-type vispy --eval-num-workers 10

  Run the full YCB-V register benchmark:
    python benchmarks/run_benchmark.py --benchmark --mode register --evaluate \\
      --bop_path /data/BOP --bop-toolkit-path /src/bop_toolkit \\
      --refine-model-path /models/refiner_net.onnx \\
      --score-model-path /models/score_net.onnx \\
      --library ./build/libfoundation_pose_nvidia.so \\
      --engine-cache-dir ./engine_cache --n-refine 5 --n-hypotheses 252 \\
      --prepare-estimators --group-register-by-object
""",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run the YCB-V benchmark.")
    parser.add_argument("--mode", default="register", choices=["register", "tracking", "both"], help="Benchmark register, tracking, or both streams.")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset layout and exit.")
    parser.add_argument("--bop_path", required=True, help="Path containing ycbv/. Must contain BOP for official tooling.")
    parser.add_argument("--models_dir", default=None, help="Override YCB-V models directory.")
    parser.add_argument("--library", default=None, help="Path to libfoundation_pose_nvidia shared library.")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "results"), help="Directory for CSV, timing JSON, selected-target JSON, and evaluator output.")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="tf32", help="TensorRT engine precision. fp32 = strict IEEE (best cross-GPU accuracy parity); tf32 is the default.")
    parser.add_argument("--n-refine", dest="n_refine", type=int, default=5, help="Register-mode refinement iterations.")
    parser.add_argument("--n-track-refine", dest="n_track_refine", type=int, default=2, help="Track-mode refinement iterations.")
    parser.add_argument("--track-all-frames", dest="track_all_frames", action="store_true", help="Video tracking: with --mode tracking, register the first frame of each scene and track EVERY consecutive frame (gap=1), scoring at the BOP19 keyframes. Requires the full-frame split (scripts/download_bop_ycbv.sh test_all).")
    parser.add_argument("--n-hypotheses", dest="n_hypotheses", type=int, default=252, help="Maximum register hypotheses before optional symmetry-aware capping.")
    parser.add_argument("--use-symmetry-hypothesis-counts", action=argparse.BooleanOptionalAction, default=True, help="Cap per-object register hypotheses to the Python/BOP symmetry-clustered rotation count.")
    parser.add_argument("--input-size", dest="input_size", type=int, default=160, help="Square crop size expected by the ONNX refiner/scorer.")
    parser.add_argument("--max-targets", type=int, default=None, help="Limit selected BOP targets for smoke runs.")
    parser.add_argument("--scene-ids", default=None, help="Comma-separated scene IDs to include.")
    parser.add_argument("--obj-ids", default=None, help="Comma-separated object IDs to include.")
    parser.add_argument("--image-ids", default=None, help="Comma-separated image IDs to include.")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index passed to the C++ estimator.")
    parser.add_argument("--max-register-estimators", type=int, default=1, help="Maximum per-object register estimators kept resident on GPU. Full-batch TensorRT contexts are memory-heavy; use 0 to keep all.")
    parser.add_argument("--keep-register-estimators-resident", action="store_true", help="Alias for --max-register-estimators 0. Keeps every created register estimator alive for the benchmark run; can require a lot of GPU memory.")
    parser.add_argument("--group-register-by-object", action="store_true", help="Process register targets grouped by object ID so one resident estimator can be reused across that object's targets.")
    parser.add_argument("--prepare-estimators", action=argparse.BooleanOptionalAction, default=True, help="Prepare TensorRT refiner/scorer contexts during setup so measured target time only covers register/track calls.")
    parser.add_argument("--mesh-unit-scale", type=float, default=0.001, help="YCB-V PLY meshes are in millimeters.")
    parser.add_argument("--engine-cache-dir", default=str(Path(__file__).resolve().parents[1] / "engine_cache"), help="Directory for TensorRT plan files.")
    parser.add_argument("--max-image-width", type=int, default=1280, help="Maximum image width used to size the persistent GPU workspace.")
    parser.add_argument("--max-image-height", type=int, default=720, help="Maximum image height used to size the persistent GPU workspace.")
    parser.add_argument("--capture-cuda-graph", action="store_true", help="Capture and replay the refinement body when shapes stay stable.")
    parser.add_argument("--refine-model-path", required=True, help="Path to refiner_net.onnx.")
    parser.add_argument("--score-model-path", required=True, help="Path to score_net.onnx.")
    parser.add_argument("--refine-model-name", default="refine", help="Name prefix used in TensorRT cache files for the refiner.")
    parser.add_argument("--score-model-name", default="score", help="Name prefix used in TensorRT cache files for the scorer.")
    parser.add_argument("--rendered-input-name", default="inputA", help="Rendered input tensor name in both ONNX models.")
    parser.add_argument("--observed-input-name", default="inputB", help="Observed input tensor name in both ONNX models.")
    parser.add_argument("--refine-translation-output-name", default="trans", help="Translation delta output tensor name in the refiner model.")
    parser.add_argument("--refine-rotation-output-name", default="rot", help="Rotation delta output tensor name in the refiner model.")
    parser.add_argument("--score-output-name", default="score", help="Score output tensor name in the scorer model.")
    parser.add_argument("--evaluate", action="store_true", help="Run BOP toolkit after writing CSV.")
    parser.add_argument("--bop-toolkit-path", default=None, help="Path to a bop_toolkit checkout. If omitted, the script relies on PYTHONPATH/import resolution.")
    parser.add_argument("--selected-targets-path", default=None, help="Where to write the selected target subset JSON. Relative paths are under output_dir.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing CSV/timing pair in output_dir.")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Write partial CSV/timing outputs every N completed targets. Use 0 to disable.")
    parser.add_argument("--eval-renderer-type", default="python", choices=["python", "vispy", "cpp"], help="Renderer backend for BOP VSD evaluation.")
    parser.add_argument("--eval-num-workers", type=int, default=4, help="BOP toolkit evaluator worker count.")
    parser.add_argument("--eval-use-gpu", action="store_true", help="Pass GPU evaluation flag through to BOP toolkit when supported.")
    parser.add_argument("--eval-device", default="cuda:0", help="Device string passed to BOP toolkit for GPU evaluation.")
    parser.add_argument("--cleanup-eval", action="store_true", help="Ask BOP toolkit to delete intermediate error folders after scoring.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
