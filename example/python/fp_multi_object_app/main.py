#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runnable entry point for the Python multi-object BOP/YCB-V example.
# Loads one RGB-D frame plus per-object visible masks from a BOP scene, calls
# the multi-object FoundationPose C API wrapper, and writes inspection
# artifacts plus poses.json.

"""Run multi-object FoundationPose registration on one BOP/YCB-V scene frame."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from foundation_pose_nvidia import (
    MultiObjectEstimator,
    MultiObjectPoseEstimate,
    ObjectSpec,
    RegisterOptions,
)
from foundation_pose_nvidia.bindings import candidate_libraries

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "BOP_datasets" / "ycbv"
YCBV_MESH_UNIT_SCALE = 0.001
MIN_VISIBLE_PIXELS = 64
GT_POSE_KEY = "camera_from_object_opencv_row_major"
FRAME_ENCODING = "rgb8"


@dataclass
class BopFrame:
    rgb: np.ndarray
    k: np.ndarray


@dataclass
class MaskStats:
    visible_pixels: int
    bbox: tuple[int, int, int, int]
    median_depth_m: float


def auto_library_path(raw: str | None) -> Path:
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    env = os.environ.get("FP_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(candidate_libraries(REPO_ROOT))
    candidates.append(REPO_ROOT / "libfoundation_pose_nvidia.so")
    for path in candidates:
        if path.exists():
            return path.resolve()
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find libfoundation_pose_nvidia.so. "
        "Pass --library or set FP_LIBRARY. Searched: " + searched
    )


def require_existing_path(value: str | None, flag: str, env_var: str | None = None) -> Path:
    raw = value or (os.environ.get(env_var) if env_var else None)
    if not raw:
        env_hint = f" unless {env_var} is set" if env_var else ""
        raise ValueError(f"{flag} is required{env_hint}")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{flag} path does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root. Supports a BOP ycbv root or a local root with test/ and ycbv_models/.",
    )
    parser.add_argument(
        "--scene",
        default="000056",
        help="BOP scene id, e.g. 56 or 000056.",
    )
    parser.add_argument(
        "--frame",
        default="1",
        help="BOP frame id inside the scene, e.g. 1 or 000001.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Optional override for the YCB-V models directory.",
    )
    parser.add_argument(
        "--objects",
        default=None,
        help="Optional comma-separated object ids or names to keep, e.g. 2,4 or obj_000002,obj_000004.",
    )
    parser.add_argument(
        "--mode",
        choices=["register", "track"],
        default="track",
        help="register runs one-frame registration; track registers the start frame and tracks later frames in the same scene.",
    )
    parser.add_argument(
        "--track-frames",
        type=int,
        default=-1,
        help="Number of later frames in the same scene to track. Use -1 for all remaining frames.",
    )
    parser.add_argument("--library", default=os.environ.get("FP_LIBRARY"),
                        help="Path to libfoundation_pose_nvidia.so.")
    parser.add_argument("--refine", default=os.environ.get("FP_REFINE_MODEL_PATH"),
                        help="Path to refiner_net.onnx.")
    parser.add_argument("--score", default=os.environ.get("FP_SCORE_MODEL_PATH"),
                        help="Path to score_net.onnx.")
    parser.add_argument("--cache", default=os.environ.get(
        "FP_ENGINE_CACHE_DIR", str(REPO_ROOT / "engine_cache")),
                        help="TensorRT engine cache directory.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--n-hypotheses", type=int, default=64,
                        help="Register hypotheses. Use -1 for library default.")
    parser.add_argument("--n-refine", type=int, default=3,
                        help="Register refinement iterations. Use -1 for library default.")
    parser.add_argument("--capture-cuda-graph", action="store_true")
    parser.add_argument("--score-output", default="score_logit",
                        help="ScoreNet output tensor name for TAO deployable_v1.0.")
    parser.add_argument("--no-prepare", action="store_true",
                        help="Skip fp_group_prepare before register.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unique_paths(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def resolve_dataset_root(raw_dataset_root: Path) -> Path:
    candidates = [
        raw_dataset_root,
        raw_dataset_root / "ycbv",
    ]
    for candidate in candidates:
        if (candidate / "test").is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find a dataset root with test/ under {raw_dataset_root}. "
        f"Searched: {searched}"
    )


def resolve_models_dir(
    raw_dataset_root: Path,
    dataset_root: Path,
    raw_models_dir: str | None,
) -> Path:
    if raw_models_dir:
        models_dir = Path(raw_models_dir).expanduser().resolve()
        if not models_dir.is_dir():
            raise FileNotFoundError(f"--models-dir does not exist: {models_dir}")
        if not (models_dir / "models_info.json").is_file():
            raise FileNotFoundError(f"models_info.json is missing under: {models_dir}")
        return models_dir

    bases = unique_paths([raw_dataset_root.resolve(), dataset_root, dataset_root.parent])
    candidates: list[Path] = []
    for base in bases:
        candidates.extend(
            [
                base / "models",
                base / "ycbv_models" / "models",
                base / "ycbv_models" / "models_eval",
                base / "ycbv_models" / "models_fine",
            ]
        )
    for candidate in unique_paths(candidates):
        if candidate.is_dir() and (candidate / "models_info.json").is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in unique_paths(candidates))
    raise FileNotFoundError(f"Could not find a YCB-V models directory. Searched: {searched}")


def normalize_scene_name(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        return f"{int(raw):06d}"
    raise ValueError(f"Unsupported scene format: {raw!r}")


def normalize_frame_id(raw: str) -> int:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    raise ValueError(f"Unsupported frame format: {raw!r}")


def frame_label(frame_id: int) -> str:
    return f"{frame_id:06d}"


def scene_dir_for(dataset_root: Path, scene: str) -> Path:
    scene_name = normalize_scene_name(scene)
    path = dataset_root / "test" / scene_name
    if not path.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {path}")
    return path


def load_scene_metadata(scene_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_camera_path = scene_dir / "scene_camera.json"
    scene_gt_path = scene_dir / "scene_gt.json"
    if not scene_camera_path.exists():
        raise FileNotFoundError(f"scene_camera.json does not exist: {scene_camera_path}")
    if not scene_gt_path.exists():
        raise FileNotFoundError(f"scene_gt.json does not exist: {scene_gt_path}")
    return read_json(scene_camera_path), read_json(scene_gt_path)


def frame_ids_for(scene_camera: dict[str, Any]) -> list[int]:
    return sorted(int(key) for key in scene_camera)


def track_frame_ids(scene_camera: dict[str, Any], start_frame_id: int, track_frames: int) -> list[int]:
    later = [frame_id for frame_id in frame_ids_for(scene_camera) if frame_id > start_frame_id]
    if track_frames >= 0:
        later = later[:track_frames]
    return later


def pose_from_bop_entry(entry: dict[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    pose[:3, 3] = np.asarray(entry["cam_t_m2c"], dtype=np.float32).reshape(3) * 0.001
    return pose


def row_major_pose(pose: np.ndarray) -> list[float]:
    return pose.reshape(-1).astype(float).tolist()


def pose_from_object_meta(obj_meta: dict[str, Any]) -> np.ndarray:
    return np.asarray(obj_meta[GT_POSE_KEY], dtype=np.float32).reshape(4, 4)


def parse_raw_scene_objects(
    scene_dir: Path,
    frame_id: int,
    scene_gt: dict[str, Any],
    models_dir: Path,
) -> list[dict[str, Any]]:
    """Read one BOP frame's object instances into a uniform metadata structure.

    This expands each GT entry into the CAD path, visible-mask path, and metric pose that the
    multi-object pipeline needs before naming or filtering objects.
    """
    raw_objects: list[dict[str, Any]] = []
    for gt_index, entry in enumerate(scene_gt.get(str(frame_id), [])):
        obj_id = int(entry["obj_id"])
        cad_path = models_dir / f"obj_{obj_id:06d}.ply"
        if not cad_path.exists():
            raise FileNotFoundError(f"CAD mesh for obj_id={obj_id} does not exist: {cad_path}")
        raw_objects.append(
            {
                "obj_id": obj_id,
                "cad_path": str(cad_path.resolve()),
                "mesh_unit_scale": YCBV_MESH_UNIT_SCALE,
                "gt_index": gt_index,
                "mask_path": str((scene_dir / "mask_visib" / f"{frame_id:06d}_{gt_index:06d}.png").resolve()),
                "pose_m": pose_from_bop_entry(entry),
            }
        )
    return raw_objects


def assign_register_object_names(raw_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign stable object names for the registration frame.

    Repeated YCB-V ids can appear multiple times in one image, so this adds `#NN` suffixes when
    needed to give each instance a unique name for later tracking and reporting.
    """
    totals = Counter(int(obj["obj_id"]) for obj in raw_objects)
    seen: defaultdict[int, int] = defaultdict(int)
    objects: list[dict[str, Any]] = []
    for raw in raw_objects:
        obj_id = int(raw["obj_id"])
        base_name = f"obj_{obj_id:06d}"
        idx = seen[obj_id]
        seen[obj_id] += 1
        name = base_name if totals[obj_id] == 1 else f"{base_name}#{idx:02d}"
        objects.append(
            {
                "name": name,
                "obj_id": obj_id,
                "cad_path": raw["cad_path"],
                "mesh_unit_scale": raw["mesh_unit_scale"],
                "gt_index": raw["gt_index"],
                "mask_path": raw["mask_path"],
                GT_POSE_KEY: row_major_pose(raw["pose_m"]),
            }
        )
    return objects


def greedy_match_by_translation(
    reference_objects: list[dict[str, Any]],
    candidate_objects: list[dict[str, Any]],
    previous_gt_poses: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Pair current-frame candidates with previously named instances of the same object id.

    The match uses 3D translation distance so the tracker can keep instance names stable across
    frames when multiple copies of the same object class are present.
    """
    references = list(reference_objects)
    candidates = list(candidate_objects)
    matched: list[dict[str, Any]] = []
    while references and candidates:
        best_ref_idx: int | None = None
        best_cand_idx: int | None = None
        best_distance = float("inf")
        for ref_idx, ref in enumerate(references):
            ref_pose = previous_gt_poses.get(ref["name"], pose_from_object_meta(ref))
            ref_t = ref_pose[:3, 3]
            for cand_idx, candidate in enumerate(candidates):
                distance = float(np.linalg.norm(ref_t - candidate["pose_m"][:3, 3]))
                if distance < best_distance:
                    best_distance = distance
                    best_ref_idx = ref_idx
                    best_cand_idx = cand_idx
        assert best_ref_idx is not None and best_cand_idx is not None
        ref = references.pop(best_ref_idx)
        candidate = candidates.pop(best_cand_idx)
        matched.append(
            {
                "name": ref["name"],
                "obj_id": ref["obj_id"],
                "cad_path": candidate["cad_path"],
                "mesh_unit_scale": candidate["mesh_unit_scale"],
                "gt_index": candidate["gt_index"],
                "mask_path": candidate["mask_path"],
                GT_POSE_KEY: row_major_pose(candidate["pose_m"]),
            }
        )
    return matched


def align_track_objects(
    reference_objects: list[dict[str, Any]],
    raw_objects: list[dict[str, Any]],
    previous_gt_poses: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Reorder raw frame objects to match the registration-frame object list.

    This keeps later tracking frames aligned to the same object names and ordering expected by the
    multi-object estimator output.
    """
    ref_by_obj_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    cand_by_obj_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for ref in reference_objects:
        ref_by_obj_id[int(ref["obj_id"])].append(ref)
    for candidate in raw_objects:
        cand_by_obj_id[int(candidate["obj_id"])].append(candidate)

    matched_by_name: dict[str, dict[str, Any]] = {}
    for obj_id, refs in ref_by_obj_id.items():
        matched = greedy_match_by_translation(refs, cand_by_obj_id.get(obj_id, []), previous_gt_poses)
        for obj in matched:
            matched_by_name[str(obj["name"])] = obj
    return [matched_by_name[ref["name"]] for ref in reference_objects if ref["name"] in matched_by_name]


def normalize_frame_meta(
    scene_dir: Path,
    frame_id: int,
    scene_gt: dict[str, Any],
    models_dir: Path,
    *,
    reference_objects: list[dict[str, Any]] | None = None,
    previous_gt_poses: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build the canonical per-frame metadata dict used by the multi-object app.

    On the first frame it assigns unique instance names; on later frames it reuses the reference
    naming so selection, mask loading, and tracking all operate on a stable object list.
    """
    raw_objects = parse_raw_scene_objects(scene_dir, frame_id, scene_gt, models_dir)
    if reference_objects is None:
        objects = assign_register_object_names(raw_objects)
    else:
        objects = align_track_objects(reference_objects, raw_objects, previous_gt_poses or {})
    return {
        "scene": scene_dir.name,
        "frame": frame_label(frame_id),
        "objects": objects,
    }


def parse_object_selection(raw: str | None) -> set[str] | None:
    """Parse the optional `--objects` filter into normalized lookup tokens.

    The multi-object example accepts ids or names because users may want to keep one subset of
    instances from a frame instead of always processing every visible object.
    """
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    if not values:
        raise ValueError("--objects was provided but no object ids/names were parsed")
    return values


def object_matches_selection(obj: dict[str, Any], selected: set[str]) -> bool:
    """Check whether one object matches any accepted user-facing selector form."""
    obj_id = int(obj["obj_id"])
    base_name = f"obj_{obj_id:06d}"
    candidates = {
        str(obj["name"]),
        base_name,
        str(obj_id),
        f"{obj_id:06d}",
    }
    return any(value in selected for value in candidates)


def filter_frame_meta_objects(frame_meta: dict[str, Any], selected: set[str] | None) -> dict[str, Any]:
    """Drop objects that are not part of the requested multi-object subset.

    This happens before estimator setup so meshes, masks, and outputs only include the instances the
    user asked for.
    """
    if not selected:
        return frame_meta
    objects = [obj for obj in frame_meta["objects"] if object_matches_selection(obj, selected)]
    if not objects:
        available = ", ".join(obj["name"] for obj in frame_meta["objects"])
        raise ValueError(
            f"--objects matched no YCB-V objects in {frame_meta['scene']}/{frame_meta['frame']}. "
            f"Available: {available}"
        )
    filtered = dict(frame_meta)
    filtered["objects"] = objects
    return filtered


def load_frame(
    scene_dir: Path,
    frame_id: int,
    scene_camera: dict[str, Any],
) -> tuple[BopFrame, np.ndarray]:
    camera = scene_camera[str(frame_id)]
    depth_scale = float(camera.get("depth_scale", 1.0))
    k = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)

    frame_name = frame_label(frame_id)
    bgr = cv2.imread(str(scene_dir / "rgb" / f"{frame_name}.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image does not exist: {scene_dir / 'rgb' / f'{frame_name}.png'}")
    depth_raw = cv2.imread(str(scene_dir / "depth" / f"{frame_name}.png"), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise FileNotFoundError(
            f"Depth image does not exist: {scene_dir / 'depth' / f'{frame_name}.png'}"
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = depth_raw.astype(np.float32) * depth_scale / 1000.0
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"Depth shape {depth.shape} does not match RGB shape {rgb.shape[:2]}")
    return BopFrame(rgb=rgb, k=k), depth


def build_object_specs(frame_meta: dict[str, Any]) -> list[ObjectSpec]:
    """Convert normalized metadata into the library's `ObjectSpec` inputs.

    The single-object sample hardcodes one mesh, but the multi-object app has to build a spec list
    dynamically from the objects that survive frame parsing and filtering.
    """
    objects: list[ObjectSpec] = []
    for raw in frame_meta["objects"]:
        cad_path = Path(str(raw["cad_path"])).expanduser().resolve()
        if not cad_path.exists():
            raise FileNotFoundError(f"CAD mesh for {raw['name']!r} does not exist: {cad_path}")
        objects.append(
            ObjectSpec(
                name=str(raw["name"]),
                cad_path=cad_path,
                mesh_unit_scale=float(raw.get("mesh_unit_scale", YCBV_MESH_UNIT_SCALE)),
            )
        )
    if not objects:
        raise ValueError("No objects remain after frame filtering")
    return objects


def load_masks(
    frame_meta: dict[str, Any],
    objects: list[ObjectSpec],
) -> dict[str, np.ndarray]:
    """Load one visible mask per selected object instance.

    The multi-object app needs a mask map keyed by object name so
    registration can seed several instances from the same RGB-D frame.
    """
    object_meta = {
        str(item["name"]): item
        for item in frame_meta.get("objects", [])
        if "name" in item
    }
    masks: dict[str, np.ndarray] = {}
    for obj in objects:
        mask_path = Path(str(object_meta[obj.name]["mask_path"])).expanduser().resolve()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask for {obj.name!r} does not exist: {mask_path}")
        masks[obj.name] = np.where(mask > 0, 255, 0).astype(np.uint8)
    return masks


def count_visible_pixels(
    objects: list[ObjectSpec],
    masks: dict[str, np.ndarray],
) -> dict[str, int]:
    return {obj.name: int(np.count_nonzero(masks[obj.name])) for obj in objects}


def filter_objects_with_masks(
    objects: list[ObjectSpec],
    frame_meta: dict[str, Any],
    masks: dict[str, np.ndarray],
    visible_pixels: dict[str, int],
) -> tuple[list[ObjectSpec], dict[str, Any], dict[str, np.ndarray], list[str]]:
    """Remove instances whose visible masks are too small to register reliably.

    The function keeps the object spec list, metadata, and mask dictionary in sync after that
    pruning step so later code can assume they still describe the same object set.
    """
    kept_objects: list[ObjectSpec] = []
    kept_meta: list[dict[str, Any]] = []
    kept_masks: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    meta_by_name = {str(obj["name"]): obj for obj in frame_meta["objects"]}
    for obj in objects:
        if visible_pixels[obj.name] < MIN_VISIBLE_PIXELS:
            skipped.append(obj.name)
            continue
        kept_objects.append(obj)
        kept_meta.append(meta_by_name[obj.name])
        kept_masks[obj.name] = masks[obj.name]
    filtered_meta = dict(frame_meta)
    filtered_meta["objects"] = kept_meta
    return kept_objects, filtered_meta, kept_masks, skipped


def compute_mask_stats(
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    visible_pixels: dict[str, int],
) -> dict[str, MaskStats]:
    """Per-object mask geometry and depth, computed once for printing and artifacts."""
    stats: dict[str, MaskStats] = {}
    for name, mask in masks.items():
        values = depth[(mask > 0) & np.isfinite(depth) & (depth > 0.0)]
        stats[name] = MaskStats(
            visible_pixels=visible_pixels[name],
            bbox=mask_bbox(mask),
            median_depth_m=float(np.median(values)) if values.size else 0.0,
        )
    return stats


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute bounding box of empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        z_min = float(values.min())
        z_max = float(values.max())
        if abs(z_max - z_min) < 1e-6:
            out[valid] = 180
        else:
            out[valid] = np.clip(
                255.0 * (1.0 - (depth[valid] - z_min) / (z_max - z_min)),
                1.0,
                255.0,
            ).astype(np.uint8)
    return out


def rotation_error_deg(est_r: np.ndarray, gt_r: np.ndarray) -> float:
    delta = est_r @ gt_r.T
    value = 0.5 * (float(np.trace(delta)) - 1.0)
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def pose_estimate_to_output(result: MultiObjectPoseEstimate) -> dict[str, Any]:
    return {
        "name": result.object_spec.name,
        "status": result.status,
        "status_name": result.status_name,
        "score": result.score,
        "pose_row_major": result.pose.reshape(-1).astype(float).tolist(),
        "cad_path": str(result.object_spec.cad_path),
    }


def enrich_pose_results(
    pose_results: list[MultiObjectPoseEstimate],
    frame_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    gt_by_name = {
        str(item["name"]): item[GT_POSE_KEY]
        for item in frame_meta.get("objects", [])
        if "name" in item and item.get(GT_POSE_KEY) is not None
    }
    enriched: list[dict[str, Any]] = []
    for result in pose_results:
        item = pose_estimate_to_output(result)
        gt_raw = gt_by_name.get(str(item["name"]))
        if gt_raw is not None:
            est = np.asarray(item["pose_row_major"], dtype=np.float64).reshape(4, 4)
            gt = np.asarray(gt_raw, dtype=np.float64).reshape(4, 4)
            item[f"gt_{GT_POSE_KEY}"] = gt.reshape(-1).astype(float).tolist()
            item["translation_error_m"] = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
            item["rotation_error_deg"] = rotation_error_deg(est[:3, :3], gt[:3, :3])
        enriched.append(item)
    return enriched


def save_input_artifacts(
    output_dir: Path,
    dataset: Path,
    frame: BopFrame,
    depth: np.ndarray,
    objects: list[ObjectSpec],
    masks: dict[str, np.ndarray],
    stats: dict[str, MaskStats],
    frame_meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    np.save(output_dir / "depth_m.npy", depth)
    cv2.imwrite(str(output_dir / "depth_preview.png"), depth_preview(depth))

    overlay = frame.rgb.copy()
    colors = [
        np.array([255, 208, 0], dtype=np.uint8),
        np.array([0, 196, 255], dtype=np.uint8),
        np.array([120, 255, 120], dtype=np.uint8),
        np.array([255, 120, 200], dtype=np.uint8),
    ]
    meta_by_name = {
        str(item["name"]): item
        for item in frame_meta.get("objects", [])
        if "name" in item
    }
    metadata_objects: list[dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        mask = masks[obj.name]
        object_stats = stats[obj.name]
        bbox = object_stats.bbox
        color = colors[idx % len(colors)]
        cv2.imwrite(str(output_dir / f"mask_{obj.name}.png"), mask)
        overlay[mask > 0] = (
            0.55 * overlay[mask > 0].astype(np.float32) +
            0.45 * color.astype(np.float32)
        ).astype(np.uint8)
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color.tolist(), 2)
        cv2.putText(
            overlay,
            f"{obj.name} z={object_stats.median_depth_m:.2f}m",
            (bbox[0], max(20, bbox[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
        source_meta = meta_by_name.get(obj.name, {})
        metadata_objects.append(
            {
                "name": obj.name,
                "obj_id": int(source_meta.get("obj_id", -1)),
                "cad_path": str(obj.cad_path),
                "mesh_unit_scale": obj.mesh_unit_scale,
                "bbox_xyxy": list(bbox),
                "visible_pixels": object_stats.visible_pixels,
                "median_depth_m": object_stats.median_depth_m,
                "mask_path": source_meta.get("mask_path"),
                f"gt_{GT_POSE_KEY}": source_meta.get(GT_POSE_KEY),
            }
        )
    cv2.imwrite(str(output_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with (output_dir / "inputs.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(dataset),
                "scene": frame_meta["scene"],
                "frame": frame_meta["frame"],
                "encoding": FRAME_ENCODING,
                "width": int(frame.rgb.shape[1]),
                "height": int(frame.rgb.shape[0]),
                "camera_matrix_row_major": frame.k.reshape(-1).astype(float).tolist(),
                "objects": metadata_objects,
            },
            f,
            indent=2,
        )


def write_pose_results(output_dir: Path, frames: list[dict[str, Any]]) -> None:
    """Write poses.json. Register and track modes share one schema: a list of frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "poses.json").open("w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, indent=2)


def print_pose_results(pose_results: list[dict[str, Any]]) -> None:
    for item in pose_results:
        pose = np.asarray(item["pose_row_major"], dtype=np.float32).reshape(4, 4)
        t = pose[:3, 3]
        error_text = ""
        if "translation_error_m" in item:
            error_text = (
                f" terr={item['translation_error_m']:.4f}m"
                f" rerr={item['rotation_error_deg']:.2f}deg"
            )
        print(
            f"  {item['name']}: status={item['status_name']} "
            f"score={item['score']:.4f} "
            f"t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})m"
            f"{error_text}"
        )


def main() -> int:
    args = parse_args()
    try:
        raw_dataset_root = Path(args.dataset).expanduser().resolve()
        if not raw_dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {raw_dataset_root}")

        dataset_root = resolve_dataset_root(raw_dataset_root)
        models_dir = resolve_models_dir(raw_dataset_root, dataset_root, args.models_dir)
        scene_dir = scene_dir_for(dataset_root, args.scene)
        scene_camera, scene_gt = load_scene_metadata(scene_dir)

        frame_id = normalize_frame_id(args.frame)
        if str(frame_id) not in scene_camera:
            raise FileNotFoundError(f"Frame {frame_id} is not present in {scene_dir / 'scene_camera.json'}")
        if str(frame_id) not in scene_gt:
            raise FileNotFoundError(f"Frame {frame_id} is not present in {scene_dir / 'scene_gt.json'}")

        selected_objects = parse_object_selection(args.objects)
        frame_meta = normalize_frame_meta(scene_dir, frame_id, scene_gt, models_dir)
        frame_meta = filter_frame_meta_objects(frame_meta, selected_objects)

        frame, depth = load_frame(scene_dir, frame_id, scene_camera)
        objects = build_object_specs(frame_meta)
        masks = load_masks(frame_meta, objects)
        visible_pixels = count_visible_pixels(objects, masks)
        objects, frame_meta, masks, skipped_objects = filter_objects_with_masks(
            objects, frame_meta, masks, visible_pixels
        )
        if not objects:
            raise ValueError(
                "No objects had a usable visible mask. "
                "Try a different frame or a smaller --objects selection."
            )
        stats = compute_mask_stats(depth, masks, visible_pixels)

        output_dir = Path(args.output_dir).expanduser().resolve()
        save_input_artifacts(
            output_dir,
            dataset_root,
            frame,
            depth,
            objects,
            masks,
            stats,
            frame_meta,
        )

        print(f"loaded YCB-V scene {frame_meta['scene']} frame {frame_meta['frame']} from {dataset_root}")
        print(f"  image={frame.rgb.shape[1]}x{frame.rgb.shape[0]} depth_units=meters")
        print(f"  models_dir={models_dir}")
        if skipped_objects:
            print(f"  skipped objects with tiny visible masks: {', '.join(skipped_objects)}")
        for obj in objects:
            print(
                f"  {obj.name}: cad={obj.cad_path} bbox={stats[obj.name].bbox} "
                f"median_depth={stats[obj.name].median_depth_m:.3f}m"
            )
        print(f"wrote input artifacts to {output_dir}")

        register_options = RegisterOptions(
            library_path=auto_library_path(args.library),
            refine_model_path=require_existing_path(
                args.refine, "--refine", "FP_REFINE_MODEL_PATH"
            ),
            score_model_path=require_existing_path(
                args.score, "--score", "FP_SCORE_MODEL_PATH"
            ),
            engine_cache_dir=Path(args.cache).expanduser().resolve(),
            device_id=args.device_id,
            n_hypotheses=args.n_hypotheses,
            n_refine=args.n_refine,
            capture_cuda_graph=args.capture_cuda_graph,
            prepare=not args.no_prepare,
            score_output_name=args.score_output,
        )

        with MultiObjectEstimator(register_options) as estimator:
            pose_results = estimator.register(frame, objects, masks, depth)
            pose_results = enrich_pose_results(pose_results, frame_meta)
            print_pose_results(pose_results)

            frame_results: list[dict[str, Any]] = [
                {
                    "frame": frame_meta["frame"],
                    "phase": "register",
                    "objects": pose_results,
                }
            ]

            if args.mode == "register":
                write_pose_results(output_dir, frame_results)
                print(f"wrote pose results to {output_dir / 'poses.json'}")
                return 0

            tracked_frame_ids = track_frame_ids(scene_camera, frame_id, args.track_frames)
            if not tracked_frame_ids:
                raise ValueError(f"No frames after {frame_label(frame_id)} available for tracking in {scene_dir}")

            previous_gt_poses = {
                str(obj["name"]): pose_from_object_meta(obj)
                for obj in frame_meta["objects"]
            }
            reference_objects = list(frame_meta["objects"])

            for track_frame_id in tracked_frame_ids:
                track_frame, track_depth = load_frame(scene_dir, track_frame_id, scene_camera)
                track_meta = normalize_frame_meta(
                    scene_dir,
                    track_frame_id,
                    scene_gt,
                    models_dir,
                    reference_objects=reference_objects,
                    previous_gt_poses=previous_gt_poses,
                )
                track_results = estimator.track(track_frame, track_depth)
                track_results = enrich_pose_results(track_results, track_meta)
                frame_results.append(
                    {
                        "frame": frame_label(track_frame_id),
                        "phase": "track",
                        "objects": track_results,
                    }
                )
                print(f"tracked frame {frame_label(track_frame_id)}:")
                print_pose_results(track_results)
                previous_gt_poses = {
                    str(obj["name"]): pose_from_object_meta(obj)
                    for obj in track_meta["objects"]
                }

            write_pose_results(output_dir, frame_results)
            print(
                f"wrote {len(frame_results)} frame pose results to "
                f"{output_dir / 'poses.json'}"
            )
        return 0
    except (FileNotFoundError, ValueError) as exc:
        # Input/configuration problems get a one-line message; anything else
        # (library, CUDA, TensorRT failures) keeps its traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
