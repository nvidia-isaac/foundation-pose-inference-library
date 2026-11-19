# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level multi-object group registration API."""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, replace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from foundation_pose_nvidia.bindings import CreateOptions, PoseResult, load_library
from foundation_pose_nvidia.core.frame import RgbdFrame

STATUS_NAMES = {
    0: "OK",
    1: "FAILED",
    2: "SKIPPED",
}


@dataclass
class ObjectSpec:
    name: str
    cad_path: Path
    mesh_unit_scale: float = 1.0
    symmetry_axes: Sequence[str] = ()


@dataclass(frozen=True)
class MultiObjectPoseEstimate:
    """Object pose returned by multi-object register/track."""

    object_spec: ObjectSpec
    pose: np.ndarray
    score: float
    status: int
    status_name: str


@dataclass
class RegisterOptions:
    library_path: Path
    refine_model_path: Path
    score_model_path: Path
    engine_cache_dir: Path
    device_id: int = 0
    n_hypotheses: int = 64
    n_refine: int = 3
    capture_cuda_graph: bool = False
    prepare: bool = True
    refine_translation_output_name: str = "trans"
    refine_rotation_output_name: str = "rot"
    score_output_name: str = "score"


def _c_bytes(path: Path | str | None) -> bytes | None:
    if path is None:
        return None
    return str(path).encode("utf-8")


class MultiObjectEstimator:
    """Application-level wrapper around the FoundationPose fp_group_* API."""

    def __init__(self, options: RegisterOptions):
        self.options = options
        self.options.engine_cache_dir.mkdir(parents=True, exist_ok=True)
        self.library = None
        self.lib = None
        self.group = None
        self.objects: list[ObjectSpec] = []
        self.config = None
        self._strings: list[bytes] = []

    def close(self) -> None:
        group = getattr(self, "group", None)
        lib = getattr(self, "lib", None)
        if group and lib:
            lib.fp_group_destroy(group)
        self.group = None
        self.lib = None
        self.library = None
        self.objects = []
        self.config = None
        self._strings = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> MultiObjectEstimator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_group(self, frame: Any, objects: list[ObjectSpec]) -> None:
        if self.group:
            normalized = [replace(obj, symmetry_axes=tuple(obj.symmetry_axes)) for obj in objects]
            if normalized != self.objects:
                raise ValueError("Cannot reuse a multi-object group with different objects or symmetry settings")
            return

        self.library = load_library(self.options.library_path, print_build_info=False)
        self.library.bind_group_api(len(objects))
        self.lib = self.library.cdll
        self.config = self.library.default_config()
        self.config.max_image_width = int(frame.rgb.shape[1])
        self.config.max_image_height = int(frame.rgb.shape[0])
        if self.options.n_hypotheses > 0:
            self.config.n_hypotheses = self.options.n_hypotheses
        if self.options.n_refine >= 0:
            self.config.n_refine_iters = self.options.n_refine
        self.config.capture_cuda_graph = 1 if self.options.capture_cuda_graph else 0

        self.objects = [replace(obj, symmetry_axes=tuple(obj.symmetry_axes)) for obj in objects]
        create_options = (CreateOptions * len(objects))()
        self._strings = []
        for idx, obj in enumerate(objects):
            cad = _c_bytes(obj.cad_path)
            refine = _c_bytes(self.options.refine_model_path)
            score = _c_bytes(self.options.score_model_path)
            cache = _c_bytes(self.options.engine_cache_dir)
            refine_trans_output = self.options.refine_translation_output_name.encode("utf-8")
            refine_rot_output = self.options.refine_rotation_output_name.encode("utf-8")
            score_output = self.options.score_output_name.encode("utf-8")
            self._strings.extend(  # type: ignore[arg-type]
                [cad, refine, score, cache, refine_trans_output, refine_rot_output, score_output]
            )
            create_options[idx].cad_path = cad
            create_options[idx].device_id = self.options.device_id
            create_options[idx].mesh_unit_scale = obj.mesh_unit_scale
            create_options[idx].refine_model_path = refine
            create_options[idx].score_model_path = score
            create_options[idx].engine_cache_dir = cache
            create_options[idx].rendered_input_name = b"inputA"
            create_options[idx].observed_input_name = b"inputB"
            create_options[idx].refine_translation_output_name = refine_trans_output
            create_options[idx].refine_rotation_output_name = refine_rot_output
            create_options[idx].score_output_name = score_output

        err = C.create_string_buffer(4096)
        self.group = self.lib.fp_group_create(
            create_options, len(objects), C.byref(self.config), err, len(err)
        )
        if not self.group:
            raise RuntimeError(
                f"fp_group_create failed: {err.value.decode(errors='replace')}"
            )
        try:
            for idx, obj in enumerate(objects):
                if obj.symmetry_axes:
                    handle = self.lib.fp_group_handle(self.group, idx)
                    self.library.set_symmetry_axes(handle, obj.symmetry_axes)
        except BaseException:
            self.close()
            raise
        size = self.lib.fp_group_size(self.group)
        if size != len(objects):
            raise RuntimeError(f"fp_group_size returned {size}, expected {len(objects)}")
        if self.options.prepare:
            batch = (
                self.config.n_hypotheses
                if self.options.n_hypotheses <= 0
                else self.options.n_hypotheses
            )
            rc = self.lib.fp_group_prepare(self.group, batch, err, len(err))
            if rc != 0:
                raise RuntimeError(
                    f"fp_group_prepare failed for {rc} object(s): "
                    f"{err.value.decode(errors='replace')}"
                )

    def _pose_estimates_from_ctypes(
        self,
        results: Any,
        statuses: Any,
    ) -> list[MultiObjectPoseEstimate]:
        estimates: list[MultiObjectPoseEstimate] = []
        for idx, obj in enumerate(self.objects):
            status = int(statuses[idx])
            estimates.append(
                MultiObjectPoseEstimate(
                    object_spec=obj,
                    pose=np.asarray(results[idx].pose_row_major, dtype=np.float32)
                    .reshape(4, 4)
                    .copy(),
                    score=float(results[idx].score),
                    status=status,
                    status_name=STATUS_NAMES.get(status, "UNKNOWN"),
                )
            )
        return estimates

    def register(
        self,
        frame: Any,
        objects: list[ObjectSpec],
        masks: dict[str, np.ndarray],
        depth: np.ndarray,
    ) -> list[MultiObjectPoseEstimate]:
        self._create_group(frame, objects)
        if not self.group or not self.lib:
            raise RuntimeError("Multi-object group was not created")

        view = RgbdFrame(frame.rgb, depth, frame.k).to_image_view(with_mask=False)
        mask_arrays = [
            np.ascontiguousarray(masks[obj.name], dtype=np.uint8) for obj in self.objects
        ]
        mask_array_type = C.POINTER(C.c_uint8) * len(self.objects)
        mask_ptrs = mask_array_type(
            *[
                mask.ctypes.data_as(C.POINTER(C.c_uint8))
                for mask in mask_arrays
            ]
        )
        results = (PoseResult * len(self.objects))()
        statuses = (C.c_int * len(self.objects))()
        err = C.create_string_buffer(4096)
        rc = self.lib.fp_group_register_frame(
            self.group,
            C.byref(view),
            mask_ptrs,
            self.options.n_refine,
            self.options.n_hypotheses,
            results,
            statuses,
            err,
            len(err),
        )
        if rc != 0:
            raise RuntimeError(
                f"fp_group_register_frame failed for {rc} object(s): "
                f"{err.value.decode(errors='replace')}"
            )

        return self._pose_estimates_from_ctypes(results, statuses)

    def track(
        self,
        frame: Any,
        depth: np.ndarray,
        *,
        active: dict[str, bool] | list[bool] | None = None,
        n_refine: int = -1,
    ) -> list[MultiObjectPoseEstimate]:
        if not self.group or not self.lib:
            raise RuntimeError("track requires a prior successful register call")

        view = RgbdFrame(frame.rgb, depth, frame.k).to_image_view(with_mask=False)
        active_ptr = None
        active_array = None
        if active is not None:
            if isinstance(active, dict):
                flags = [1 if active.get(obj.name, True) else 0 for obj in self.objects]
            else:
                flags = [1 if value else 0 for value in active]
                if len(flags) != len(self.objects):
                    raise ValueError(
                        f"active has {len(flags)} entries, expected {len(self.objects)}"
                    )
            active_array = (C.c_int * len(self.objects))(*flags)
            active_ptr = active_array

        results = (PoseResult * len(self.objects))()
        statuses = (C.c_int * len(self.objects))()
        err = C.create_string_buffer(4096)
        rc = self.lib.fp_group_track_frame(
            self.group,
            C.byref(view),
            active_ptr,
            n_refine,
            results,
            statuses,
            err,
            len(err),
        )
        if rc != 0:
            raise RuntimeError(
                f"fp_group_track_frame failed for {rc} object(s): "
                f"{err.value.decode(errors='replace')}"
            )

        return self._pose_estimates_from_ctypes(results, statuses)
