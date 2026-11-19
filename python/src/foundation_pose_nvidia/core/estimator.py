# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level single-object estimator API."""

from __future__ import annotations

import ctypes as C
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from foundation_pose_nvidia.bindings import Config, CreateOptions, FPLibrary, PoseResult, load_library
from foundation_pose_nvidia.core.config import RuntimeConfig
from foundation_pose_nvidia.exceptions import FoundationPoseError
from foundation_pose_nvidia.core.frame import RgbdFrame
from foundation_pose_nvidia.core.options import EstimatorOptions
from foundation_pose_nvidia.core.types import PoseEstimate


class Estimator:
    """Context-managed wrapper around fp_create / fp_register_frame / fp_track_frame."""

    def __init__(
        self,
        options: EstimatorOptions,
        config: RuntimeConfig | Config,
        *,
        library: FPLibrary | None = None,
        obj_id: int = 0,
        prepare_batch: int | None = None,
        prepare_track_batch: bool = False,
    ):
        self.handle = None
        self._symmetry_axes: tuple[str, ...] = ()
        self._owns_library = library is None
        self.library = library or load_library(print_build_info=True)
        self.lib = self.library.cdll
        self.obj_id = int(obj_id)
        self.setup_timings: list[dict[str, Any]] = []
        self._strings = options.encoded_strings()

        if isinstance(config, RuntimeConfig):
            self._config = config.to_ctypes(self.library)
        else:
            self._config = config

        create_options = CreateOptions(
            C.c_char_p(self._strings[0]),
            int(options.device_id),
            C.c_float(options.mesh_unit_scale),
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
        self.handle = self.lib.fp_create(
            C.byref(create_options), C.byref(self._config), err, len(err)
        )
        create_elapsed = time.perf_counter() - start
        self.setup_timings.append(
            {
                "obj_id": self.obj_id,
                "type": "create_estimator",
                "time_s": create_elapsed,
                "cad_path": str(options.cad_path),
            }
        )
        if not self.handle:
            raise FoundationPoseError(err.value.decode("utf-8"))

        try:
            if options.symmetry_axes:
                self.set_symmetry_axes(options.symmetry_axes)
            if options.prepare and prepare_batch is not None:
                self._prepare(int(prepare_batch))
            elif prepare_track_batch:
                self._prepare(1)
        except BaseException:
            self.close()
            raise
            

    @property
    def config(self) -> Config:
        return self._config

    @property
    def symmetry_axes(self) -> tuple[str, ...]:
        return self._symmetry_axes

    def set_symmetry_axes(self, axes: Sequence[str]) -> None:
        """Select temporal equivalents, e.g. ['z_180']; [] disables filtering.

        Retains the previous accepted pose across reconfiguration/registration.
        Numeric angles are explicit alternatives; *_full samples every 30 deg.
        """
        self.library.set_symmetry_axes(self.handle, axes)
        self._symmetry_axes = tuple(axes)

    def reset_tracking(self, reference_pose: np.ndarray | None = None) -> None:
        """Clear tracking, or restore an accepted original-frame cam_T_obj.

        Use a saved pose to undo an output rejected by an external quality gate.
        None starts a new episode and requires registration before tracking.
        """
        self.library.reset_tracking(self.handle, reference_pose)

    def _prepare(self, batch_size: int) -> None:
        err = C.create_string_buffer(4096)
        start = time.perf_counter()
        status = self.lib.fp_prepare(self.handle, batch_size, err, len(err))
        self.library.synchronize()
        prepare_elapsed = time.perf_counter() - start
        self.setup_timings.append(
            {
                "obj_id": self.obj_id,
                "type": "prepare_estimator",
                "time_s": prepare_elapsed,
                "batch_size": batch_size,
            }
        )
        if status != 0:
            raise FoundationPoseError(err.value.decode("utf-8"))

    def close(self) -> None:
        if self.handle:
            self.lib.fp_destroy(self.handle)
            self.handle = None

    __del__ = close

    def __enter__(self) -> Estimator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register(
        self,
        frame: RgbdFrame | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        n_refine: int = -1,
        n_hypotheses: int = -1,
    ) -> PoseEstimate:
        if isinstance(frame, RgbdFrame):
            if frame.mask is None:
                raise ValueError("register requires a mask")
            return self._register_arrays(
                frame.rgb, frame.depth_m, frame.mask, frame.intrinsics, n_refine, n_hypotheses
            )
        else:
            rgb, depth_m, mask, intrinsics = frame
            return self._register_arrays(rgb, depth_m, mask, intrinsics, n_refine, n_hypotheses)

    def track(
        self,
        frame: RgbdFrame | tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        n_refine: int = -1,
    ) -> PoseEstimate:
        if isinstance(frame, RgbdFrame):
            view = frame.to_image_view(with_mask=False)
        else:
            rgb, depth_m, k = frame
            view = RgbdFrame(rgb, depth_m, k).to_image_view(with_mask=False)
        return self._track_view(view, n_refine)

    def register_arrays(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        n_refine: int,
        n_hypotheses: int,
    ) -> tuple[tuple[np.ndarray, float], float]:
        """Benchmark-compatible register API (returns ((pose, score), elapsed))."""
        result = self._register_arrays(rgb, depth_m, mask, intrinsics, n_refine, n_hypotheses)
        return (result.pose, result.score), result.elapsed_s

    def track_arrays(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        n_refine: int,
    ) -> tuple[np.ndarray, float]:
        """Benchmark-compatible track API (returns (pose, elapsed))."""
        result = self.track((rgb, depth_m, intrinsics), n_refine=n_refine)
        return result.pose, result.elapsed_s

    def _register_arrays(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        n_refine: int,
        n_hypotheses: int,
    ) -> PoseEstimate:
        view = RgbdFrame(rgb, depth_m, intrinsics, mask).to_image_view(with_mask=True)
        result = PoseResult()
        err = C.create_string_buffer(4096)
        self.library.synchronize()
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
        self.library.synchronize()
        elapsed = time.perf_counter() - start
        if status != 0:
            raise FoundationPoseError(err.value.decode("utf-8"))
        return PoseEstimate.from_ctypes(result.pose_row_major, result.score, elapsed)

    def _track_view(self, view: Any, n_refine: int) -> PoseEstimate:
        result = PoseResult()
        err = C.create_string_buffer(4096)
        self.library.synchronize()
        start = time.perf_counter()
        status = self.lib.fp_track_frame(
            self.handle,
            C.byref(view),
            n_refine,
            C.byref(result),
            err,
            len(err),
        )
        self.library.synchronize()
        elapsed = time.perf_counter() - start
        if status != 0:
            raise FoundationPoseError(err.value.decode("utf-8"))
        return PoseEstimate.from_ctypes(result.pose_row_major, result.score, elapsed)
