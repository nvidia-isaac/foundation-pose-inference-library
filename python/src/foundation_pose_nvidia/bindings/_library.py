# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed wrapper around the FoundationPose C ABI (ctypes)."""

from __future__ import annotations

import ctypes as C
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from foundation_pose_nvidia.bindings._ctypes_types import (
    Config,
    CreateOptions,
    ImageView,
    PoseResult,
    ReferenceImageView,
)
from foundation_pose_nvidia.bindings._loader import resolve_library_path
from foundation_pose_nvidia.exceptions import FoundationPoseError


class FPLibrary:
    """Low-level bindings to libfoundation_pose_nvidia.so."""

    def __init__(self, path: Path | str | None = None, *, print_build_info: bool = True):
        self.path = resolve_library_path(path)
        self._lib = C.CDLL(str(self.path))
        self._bind_core()
        if print_build_info:
            print(self.build_info)

    @property
    def cdll(self) -> C.CDLL:
        """Raw ctypes handle for advanced callers (e.g. group API tests)."""
        return self._lib

    @property
    def build_info(self) -> str:
        return self._lib.fp_build_info().decode("utf-8")

    def default_config(self) -> Config:
        config = Config()
        self._lib.fp_default_config(C.byref(config))
        return config

    def synchronize(self, err_size: int = 1024) -> None:
        err = C.create_string_buffer(err_size)
        if self._lib.fp_synchronize_device(err, len(err)) != 0:
            raise FoundationPoseError(err.value.decode("utf-8"))

    def set_symmetry_axes(self, handle: int, axes: Sequence[str]) -> None:
        """Configure one native handle without changing the fp_config_t ABI."""
        if not handle:
            raise FoundationPoseError("Estimator is closed")
        if isinstance(axes, (str, bytes)) or not isinstance(axes, Sequence):
            raise TypeError("symmetry_axes must be a sequence of strings, e.g. ['z_180']")
        if any(not isinstance(axis, str) or "\0" in axis for axis in axes):
            raise ValueError("symmetry_axes entries must be strings without NUL characters")
        if not hasattr(self._lib, "fp_set_symmetry_axes"):
            raise FoundationPoseError("Rebuild the native library to use symmetry filtering")
        encoded = [axis.encode("utf-8") for axis in axes]
        array = (C.c_char_p * len(encoded))(*encoded)
        err = C.create_string_buffer(4096)
        if self._lib.fp_set_symmetry_axes(handle, array, len(encoded), err, len(err)) != 0:
            raise FoundationPoseError(err.value.decode("utf-8"))

    def reset_tracking(self, handle: int, reference_pose: np.ndarray | None = None) -> None:
        if not handle:
            raise FoundationPoseError("Estimator is closed")
        if not hasattr(self._lib, "fp_reset_tracking"):
            raise FoundationPoseError("Rebuild the native library to reset tracking")
        pose = None
        pointer = None
        if reference_pose is not None:
            pose = np.ascontiguousarray(reference_pose, dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("reference_pose must be a finite 4x4 cam_T_obj in metres")
            pointer = pose.ctypes.data_as(C.POINTER(C.c_float))
        err = C.create_string_buffer(4096)
        if self._lib.fp_reset_tracking(handle, pointer, err, len(err)) != 0:
            raise FoundationPoseError(err.value.decode("utf-8"))

    def bind_group_api(self, object_count: int) -> None:
        mask_array = C.POINTER(C.c_uint8) * object_count
        lib = self._lib
        lib.fp_group_create.restype = C.c_void_p
        lib.fp_group_create.argtypes = [
            C.POINTER(CreateOptions),
            C.c_size_t,
            C.POINTER(Config),
            C.c_char_p,
            C.c_size_t,
        ]
        lib.fp_group_size.restype = C.c_size_t
        lib.fp_group_size.argtypes = [C.c_void_p]
        lib.fp_group_handle.restype = C.c_void_p
        lib.fp_group_handle.argtypes = [C.c_void_p, C.c_size_t]
        lib.fp_group_prepare.restype = C.c_int
        lib.fp_group_prepare.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.c_size_t]
        lib.fp_group_register_frame.restype = C.c_int
        lib.fp_group_register_frame.argtypes = [
            C.c_void_p,
            C.POINTER(ImageView),
            mask_array,
            C.c_int,
            C.c_int,
            C.POINTER(PoseResult),
            C.POINTER(C.c_int),
            C.c_char_p,
            C.c_size_t,
        ]
        lib.fp_group_track_frame.restype = C.c_int
        lib.fp_group_track_frame.argtypes = [
            C.c_void_p,
            C.POINTER(ImageView),
            C.POINTER(C.c_int),
            C.c_int,
            C.POINTER(PoseResult),
            C.POINTER(C.c_int),
            C.c_char_p,
            C.c_size_t,
        ]
        lib.fp_group_destroy.restype = None
        lib.fp_group_destroy.argtypes = [C.c_void_p]

    def _bind_core(self) -> None:
        lib = self._lib
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
        # Additive C APIs: older libraries remain usable when filtering/reset
        # is not requested. Existing public C structure layouts are unchanged.
        if hasattr(lib, "fp_set_symmetry_axes"):
            lib.fp_set_symmetry_axes.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_size_t, C.c_char_p, C.c_size_t]
            lib.fp_set_symmetry_axes.restype = C.c_int
        if hasattr(lib, "fp_reset_tracking"):
            lib.fp_reset_tracking.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_char_p, C.c_size_t]
            lib.fp_reset_tracking.restype = C.c_int
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


def load_library(path: Path | str | None = None, *, print_build_info: bool = True) -> FPLibrary:
    return FPLibrary(path, print_build_info=print_build_info)
