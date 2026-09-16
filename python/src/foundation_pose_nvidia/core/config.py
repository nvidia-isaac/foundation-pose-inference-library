# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from foundation_pose_nvidia.bindings._ctypes_types import Config

if TYPE_CHECKING:
    from foundation_pose_nvidia.bindings._library import FPLibrary


class Precision(IntEnum):
    """TensorRT engine precision (maps to fp_precision_t)."""

    FP32 = 0  # strict IEEE FP32 (TF32 disabled) — most accurate/deterministic
    TF32 = 1  # allow TF32 tactics (TensorRT default)
    FP16 = 2
    BF16 = 3


@dataclass
class RuntimeConfig:
    """Python-facing runtime configuration (maps to fp_config_t)."""

    n_hypotheses: int | None = None
    n_refine_iters: int | None = None
    n_track_iters: int | None = None
    crop_ratio: float | None = None
    input_width: int | None = None
    input_height: int | None = None
    max_image_width: int | None = None
    max_image_height: int | None = None
    capture_cuda_graph: bool | None = None
    tensorrt_precision: Precision | int | None = None
    # Appended last so positional construction of the pre-existing fields is
    # unchanged (mirrors the fp_config_t append-only layout).
    batch_size: int | None = None

    def to_ctypes(self, library: FPLibrary) -> Config:
        config = library.default_config()
        for field_name, value in self.__dict__.items():
            if value is None:
                continue
            if field_name in ("capture_cuda_graph", "tensorrt_precision"):
                setattr(config, field_name, int(value))
            else:
                setattr(config, field_name, value)
        return config

    @classmethod
    def from_ctypes(cls, config: Config) -> RuntimeConfig:
        return cls(
            n_hypotheses=config.n_hypotheses,
            n_refine_iters=config.n_refine_iters,
            n_track_iters=config.n_track_iters,
            crop_ratio=config.crop_ratio,
            input_width=config.input_width,
            input_height=config.input_height,
            max_image_width=config.max_image_width,
            max_image_height=config.max_image_height,
            capture_cuda_graph=bool(config.capture_cuda_graph),
            tensorrt_precision=Precision(config.tensorrt_precision),
            batch_size=config.batch_size,
        )
