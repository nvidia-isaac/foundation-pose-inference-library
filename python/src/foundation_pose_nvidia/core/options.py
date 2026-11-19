# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path


def _encode_optional(value: str | Path | None) -> bytes | None:
    if value is None:
        return None
    text = str(value)
    return text.encode("utf-8") if text else None


@dataclass
class EstimatorOptions:
    """Paths, runtime options, and per-object symmetry configuration."""

    cad_path: Path
    refine_model_path: Path
    score_model_path: Path
    engine_cache_dir: Path
    device_id: int = 0
    mesh_unit_scale: float = 1.0
    refine_model_name: str | None = None
    score_model_name: str | None = None
    rendered_input_name: str | None = None
    observed_input_name: str | None = None
    refine_translation_output_name: str | None = None
    refine_rotation_output_name: str | None = None
    score_output_name: str | None = None
    prepare: bool = True
    # Per-object, object-frame symmetries; empty disables temporal filtering.
    symmetry_axes: Sequence[str] = ()

    @classmethod
    def from_env(
        cls,
        cad_path: str | Path,
        *,
        refine_model_path: str | Path | None = None,
        score_model_path: str | Path | None = None,
        engine_cache_dir: str | Path | None = None,
        device_id: int = 0,
        mesh_unit_scale: float = 1.0,
        prepare: bool = True,
        symmetry_axes: Sequence[str] = (),
    ) -> EstimatorOptions:
        return cls(
            cad_path=Path(cad_path),
            refine_model_path=Path(refine_model_path or os.environ["FP_REFINE_MODEL_PATH"]),
            score_model_path=Path(score_model_path or os.environ["FP_SCORE_MODEL_PATH"]),
            engine_cache_dir=Path(
                engine_cache_dir or os.environ.get("FP_ENGINE_CACHE_DIR", "engine_cache")
            ),
            device_id=device_id,
            mesh_unit_scale=mesh_unit_scale,
            prepare=prepare,
            symmetry_axes=symmetry_axes,
        )

    @classmethod
    def from_namespace(cls, args: argparse.Namespace, cad_path: Path) -> EstimatorOptions:
        return cls(
            cad_path=Path(cad_path),
            refine_model_path=Path(args.refine_model_path),
            score_model_path=Path(args.score_model_path),
            engine_cache_dir=Path(args.engine_cache_dir),
            device_id=int(args.device_id),
            mesh_unit_scale=float(args.mesh_unit_scale),
            refine_model_name=args.refine_model_name,
            score_model_name=args.score_model_name,
            rendered_input_name=args.rendered_input_name,
            observed_input_name=args.observed_input_name,
            refine_translation_output_name=args.refine_translation_output_name,
            refine_rotation_output_name=args.refine_rotation_output_name,
            score_output_name=args.score_output_name,
            prepare=bool(args.prepare_estimators),
            symmetry_axes=getattr(args, "symmetry_axes", ()),
        )

    def encoded_strings(self) -> list[bytes | None]:
        return [
            _encode_optional(self.cad_path),
            _encode_optional(self.refine_model_path),
            _encode_optional(self.score_model_path),
            _encode_optional(self.engine_cache_dir),
            _encode_optional(self.refine_model_name),
            _encode_optional(self.score_model_name),
            _encode_optional(self.rendered_input_name),
            _encode_optional(self.observed_input_name),
            _encode_optional(self.refine_translation_output_name),
            _encode_optional(self.refine_rotation_output_name),
            _encode_optional(self.score_output_name),
        ]
