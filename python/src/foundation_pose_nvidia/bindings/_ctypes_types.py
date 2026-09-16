# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ctypes Structure definitions matching foundation_pose_nvidia/c_api.h."""

from __future__ import annotations

import ctypes as C


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
