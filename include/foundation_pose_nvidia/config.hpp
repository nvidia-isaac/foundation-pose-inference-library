/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

namespace foundation_pose_nvidia {

// TensorRT compute precision for the RefineNet/ScoreNet engines.
//   kFP32 : strict IEEE FP32 (TF32 disabled) — most accurate/deterministic.
//   kTF32 : allow TF32 tensor-core tactics (TensorRT default).
//   kFP16 : allow FP16 tactics.
//   kBF16 : allow BF16 tactics.
enum class TensorrtPrecision { kFP32 = 0, kTF32 = 1, kFP16 = 2, kBF16 = 3 };

struct Config {
  int n_hypotheses = 252;
  int n_refine_iters = 5;
  int n_track_iters = 2;
  float crop_ratio = 1.2f;
  int input_width = 160;
  int input_height = 160;
  int n_channels = 6;
  int max_image_width = 1280;
  int max_image_height = 720;

  int min_n_views = 40;
  float inplane_step_deg = 60.0f;
  float cluster_threshold_deg = 30.0f;

  float depth_min = 0.001f;
  float depth_max = 100.0f;
  int erosion_radius = 2;
  int bilateral_radius = 2;
  float depth_diff_threshold = 0.001f;
  float erosion_ratio_threshold = 0.8f;
  float bilateral_sigma_d = 2.0f;
  float bilateral_sigma_r = 100000.0f;

  float rotation_normalizer_rad = 0.3490658503988659f;  // 20 degrees.
  bool normalize_xyz = true;

  float znear = 0.001f;
  float zfar = 100.0f;
  float ambient_weight = 0.8f;
  float diffuse_weight = 0.5f;

  int mesh_sample_points = 10000;
  float min_voxel_size = 0.003f;
  float voxel_downsample_ratio = 20.0f;

  int model_free_sample_stride = 4;
  int model_free_min_reference_images = 1;
  int model_free_max_vertices = 250000;
  float model_free_depth_edge_threshold = 0.02f;

  std::size_t tensorrt_workspace_bytes = 8ULL << 30;
  bool capture_cuda_graph = false;
  // Default matches TensorRT's own default (TF32 allowed). Set kFP32 for strict
  // IEEE FP32 (recommended for cross-GPU accuracy parity, e.g. x86 vs Jetson).
  TensorrtPrecision tensorrt_precision = TensorrtPrecision::kTF32;
  int batch_size = 252;  // RefineNet mini-batch chunk size (effective bs = min(n_hypotheses, batch_size); ScoreNet is not micro-batched)
};

struct RuntimeOptions {
  int device_id = 0;
  float mesh_unit_scale = 1.0f;

  std::string refine_model_path;
  std::string score_model_path;
  std::string engine_cache_dir = "engine_cache";
  std::string refine_model_name = "refine";
  std::string score_model_name = "score";
  std::string rendered_input_name = "inputA";
  std::string observed_input_name = "inputB";
  std::string refine_translation_output_name = "trans";
  std::string refine_rotation_output_name = "rot";
  std::string score_output_name = "score";
};

}  // namespace foundation_pose_nvidia
