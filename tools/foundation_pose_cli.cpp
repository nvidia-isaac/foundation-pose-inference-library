/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "foundation_pose_nvidia/c_api.h"

namespace {

void printUsage() {
  std::cout << R"(FoundationPose NVIDIA diagnostic CLI

Usage:
  foundation_pose_nvidia_cli --help
  foundation_pose_nvidia_cli --version
  foundation_pose_nvidia_cli --smoke --mode register --cad <mesh.{ply|obj}> \
      --refine <refiner.onnx> --score <score.onnx> [options]
  foundation_pose_nvidia_cli --smoke --mode track --cad <mesh.{ply|obj}> \
      --refine <refiner.onnx> --score <score.onnx> [options]
  foundation_pose_nvidia_cli --model-free-smoke \
      --refine <refiner.onnx> --score <score.onnx> [options]

What this CLI does:
  Runs small synthetic 160x160 RGB-D smoke tests against the shared library.
  It is meant to verify CUDA/TensorRT/nvdiffrast wiring, model loading, engine
  cache creation, mesh loading, register mode, track mode, and model-free handle
  creation. It is not an accuracy benchmark; use benchmarks/run_benchmark.py for
  BOP evaluation.

Required arguments:
  --refine <path>            FoundationPose refiner ONNX model.
  --score <path>             FoundationPose scorer ONNX model.
  --cad <path>               CAD mesh for --smoke. Required unless using
                             --model-free-smoke.

Modes:
  --smoke                    Create a model-based estimator from --cad.
  --model-free-smoke         Create a model-free estimator from one synthetic
                             RGB-D-mask reference view.
  --mode register            Register one synthetic frame. This is the default.
  --mode track               Register once, then run one track update.

Options:
  --cache <dir>              TensorRT engine cache directory.
                             Default: engine_cache
  --mesh-unit-scale <float>  Scale applied to CAD vertices before preprocessing.
                             Use 0.001 for BOP YCB-V meshes in millimeters.
                             Default: 0.001
  --symmetry-axis <spec>     Temporal symmetry filter, e.g. z_180 or x_full.
                             Repeat for multiple alternatives. Default: disabled.
  -h, --help                 Show this help text.
  --version                  Print build/backend information.

Examples:
  foundation_pose_nvidia_cli --smoke --mode register \
      --cad /data/BOP/ycbv/models/obj_000001.ply \
      --refine /models/foundationpose_deployable_v1.0/refiner_net.onnx \
      --score /models/foundationpose_deployable_v1.0/score_net.onnx \
      --cache ./engine_cache

  foundation_pose_nvidia_cli --smoke --mode track \
      --cad /data/BOP/ycbv/models/obj_000001.ply \
      --refine /models/foundationpose_deployable_v1.0/refiner_net.onnx \
      --score /models/foundationpose_deployable_v1.0/score_net.onnx

  foundation_pose_nvidia_cli --model-free-smoke \
      --refine /models/foundationpose_deployable_v1.0/refiner_net.onnx \
      --score /models/foundationpose_deployable_v1.0/score_net.onnx

Output:
  score=<float>
  pose_row_major=<16 float row-major camera_T_object values>
)";
}

const char* cstrOrNull(const std::string& value) {
  return value.empty() ? nullptr : value.c_str();
}

int runSmoke(const std::string& cad,
             const std::string& refine,
             const std::string& score,
             const std::string& cache,
             float mesh_unit_scale,
             std::string_view mode,
             bool model_free,
             const std::vector<std::string>& symmetry_axes) {
  fp_config_t config{};
  fp_default_config(&config);
  config.n_hypotheses = 12;
  config.n_refine_iters = 1;
  config.n_track_iters = 1;
  config.input_width = 160;
  config.input_height = 160;
  config.max_image_width = 160;
  config.max_image_height = 160;
  config.model_free_sample_stride = 8;

  fp_create_options_t options{};
  options.cad_path = cstrOrNull(cad);
  options.device_id = 0;
  options.mesh_unit_scale = mesh_unit_scale;
  options.refine_model_path = refine.c_str();
  options.score_model_path = score.c_str();
  options.engine_cache_dir = cstrOrNull(cache);
  options.rendered_input_name = "inputA";
  options.observed_input_name = "inputB";
  options.refine_translation_output_name = "trans";
  options.refine_rotation_output_name = "rot";
  options.score_output_name = "score";

  std::array<char, 4096> error{};
  constexpr int width = 160;
  constexpr int height = 160;
  std::vector<std::uint8_t> rgb(width * height * 3, 96);
  std::vector<float> depth(width * height, 0.0f);
  std::vector<std::uint8_t> mask(width * height, 0);
  for (int y = 48; y < 112; ++y) {
    for (int x = 48; x < 112; ++x) {
      const int idx = y * width + x;
      rgb[idx * 3 + 0] = 180;
      rgb[idx * 3 + 1] = 130;
      rgb[idx * 3 + 2] = 90;
      depth[idx] = 0.7f;
      mask[idx] = 255;
    }
  }
  const std::array<float, 9> k = {180.0f, 0.0f, 80.0f, 0.0f, 180.0f,
                                  80.0f, 0.0f, 0.0f, 1.0f};

  fp_handle_t* handle = nullptr;
  if (model_free) {
    const std::array<float, 16> camera_to_world = {1.0f, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 1.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 1.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 1.0f};
    fp_reference_image_view_t reference{};
    reference.width = width;
    reference.height = height;
    reference.rgb_u8 = rgb.data();
    reference.depth_m = depth.data();
    reference.mask_u8 = mask.data();
    reference.k_row_major = k.data();
    reference.camera_to_world_row_major = camera_to_world.data();
    handle =
        fp_create_model_free(&reference, 1, &options, &config, error.data(), error.size());
  } else {
    handle = fp_create(&options, &config, error.data(), error.size());
  }
  if (handle == nullptr) {
    std::cerr << error.data() << '\n';
    return 1;
  }
  std::vector<const char*> axes;
  for (const auto& axis : symmetry_axes) axes.push_back(axis.c_str());
  if (fp_set_symmetry_axes(handle, axes.data(), axes.size(), error.data(), error.size()) != 0) {
    std::cerr << error.data() << '\n';
    fp_destroy(handle);
    return 1;
  }

  fp_image_view_t view{};
  view.width = width;
  view.height = height;
  view.rgb_u8 = rgb.data();
  view.depth_m = depth.data();
  view.mask_u8 = mask.data();
  view.k_row_major = k.data();
  fp_pose_result_t result{};
  int status =
      fp_register_frame(handle, &view, config.n_refine_iters, config.n_hypotheses,
                        &result, error.data(), error.size());
  if (status == 0 && mode == "track") {
    status = fp_track_frame(handle, &view, config.n_track_iters, &result, error.data(),
                            error.size());
  }
  fp_destroy(handle);
  if (status != 0) {
    std::cerr << error.data() << '\n';
    return 1;
  }
  std::cout << "score=" << result.score << "\npose_row_major=";
  for (int i = 0; i < 16; ++i) {
    if (i != 0) {
      std::cout << ' ';
    }
    std::cout << result.pose_row_major[i];
  }
  std::cout << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool smoke = false;
  bool model_free_smoke = false;
  std::string cad;
  std::string refine;
  std::string score;
  std::string cache = "engine_cache";
  std::string mode = "register";
  float mesh_unit_scale = 0.001f;
  std::vector<std::string> symmetry_axes;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--version") {
      std::cout << fp_build_info() << '\n';
      return 0;
    }
    if (arg == "--smoke") {
      smoke = true;
    } else if (arg == "--model-free-smoke") {
      model_free_smoke = true;
    } else if (arg == "--cad" && i + 1 < argc) {
      cad = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--refine" && i + 1 < argc) {
      refine = argv[++i];
    } else if (arg == "--score" && i + 1 < argc) {
      score = argv[++i];
    } else if (arg == "--cache" && i + 1 < argc) {
      cache = argv[++i];
    } else if (arg == "--mesh-unit-scale" && i + 1 < argc) {
      mesh_unit_scale = std::stof(argv[++i]);
    } else if (arg == "--symmetry-axis" && i + 1 < argc) {
      symmetry_axes.emplace_back(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      printUsage();
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      printUsage();
      return 2;
    }
  }
  if (!smoke && !model_free_smoke) {
    printUsage();
    return 2;
  }
  if (mode != "register" && mode != "track") {
    std::cerr << "--mode must be register or track\n";
    return 2;
  }
  if (!model_free_smoke && cad.empty()) {
    std::cerr << "--smoke requires --cad unless --model-free-smoke is used\n";
    return 2;
  }
  if (refine.empty() || score.empty()) {
    std::cerr << "Smoke tests require --refine and --score\n";
    return 2;
  }
  return runSmoke(cad, refine, score, cache, mesh_unit_scale, mode, model_free_smoke, symmetry_axes);
}
