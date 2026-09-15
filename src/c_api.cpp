/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#include "foundation_pose_nvidia/foundation_pose.hpp"
#include "foundation_pose_nvidia/exception.hpp"

using foundation_pose_nvidia::CameraIntrinsics;
using foundation_pose_nvidia::Config;
using foundation_pose_nvidia::FoundationPose;
using foundation_pose_nvidia::FoundationPoseError;
using foundation_pose_nvidia::Mat4f;
using foundation_pose_nvidia::ModelFreeReferenceView;
using foundation_pose_nvidia::RuntimeOptions;
using foundation_pose_nvidia::TensorrtPrecision;

struct fp_handle {
  std::unique_ptr<FoundationPose> estimator;
};

struct fp_group {
  std::vector<std::unique_ptr<fp_handle>> handles;
  std::vector<int> device_ids;
};

namespace {

void writeError(char* buffer, size_t size, std::string_view message) {
  if (buffer == nullptr || size == 0) {
    return;
  }
  const std::size_t count = std::min(size - 1, message.size());
  std::memcpy(buffer, message.data(), count);
  buffer[count] = '\0';
}

std::string strOrEmpty(const char* value) {
  return value == nullptr ? std::string{} : std::string(value);
}

Config toConfig(const fp_config_t* input) {
  Config config;
  if (input == nullptr) {
    return config;
  }
  if (input->n_hypotheses > 0) {
    config.n_hypotheses = input->n_hypotheses;
  }
  if (input->n_refine_iters >= 0) {
    config.n_refine_iters = input->n_refine_iters;
  }
  if (input->n_track_iters >= 0) {
    config.n_track_iters = input->n_track_iters;
  }
  if (input->crop_ratio > 0.0f) {
    config.crop_ratio = input->crop_ratio;
  }
  if (input->input_width > 0) {
    config.input_width = input->input_width;
  }
  if (input->input_height > 0) {
    config.input_height = input->input_height;
  }
  if (input->max_image_width > 0) {
    config.max_image_width = input->max_image_width;
  }
  if (input->max_image_height > 0) {
    config.max_image_height = input->max_image_height;
  }
  config.capture_cuda_graph = input->capture_cuda_graph != 0;
  if (input->model_free_sample_stride > 0) {
    config.model_free_sample_stride = input->model_free_sample_stride;
  }
  if (input->model_free_max_vertices > 0) {
    config.model_free_max_vertices = input->model_free_max_vertices;
  }
  if (input->model_free_depth_edge_threshold > 0.0f) {
    config.model_free_depth_edge_threshold = input->model_free_depth_edge_threshold;
  }
  using enum TensorrtPrecision;
  switch (input->tensorrt_precision) {
    case FP_PRECISION_TF32:
      config.tensorrt_precision = kTF32;
      break;
    case FP_PRECISION_FP16:
      config.tensorrt_precision = kFP16;
      break;
    case FP_PRECISION_BF16:
      config.tensorrt_precision = kBF16;
      break;
    case FP_PRECISION_FP32:
    default:
      config.tensorrt_precision = kFP32;
      break;
  }
  if (input->batch_size > 0) {
    config.batch_size = input->batch_size;
  }
  return config;
}

RuntimeOptions toRuntimeOptions(const fp_create_options_t& input) {
  RuntimeOptions options;
  options.device_id = input.device_id;
  if (input.mesh_unit_scale > 0.0f) {
    options.mesh_unit_scale = input.mesh_unit_scale;
  }
  options.refine_model_path = strOrEmpty(input.refine_model_path);
  options.score_model_path = strOrEmpty(input.score_model_path);
  options.engine_cache_dir = strOrEmpty(input.engine_cache_dir);
  if (input.refine_model_name != nullptr) {
    options.refine_model_name = input.refine_model_name;
  }
  if (input.score_model_name != nullptr) {
    options.score_model_name = input.score_model_name;
  }
  if (input.rendered_input_name != nullptr) {
    options.rendered_input_name = input.rendered_input_name;
  }
  if (input.observed_input_name != nullptr) {
    options.observed_input_name = input.observed_input_name;
  }
  if (input.refine_translation_output_name != nullptr) {
    options.refine_translation_output_name = input.refine_translation_output_name;
  }
  if (input.refine_rotation_output_name != nullptr) {
    options.refine_rotation_output_name = input.refine_rotation_output_name;
  }
  if (input.score_output_name != nullptr) {
    options.score_output_name = input.score_output_name;
  }
  return options;
}

CameraIntrinsics makeIntrinsics(const fp_image_view_t& view) {
  if (view.k_row_major == nullptr) {
    throw FoundationPoseError("k_row_major is required");
  }
  return CameraIntrinsics::fromRowMajor(view.k_row_major);
}

Mat4f makeMat4OrIdentity(const float* row_major) {
  Mat4f out = Mat4f::identity();
  if (row_major == nullptr) {
    return out;
  }
  for (int i = 0; i < 16; ++i) {
    out.values[static_cast<std::size_t>(i)] = row_major[i];
  }
  return out;
}

ModelFreeReferenceView makeReferenceView(const fp_reference_image_view_t& view) {
  if (view.k_row_major == nullptr) {
    throw FoundationPoseError("Reference k_row_major is required");
  }
  ModelFreeReferenceView out;
  out.width = view.width;
  out.height = view.height;
  out.rgb_u8 = view.rgb_u8;
  out.depth_m = view.depth_m;
  out.mask_u8 = view.mask_u8;
  out.intrinsics = CameraIntrinsics::fromRowMajor(view.k_row_major);
  out.camera_to_world = makeMat4OrIdentity(view.camera_to_world_row_major);
  return out;
}

void writeResult(const foundation_pose_nvidia::PoseEstimate& estimate,
                 fp_pose_result_t* result) {
  if (result == nullptr) {
    return;
  }
  for (int i = 0; i < 16; ++i) {
    result->pose_row_major[i] = estimate.pose.values[static_cast<std::size_t>(i)];
  }
  result->score = estimate.score;
}

// Runs one worker thread per non-skipped object. Each estimator owns its own
// CUDA stream, so per-object GPU work overlaps; CUDA device selection is
// per-thread state, so every worker re-selects its estimator's device.
// `evaluate` runs with the object's device current and returns the object's
// PoseEstimate. Skipped objects keep FP_OBJECT_SKIPPED and untouched results.
// Returns the number of failed objects; the first failure message (annotated
// with the object index) goes to the error buffer.
int runGroup(fp_group* group,
             const std::vector<bool>& skip,
             fp_pose_result_t* results,
             int* statuses,
             char* error_buffer,
             size_t error_buffer_size,
             const std::function<foundation_pose_nvidia::PoseEstimate(FoundationPose&, size_t)>&
                 evaluate) {
  const size_t count = group->handles.size();
  std::vector<int> status(count, FP_OBJECT_OK);
  std::vector<std::string> errors(count);
  std::vector<std::jthread> workers;
  workers.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (skip[i]) {
      status[i] = FP_OBJECT_SKIPPED;
      continue;
    }
    workers.emplace_back([group, i, results, &status, &errors, &evaluate]() {
      try {
        cudaError_t cuda_status = cudaSetDevice(group->device_ids[i]);
        if (cuda_status != cudaSuccess) {
          throw FoundationPoseError(cudaGetErrorString(cuda_status));
        }
        const foundation_pose_nvidia::PoseEstimate estimate =
            evaluate(*group->handles[i]->estimator, i);
        if (results != nullptr) {
          writeResult(estimate, results + i);
        }
      } catch (const std::exception& e) {
        status[i] = FP_OBJECT_FAILED;
        errors[i] = e.what();
      }
    });
  }
  for (std::jthread& worker : workers) {
    worker.join();
  }
  int failures = 0;
  for (size_t i = 0; i < count; ++i) {
    if (statuses != nullptr) {
      statuses[i] = status[i];
    }
    if (status[i] == FP_OBJECT_FAILED) {
      if (failures == 0) {
        writeError(error_buffer, error_buffer_size,
                   std::format("object {}: {}", i, errors[i]));
      }
      ++failures;
    }
  }
  if (failures == 0) {
    writeError(error_buffer, error_buffer_size, "");
  }
  return failures;
}

}  // namespace

extern "C" {

void fp_default_config(fp_config_t* config) {
  if (config == nullptr) {
    return;
  }
  Config defaults;
  config->n_hypotheses = defaults.n_hypotheses;
  config->n_refine_iters = defaults.n_refine_iters;
  config->n_track_iters = defaults.n_track_iters;
  config->crop_ratio = defaults.crop_ratio;
  config->input_width = defaults.input_width;
  config->input_height = defaults.input_height;
  config->max_image_width = defaults.max_image_width;
  config->max_image_height = defaults.max_image_height;
  config->capture_cuda_graph = defaults.capture_cuda_graph ? 1 : 0;
  config->model_free_sample_stride = defaults.model_free_sample_stride;
  config->model_free_max_vertices = defaults.model_free_max_vertices;
  config->model_free_depth_edge_threshold = defaults.model_free_depth_edge_threshold;
  config->tensorrt_precision = static_cast<int>(defaults.tensorrt_precision);
  config->batch_size = defaults.batch_size;
}

fp_handle_t* fp_create(const fp_create_options_t* options,
                       const fp_config_t* config,
                       char* error_buffer,
                       size_t error_buffer_size) {
  try {
    if (options == nullptr || options->cad_path == nullptr) {
      throw FoundationPoseError("cad_path is required");
    }
    auto handle = std::make_unique<fp_handle>();
    handle->estimator = std::make_unique<FoundationPose>(
        FoundationPose::createFromCadFile(options->cad_path,
                                          toRuntimeOptions(*options),
                                          toConfig(config)));
    writeError(error_buffer, error_buffer_size, "");
    return handle.release();
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return nullptr;
  }
}

fp_handle_t* fp_create_model_free(const fp_reference_image_view_t* references,
                                  size_t reference_count,
                                  const fp_create_options_t* options,
                                  const fp_config_t* config,
                                  char* error_buffer,
                                  size_t error_buffer_size) {
  try {
    if (references == nullptr || reference_count == 0) {
      throw FoundationPoseError("At least one model-free reference image is required");
    }
    if (options == nullptr) {
      throw FoundationPoseError("create options are required");
    }
    std::vector<ModelFreeReferenceView> views;
    views.reserve(reference_count);
    for (size_t i = 0; i < reference_count; ++i) {
      views.push_back(makeReferenceView(references[i]));
    }
    auto handle = std::make_unique<fp_handle>();
    handle->estimator = std::make_unique<FoundationPose>(
        FoundationPose::createFromReferenceImages(views, toRuntimeOptions(*options),
                                                  toConfig(config)));
    writeError(error_buffer, error_buffer_size, "");
    return handle.release();
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return nullptr;
  }
}

int fp_prepare(fp_handle_t* handle,
               int batch_size,
               char* error_buffer,
               size_t error_buffer_size) {
  try {
    if (handle == nullptr || !handle->estimator) {
      throw FoundationPoseError("Invalid handle");
    }
    handle->estimator->prepareForBatch(batch_size);
    writeError(error_buffer, error_buffer_size, "");
    return 0;
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return 1;
  }
}

void fp_destroy(fp_handle_t* handle) {
  delete handle;
}

int fp_register_frame(fp_handle_t* handle,
                      const fp_image_view_t* image,
                      int n_refine,
                      int n_hypotheses,
                      fp_pose_result_t* result,
                      char* error_buffer,
                      size_t error_buffer_size) {
  try {
    if (handle == nullptr || handle->estimator == nullptr || image == nullptr) {
      throw FoundationPoseError("Invalid handle or image");
    }
    writeResult(handle->estimator->registerFrame(
                    image->rgb_u8, image->depth_m, image->mask_u8, image->width,
                    image->height, makeIntrinsics(*image), n_refine, n_hypotheses),
                result);
    writeError(error_buffer, error_buffer_size, "");
    return 0;
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return 1;
  }
}

int fp_track_frame(fp_handle_t* handle,
                   const fp_image_view_t* image,
                   int n_refine,
                   fp_pose_result_t* result,
                   char* error_buffer,
                   size_t error_buffer_size) {
  try {
    if (handle == nullptr || handle->estimator == nullptr || image == nullptr) {
      throw FoundationPoseError("Invalid handle or image");
    }
    writeResult(handle->estimator->trackFrame(image->rgb_u8, image->depth_m,
                                              image->width, image->height,
                                              makeIntrinsics(*image), n_refine),
                result);
    writeError(error_buffer, error_buffer_size, "");
    return 0;
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return 1;
  }
}

fp_group_t* fp_group_create(const fp_create_options_t* objects,
                            size_t object_count,
                            const fp_config_t* config,
                            char* error_buffer,
                            size_t error_buffer_size) {
  try {
    if (objects == nullptr || object_count == 0) {
      throw FoundationPoseError("At least one object is required");
    }
    auto group = std::make_unique<fp_group>();
    group->handles.reserve(object_count);
    group->device_ids.reserve(object_count);
    for (size_t i = 0; i < object_count; ++i) {
      if (objects[i].cad_path == nullptr) {
        throw FoundationPoseError(std::format("object {}: cad_path is required", i));
      }
      auto handle = std::make_unique<fp_handle>();
      handle->estimator = std::make_unique<FoundationPose>(
          FoundationPose::createFromCadFile(objects[i].cad_path,
                                            toRuntimeOptions(objects[i]),
                                            toConfig(config)));
      group->handles.push_back(std::move(handle));
      group->device_ids.push_back(objects[i].device_id);
    }
    writeError(error_buffer, error_buffer_size, "");
    return group.release();
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return nullptr;
  }
}

size_t fp_group_size(const fp_group_t* group) {
  return group == nullptr ? 0 : group->handles.size();
}

fp_handle_t* fp_group_handle(fp_group_t* group, size_t index) {
  if (group == nullptr || index >= group->handles.size()) {
    return nullptr;
  }
  return group->handles[index].get();
}

int fp_group_prepare(fp_group_t* group,
                     int batch_size,
                     char* error_buffer,
                     size_t error_buffer_size) {
  if (group == nullptr) {
    writeError(error_buffer, error_buffer_size, "Invalid group");
    return 1;
  }
  const std::vector<bool> skip(group->handles.size(), false);
  return runGroup(group, skip, nullptr, nullptr, error_buffer, error_buffer_size,
                  [batch_size](FoundationPose& estimator, size_t) {
                    estimator.prepareForBatch(batch_size);
                    return foundation_pose_nvidia::PoseEstimate{};
                  });
}

int fp_group_register_frame(fp_group_t* group,
                            const fp_image_view_t* image,
                            const uint8_t* const* masks,
                            int n_refine,
                            int n_hypotheses,
                            fp_pose_result_t* results,
                            int* statuses,
                            char* error_buffer,
                            size_t error_buffer_size) {
  try {
    if (group == nullptr || image == nullptr || masks == nullptr || results == nullptr) {
      throw FoundationPoseError("Invalid group, image, masks, or results");
    }
    const CameraIntrinsics intrinsics = makeIntrinsics(*image);
    std::vector<bool> skip(group->handles.size());
    for (size_t i = 0; i < group->handles.size(); ++i) {
      skip[i] = masks[i] == nullptr;
    }
    return runGroup(
        group, skip, results, statuses, error_buffer, error_buffer_size,
        [image, masks, intrinsics, n_refine, n_hypotheses](FoundationPose& estimator,
                                                           size_t i) {
          return estimator.registerFrame(image->rgb_u8, image->depth_m, masks[i],
                                         image->width, image->height, intrinsics,
                                         n_refine, n_hypotheses);
        });
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return group == nullptr ? 1 : static_cast<int>(group->handles.size());
  }
}

int fp_group_track_frame(fp_group_t* group,
                         const fp_image_view_t* image,
                         const int* active,
                         int n_refine,
                         fp_pose_result_t* results,
                         int* statuses,
                         char* error_buffer,
                         size_t error_buffer_size) {
  try {
    if (group == nullptr || image == nullptr || results == nullptr) {
      throw FoundationPoseError("Invalid group, image, or results");
    }
    const CameraIntrinsics intrinsics = makeIntrinsics(*image);
    std::vector<bool> skip(group->handles.size());
    for (size_t i = 0; i < group->handles.size(); ++i) {
      skip[i] = active != nullptr && active[i] == 0;
    }
    return runGroup(group, skip, results, statuses, error_buffer, error_buffer_size,
                    [image, intrinsics, n_refine](FoundationPose& estimator, size_t) {
                      return estimator.trackFrame(image->rgb_u8, image->depth_m,
                                                  image->width, image->height,
                                                  intrinsics, n_refine);
                    });
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return group == nullptr ? 1 : static_cast<int>(group->handles.size());
  }
}

void fp_group_destroy(fp_group_t* group) {
  delete group;
}

int fp_synchronize_device(char* error_buffer, size_t error_buffer_size) {
  try {
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      throw FoundationPoseError(cudaGetErrorString(status));
    }
    writeError(error_buffer, error_buffer_size, "");
    return 0;
  } catch (const std::exception& e) {
    writeError(error_buffer, error_buffer_size, e.what());
    return 1;
  }
}

const char* fp_build_info(void) {
  return "foundation-pose nvidia-wide (CUDA 12 / TensorRT 10 FP32 / nvdiffrast CUDA)";
}

}  // extern "C"
