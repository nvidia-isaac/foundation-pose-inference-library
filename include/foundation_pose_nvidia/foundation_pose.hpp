/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/cuda_pipeline.hpp"
#include "foundation_pose_nvidia/device_buffer.hpp"
#include "foundation_pose_nvidia/inference_runner.hpp"
#include "foundation_pose_nvidia/mesh.hpp"
#include "foundation_pose_nvidia/renderer.hpp"
#include "foundation_pose_nvidia/symmetry_filter.hpp"
#include "foundation_pose_nvidia/tensorrt_runner.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {

class FoundationPose {
 public:
  FoundationPose(PreprocessedMesh mesh, RuntimeOptions runtime_options, Config config = {});
  FoundationPose(PreprocessedMesh mesh,
                 RuntimeOptions runtime_options,
                 std::unique_ptr<IInferenceRunner> inference_runner,
                 std::unique_ptr<IRenderer> renderer,
                 Config config = {});
  ~FoundationPose();

  FoundationPose(const FoundationPose&) = delete;
  FoundationPose& operator=(const FoundationPose&) = delete;
  FoundationPose(FoundationPose&&) noexcept;
  FoundationPose& operator=(FoundationPose&&) noexcept;

  static FoundationPose createFromCadFile(const std::filesystem::path& cad_path,
                                          RuntimeOptions runtime_options,
                                          Config config = {});
  static FoundationPose createFromReferenceImages(
      std::span<const ModelFreeReferenceView> references,
      RuntimeOptions runtime_options,
      Config config = {});

  void prepareForBatch(int batch_size);

  // Per-object temporal symmetry filtering; empty axes disable it. Changes
  // preserve the last accepted pose. Registration hypotheses are unchanged.
  void setSymmetryAxes(const std::vector<std::string>& axes);
  // Clear tracking/continuity for a new episode, or restore an accepted
  // original-frame pose (metres) as both the orientation anchor and track seed.
  void resetTracking(std::optional<Mat4f> reference_pose = std::nullopt);

  PoseEstimate registerFrame(const std::uint8_t* rgb_u8,
                             const float* depth_m,
                             const std::uint8_t* mask_u8,
                             int width,
                             int height,
                             CameraIntrinsics intrinsics,
                             int n_refine = -1,
                             int n_hypotheses = -1);

  PoseEstimate trackFrame(const std::uint8_t* rgb_u8,
                          const float* depth_m,
                          int width,
                          int height,
                          CameraIntrinsics intrinsics,
                          int n_refine = -1);

  const PreprocessedMesh& mesh() const { return mesh_; }
  const Config& config() const { return config_; }

 private:
  void validateFrame(const std::uint8_t* rgb_u8,
                     const float* depth_m,
                     const std::uint8_t* mask_u8,
                     int width,
                     int height,
                     bool require_mask) const;

  void prepareNetworkInputs(const float* poses,
                            int batch_size,
                            int image_width,
                            int image_height,
                            CameraIntrinsics intrinsics,
                            float invalid_depth_threshold);

  void runRefinementLoop(float* poses,
                         int refine_iters,
                         int batch_size,
                         int image_width,
                         int image_height,
                         CameraIntrinsics intrinsics);
  void captureRefinementGraph(float* poses,
                              int batch_size,
                              int image_width,
                              int image_height,
                              CameraIntrinsics intrinsics);
  void destroyRefinementGraph() noexcept;
  PoseEstimate readBestResult();
  void storePreviousPose(const Mat4f& pose);

  PreprocessedMesh mesh_;
  RuntimeOptions runtime_options_;
  Config config_;
  int max_hypotheses_ = 0;
  int active_hypotheses_ = 0;
  std::optional<Mat4f> previous_pose_;
  SymmetryFilter symmetry_filter_;

  std::unique_ptr<CudaStream> stream_;
  std::unique_ptr<GpuWorkspace> workspace_;
  DeviceMesh device_mesh_;
  std::unique_ptr<IInferenceRunner> inference_;
  std::unique_ptr<IRenderer> renderer_;

  cudaGraph_t refine_graph_ = nullptr;
  cudaGraphExec_t refine_graph_exec_ = nullptr;
  int graph_batch_size_ = 0;
  int graph_image_width_ = 0;
  int graph_image_height_ = 0;
  CameraIntrinsics graph_intrinsics_{};
};

}  // namespace foundation_pose_nvidia
