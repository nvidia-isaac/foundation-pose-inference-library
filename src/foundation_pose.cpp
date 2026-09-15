/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/foundation_pose.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include "foundation_pose_nvidia/builtin_raster_renderer.hpp"
#include "foundation_pose_nvidia/exception.hpp"
#include "pipeline_utils.hpp"

namespace foundation_pose_nvidia {
namespace {

CameraIntrinsics validateIntrinsics(CameraIntrinsics intrinsics) {
  if (intrinsics.fx <= 0.0f || intrinsics.fy <= 0.0f) {
    throw FoundationPoseError("Camera intrinsics must have positive focal lengths");
  }
  return intrinsics;
}

bool sameIntrinsics(CameraIntrinsics a, CameraIntrinsics b) {
  return a.fx == b.fx && a.fy == b.fy && a.cx == b.cx && a.cy == b.cy;
}

// Default-renderer selection. The built-in CUDA rasterizer is the default.
// Setting FP_RENDERER=nvdiffrast loads the optional nvdiffrast renderer
// plugin (libfoundation_pose_nvdiffrast.so, built with
// -DFOUNDATION_POSE_WITH_NVDIFFRAST=ON) via dlopen; the main library never
// links nvdiffrast code. The plugin handle is intentionally never closed.
using PluginRendererFactory = IRenderer* (*)(const Config*, int, int);

std::unique_ptr<IRenderer> makeDefaultRenderer(const Config& config,
                                               int device_id,
                                               int max_batch) {
  const char* choice = std::getenv("FP_RENDERER");
  if (choice == nullptr || std::string_view(choice) != "nvdiffrast") {
    return std::make_unique<BuiltinRasterRenderer>(config, device_id, max_batch);
  }
  static void* plugin = dlopen("libfoundation_pose_nvdiffrast.so", RTLD_NOW | RTLD_LOCAL);
  if (plugin == nullptr) {
    throw FoundationPoseError(
        std::string("FP_RENDERER=nvdiffrast requested but the renderer plugin could not "
                    "be loaded (build with -DFOUNDATION_POSE_WITH_NVDIFFRAST=ON and keep "
                    "libfoundation_pose_nvdiffrast.so next to the main library): ") +
        dlerror());
  }
  auto factory = reinterpret_cast<PluginRendererFactory>(
      dlsym(plugin, "fp_plugin_create_renderer"));
  if (factory == nullptr) {
    throw FoundationPoseError("nvdiffrast renderer plugin is missing fp_plugin_create_renderer");
  }
  return std::unique_ptr<IRenderer>(factory(&config, device_id, max_batch));
}

int maxHypothesesFor(const Config& config) {
  return std::max(1, config.n_hypotheses);
}

Vec3f transformPoint(const Mat4f& m, Vec3f p) {
  return {
      m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2) * p.z + m(0, 3),
      m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2) * p.z + m(1, 3),
      m(2, 0) * p.x + m(2, 1) * p.y + m(2, 2) * p.z + m(2, 3),
  };
}

bool validReferenceDepth(float z, const Config& config) {
  return std::isfinite(z) && z >= config.depth_min && z < config.depth_max;
}

bool smoothTriangle(float za, float zb, float zc, float threshold) {
  return std::abs(za - zb) <= threshold && std::abs(za - zc) <= threshold &&
         std::abs(zb - zc) <= threshold;
}

Mesh reconstructReferenceMesh(std::span<const ModelFreeReferenceView> references,
                              const Config& config) {
  if (static_cast<int>(references.size()) < config.model_free_min_reference_images) {
    throw FoundationPoseError("Model-free mode requires at least one reference image");
  }

  std::size_t total_pixels = 0;
  for (const ModelFreeReferenceView& ref : references) {
    validateIntrinsics(ref.intrinsics);
    if (ref.width <= 0 || ref.height <= 0 || ref.rgb_u8 == nullptr ||
        ref.depth_m == nullptr || ref.mask_u8 == nullptr) {
      throw FoundationPoseError("Each model-free reference needs RGB, depth, mask, and dimensions");
    }
    total_pixels += static_cast<std::size_t>(ref.width) * ref.height;
  }

  const int max_vertices = std::max(config.model_free_max_vertices, 4);
  const int adaptive_stride = std::max(
      1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(total_pixels) /
                                             static_cast<double>(max_vertices)))));
  const int stride = std::max(config.model_free_sample_stride, adaptive_stride);

  Mesh mesh;
  mesh.source_path = "model_free_references";
  for (const ModelFreeReferenceView& ref : references) {
    std::vector<int> grid_index(static_cast<std::size_t>(ref.width) * ref.height, -1);
    std::vector<float> sampled_depth(static_cast<std::size_t>(ref.width) * ref.height,
                                     0.0f);

    for (int y = 0; y < ref.height; y += stride) {
      for (int x = 0; x < ref.width; x += stride) {
        const std::size_t pixel = static_cast<std::size_t>(y) * ref.width + x;
        const float z = ref.depth_m[pixel];
        if (ref.mask_u8[pixel] == 0 || !validReferenceDepth(z, config)) {
          continue;
        }
        const Vec3f cam{
            (static_cast<float>(x) - ref.intrinsics.cx) * z / ref.intrinsics.fx,
            (static_cast<float>(y) - ref.intrinsics.cy) * z / ref.intrinsics.fy,
            z,
        };
        if (mesh.vertices.size() >= static_cast<std::size_t>(max_vertices)) {
          break;
        }
        const std::size_t rgb = pixel * 3;
        mesh.vertices.push_back(transformPoint(ref.camera_to_world, cam));
        mesh.vertex_colors.push_back(
            Vec3u8{ref.rgb_u8[rgb + 0], ref.rgb_u8[rgb + 1], ref.rgb_u8[rgb + 2]});
        grid_index[pixel] = static_cast<int>(mesh.vertices.size() - 1);
        sampled_depth[pixel] = z;
      }
      if (mesh.vertices.size() >= static_cast<std::size_t>(max_vertices)) {
        break;
      }
    }

    for (int y = 0; y + stride < ref.height; y += stride) {
      for (int x = 0; x + stride < ref.width; x += stride) {
        const std::size_t ia = static_cast<std::size_t>(y) * ref.width + x;
        const std::size_t ib = static_cast<std::size_t>(y) * ref.width + x + stride;
        const std::size_t ic = static_cast<std::size_t>(y + stride) * ref.width + x;
        const std::size_t id =
            static_cast<std::size_t>(y + stride) * ref.width + x + stride;
        const int a = grid_index[ia];
        const int b = grid_index[ib];
        const int c = grid_index[ic];
        const int d = grid_index[id];
        if (a >= 0 && b >= 0 && c >= 0 &&
            smoothTriangle(sampled_depth[ia], sampled_depth[ib], sampled_depth[ic],
                           config.model_free_depth_edge_threshold)) {
          mesh.faces.push_back({static_cast<std::uint32_t>(a),
                                static_cast<std::uint32_t>(b),
                                static_cast<std::uint32_t>(c)});
        }
        if (b >= 0 && d >= 0 && c >= 0 &&
            smoothTriangle(sampled_depth[ib], sampled_depth[id], sampled_depth[ic],
                           config.model_free_depth_edge_threshold)) {
          mesh.faces.push_back({static_cast<std::uint32_t>(b),
                                static_cast<std::uint32_t>(d),
                                static_cast<std::uint32_t>(c)});
        }
      }
    }
  }

  if (mesh.vertices.size() < 4 || mesh.faces.empty()) {
    throw FoundationPoseError(
        "Model-free reconstruction produced too little geometry; check masks and depth");
  }
  computeVertexNormals(mesh);
  return mesh;
}

}  // namespace

FoundationPose::FoundationPose(PreprocessedMesh mesh,
                               RuntimeOptions runtime_options,
                               Config config)
    : FoundationPose(std::move(mesh),
                     runtime_options,
                     std::make_unique<TensorRtRunner>(runtime_options, config,
                                                      maxHypothesesFor(config)),
                     makeDefaultRenderer(config, runtime_options.device_id,
                                         maxHypothesesFor(config)),
                     config) {}

FoundationPose::FoundationPose(PreprocessedMesh mesh,
                               RuntimeOptions runtime_options,
                               std::unique_ptr<IInferenceRunner> inference_runner,
                               std::unique_ptr<IRenderer> renderer,
                               Config config)
    : mesh_(std::move(mesh)),
      runtime_options_(std::move(runtime_options)),
      config_(config),
      max_hypotheses_(maxHypothesesFor(config_)) {
  if (!inference_runner || !renderer) {
    throw FoundationPoseError("FoundationPose requires inference and renderer backends");
  }
  // Add this check to avoid duplicated call to the cudaSetDevice is the current device is already the desired one for run time.
  int current_device = -1;
  checkCuda(cudaGetDevice(&current_device), "cudaGetDevice");
  if (current_device != runtime_options_.device_id) {
    checkCuda(cudaSetDevice(runtime_options_.device_id), "cudaSetDevice");
  }
  stream_ = std::make_unique<CudaStream>();
  workspace_ = std::make_unique<GpuWorkspace>(config_, max_hypotheses_);
  inference_ = std::move(inference_runner);
  renderer_ = std::move(renderer);

  uploadMeshToDevice(mesh_.centered_mesh, device_mesh_, stream_->get());
  renderer_->reserveForMesh(device_mesh_.num_vertices, device_mesh_.num_faces);

  std::vector<Mat3f> rotations = detail::makeRotationGrid(config_);
  if (rotations.empty()) {
    throw FoundationPoseError("Rotation grid generation returned no hypotheses");
  }
  if (static_cast<int>(rotations.size()) > max_hypotheses_) {
    rotations.resize(static_cast<std::size_t>(max_hypotheses_));
  }
  uploadRotationGridToDevice(rotations, *workspace_, stream_->get());
  stream_->synchronize();
}

FoundationPose::~FoundationPose() {
  destroyRefinementGraph();
}

FoundationPose::FoundationPose(FoundationPose&& other) noexcept
    : mesh_(std::move(other.mesh_)),
      runtime_options_(std::move(other.runtime_options_)),
      config_(other.config_),
      max_hypotheses_(std::exchange(other.max_hypotheses_, 0)),
      active_hypotheses_(std::exchange(other.active_hypotheses_, 0)),
      has_previous_pose_(std::exchange(other.has_previous_pose_, false)),
      stream_(std::move(other.stream_)),
      workspace_(std::move(other.workspace_)),
      device_mesh_(std::move(other.device_mesh_)),
      inference_(std::move(other.inference_)),
      renderer_(std::move(other.renderer_)),
      refine_graph_(std::exchange(other.refine_graph_, nullptr)),
      refine_graph_exec_(std::exchange(other.refine_graph_exec_, nullptr)),
      graph_batch_size_(std::exchange(other.graph_batch_size_, 0)),
      graph_image_width_(std::exchange(other.graph_image_width_, 0)),
      graph_image_height_(std::exchange(other.graph_image_height_, 0)),
      graph_intrinsics_(other.graph_intrinsics_) {}

FoundationPose& FoundationPose::operator=(FoundationPose&& other) noexcept {
  if (this != &other) {
    destroyRefinementGraph();
    mesh_ = std::move(other.mesh_);
    runtime_options_ = std::move(other.runtime_options_);
    config_ = other.config_;
    max_hypotheses_ = std::exchange(other.max_hypotheses_, 0);
    active_hypotheses_ = std::exchange(other.active_hypotheses_, 0);
    has_previous_pose_ = std::exchange(other.has_previous_pose_, false);
    stream_ = std::move(other.stream_);
    workspace_ = std::move(other.workspace_);
    device_mesh_ = std::move(other.device_mesh_);
    inference_ = std::move(other.inference_);
    renderer_ = std::move(other.renderer_);
    refine_graph_ = std::exchange(other.refine_graph_, nullptr);
    refine_graph_exec_ = std::exchange(other.refine_graph_exec_, nullptr);
    graph_batch_size_ = std::exchange(other.graph_batch_size_, 0);
    graph_image_width_ = std::exchange(other.graph_image_width_, 0);
    graph_image_height_ = std::exchange(other.graph_image_height_, 0);
    graph_intrinsics_ = other.graph_intrinsics_;
  }
  return *this;
}

FoundationPose FoundationPose::createFromCadFile(const std::filesystem::path& cad_path,
                                                 RuntimeOptions runtime_options,
                                                 Config config) {
  Mesh mesh = loadMesh(cad_path);
  if (runtime_options.mesh_unit_scale != 1.0f) {
    for (Vec3f& vertex : mesh.vertices) {
      vertex.x *= runtime_options.mesh_unit_scale;
      vertex.y *= runtime_options.mesh_unit_scale;
      vertex.z *= runtime_options.mesh_unit_scale;
    }
  }
  PreprocessedMesh preprocessed = preprocessMesh(mesh, config);
  return FoundationPose(std::move(preprocessed), std::move(runtime_options), config);
}

FoundationPose FoundationPose::createFromReferenceImages(
    std::span<const ModelFreeReferenceView> references,
    RuntimeOptions runtime_options,
    Config config) {
  Mesh mesh = reconstructReferenceMesh(references, config);
  PreprocessedMesh preprocessed = preprocessMesh(mesh, config);
  return FoundationPose(std::move(preprocessed), std::move(runtime_options), config);
}

void FoundationPose::prepareForBatch(int batch_size) {
  if (batch_size <= 0 || batch_size > max_hypotheses_) {
    throw FoundationPoseError("prepare batch size exceeds configured maximum");
  }
  inference_->prepareForBatch(batch_size);
}

PoseEstimate FoundationPose::registerFrame(const std::uint8_t* rgb_u8,
                                           const float* depth_m,
                                           const std::uint8_t* mask_u8,
                                           int width,
                                           int height,
                                           CameraIntrinsics intrinsics,
                                           int n_refine,
                                           int n_hypotheses) {
  validateFrame(rgb_u8, depth_m, mask_u8, width, height, true);
  intrinsics = validateIntrinsics(intrinsics);
  const int refine_iters = n_refine >= 0 ? n_refine : config_.n_refine_iters;
  active_hypotheses_ =
      n_hypotheses > 0 ? std::min(n_hypotheses, workspace_->rotation_count)
                       : std::min(config_.n_hypotheses, workspace_->rotation_count);
  active_hypotheses_ = std::min(active_hypotheses_, max_hypotheses_);
  if (active_hypotheses_ <= 0) {
    throw FoundationPoseError("No active hypotheses available");
  }
  const int batch_size = config_.batch_size > 0 ? config_.batch_size : active_hypotheses_;
  const int effective_bs = std::min(active_hypotheses_, batch_size);
  if (effective_bs > 0 && active_hypotheses_ % effective_bs != 0) {
    throw FoundationPoseError("active_hypotheses (" + std::to_string(active_hypotheses_) +
                              ") must be divisible by batch_size (" +
                              std::to_string(effective_bs) + ")");
  }
  // Micro-batching and CUDA graph capture are mutually exclusive. Chunked
  // refinement rebinds the shared TensorRT execution context between chunks and
  // must synchronize the stream in between, which CUDA forbids while the stream
  // is capturing: the capture would fail deep inside enqueueRefine with an
  // opaque error. Reject the combination here, where the message can name both
  // settings. Tracking is unaffected (batch size 1 never chunks).
  if (config_.capture_cuda_graph && effective_bs < active_hypotheses_) {
    throw FoundationPoseError(
        "capture_cuda_graph is not supported with inference micro-batching: batch_size (" +
        std::to_string(effective_bs) + ") is smaller than active_hypotheses (" +
        std::to_string(active_hypotheses_) +
        "), so refinement would run in multiple TensorRT chunks that cannot be captured "
        "into a CUDA graph. Raise batch_size to at least active_hypotheses, or disable "
        "capture_cuda_graph.");
  }

  uploadFrameToDevice(rgb_u8, depth_m, mask_u8, width, height, *workspace_,
                      stream_->get());
  runDepthPreprocess(*workspace_, width, height, config_, stream_->get());
  runDepthToXyz(*workspace_, width, height, intrinsics, config_, stream_->get());
  runInitialTranslation(*workspace_, width, height, intrinsics, config_, stream_->get());
  seedPosesFromRotations(*workspace_, active_hypotheses_, stream_->get());

  float* poses = workspace_->poses.as<float>();
  runRefinementLoop(poses, refine_iters, active_hypotheses_, width, height, intrinsics);

  prepareNetworkInputs(poses, active_hypotheses_, width, height, intrinsics, 0.1f);
  inference_->enqueueScore(workspace_->network_rendered.as<float>(),
                           workspace_->network_observed.as<float>(),
                           workspace_->scores.as<float>(), active_hypotheses_,
                           stream_->get());
  selectBestPoseOnDevice(poses, workspace_->scores.as<float>(), active_hypotheses_,
                         mesh_.center, workspace_->best_result.as<DevicePoseResult>(),
                         stream_->get());

  DevicePoseResult host_result;
  checkCuda(cudaMemcpyAsync(&host_result, workspace_->best_result.data(), sizeof(host_result),
                            cudaMemcpyDeviceToHost, stream_->get()),
            "cudaMemcpyAsync best result");
  stream_->synchronize();
  checkCuda(cudaMemcpyAsync(workspace_->previous_centered_pose.data(),
                            poses + host_result.index * 16, 16 * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_->get()),
            "cudaMemcpyAsync previous pose");
  has_previous_pose_ = true;
  PoseEstimate estimate;
  estimate.pose = host_result.pose;
  estimate.score = host_result.score;
  return estimate;
}

PoseEstimate FoundationPose::trackFrame(const std::uint8_t* rgb_u8,
                                        const float* depth_m,
                                        int width,
                                        int height,
                                        CameraIntrinsics intrinsics,
                                        int n_refine) {
  validateFrame(rgb_u8, depth_m, nullptr, width, height, false);
  intrinsics = validateIntrinsics(intrinsics);
  if (!has_previous_pose_) {
    throw FoundationPoseError("No previous pose. Call registerFrame before trackFrame.");
  }
  const int refine_iters = n_refine >= 0 ? n_refine : config_.n_track_iters;
  uploadFrameToDevice(rgb_u8, depth_m, nullptr, width, height, *workspace_,
                      stream_->get());
  runDepthPreprocess(*workspace_, width, height, config_, stream_->get());
  runDepthToXyz(*workspace_, width, height, intrinsics, config_, stream_->get());
  copyPoseOnDevice(workspace_->previous_centered_pose.as<float>(),
                   workspace_->poses.as<float>(), stream_->get());

  float* poses = workspace_->poses.as<float>();
  runRefinementLoop(poses, refine_iters, 1, width, height, intrinsics);

  const float score_one = 1.0f;
  checkCuda(cudaMemcpyAsync(workspace_->scores.data(), &score_one, sizeof(float),
                            cudaMemcpyHostToDevice, stream_->get()),
            "cudaMemcpyAsync tracking score");
  selectBestPoseOnDevice(poses, workspace_->scores.as<float>(), 1, mesh_.center,
                         workspace_->best_result.as<DevicePoseResult>(),
                         stream_->get());
  copyPoseOnDevice(poses, workspace_->previous_centered_pose.as<float>(), stream_->get());
  return readBestResult();
}

void FoundationPose::validateFrame(const std::uint8_t* rgb_u8,
                                   const float* depth_m,
                                   const std::uint8_t* mask_u8,
                                   int width,
                                   int height,
                                   bool require_mask) const {
  if (rgb_u8 == nullptr || depth_m == nullptr || (require_mask && mask_u8 == nullptr)) {
    throw FoundationPoseError("RGB, depth, and mask pointers are required");
  }
  if (width <= 0 || height <= 0) {
    throw FoundationPoseError("Image dimensions must be positive");
  }
  if (width > config_.max_image_width || height > config_.max_image_height) {
    throw FoundationPoseError("Image exceeds configured max_image_width/max_image_height");
  }
}

void FoundationPose::prepareNetworkInputs(const float* poses,
                                          int batch_size,
                                          int image_width,
                                          int image_height,
                                          CameraIntrinsics intrinsics,
                                          float invalid_depth_threshold) {
  computeCropBoxesOnDevice(poses, batch_size, intrinsics, mesh_.diameter, config_,
                           workspace_->crop_boxes.as<float>(), stream_->get());
  renderer_->render(device_mesh_, poses, workspace_->crop_boxes.as<float>(), batch_size,
                    intrinsics, image_width, image_height,
                    workspace_->rendered_rgb.as<float3>(),
                    workspace_->rendered_xyz.as<float3>(), stream_->get());
  prepareNetworkInputsOnDevice(*workspace_, poses, batch_size, image_width, image_height,
                               mesh_.diameter, invalid_depth_threshold, stream_->get());
}

void FoundationPose::runRefinementLoop(float* poses,
                                       int refine_iters,
                                       int batch_size,
                                       int image_width,
                                       int image_height,
                                       CameraIntrinsics intrinsics) {
  if (refine_iters <= 0) {
    return;
  }
  if (config_.capture_cuda_graph) {
    const bool graph_stale =
        refine_graph_exec_ == nullptr || graph_batch_size_ != batch_size ||
        graph_image_width_ != image_width || graph_image_height_ != image_height ||
        !sameIntrinsics(graph_intrinsics_, intrinsics);
    if (graph_stale) {
      captureRefinementGraph(poses, batch_size, image_width, image_height, intrinsics);
    }
    for (int iter = 0; iter < refine_iters; ++iter) {
      checkCuda(cudaGraphLaunch(refine_graph_exec_, stream_->get()),
                "cudaGraphLaunch refinement graph");
    }
    return;
  }

  for (int iter = 0; iter < refine_iters; ++iter) {
    prepareNetworkInputs(poses, batch_size, image_width, image_height, intrinsics,
                         config_.depth_min);
    inference_->enqueueRefine(workspace_->network_rendered.as<float>(),
                              workspace_->network_observed.as<float>(),
                              workspace_->delta_translation.as<float>(),
                              workspace_->delta_rotation.as<float>(), batch_size,
                              stream_->get());
    applyPoseDeltasOnDevice(poses, workspace_->delta_translation.as<float>(),
                            workspace_->delta_rotation.as<float>(), batch_size,
                            mesh_.diameter, config_, stream_->get());
  }
}

void FoundationPose::captureRefinementGraph(float* poses,
                                            int batch_size,
                                            int image_width,
                                            int image_height,
                                            CameraIntrinsics intrinsics) {
  destroyRefinementGraph();
  inference_->prepareForBatch(batch_size);
  checkCuda(cudaStreamBeginCapture(stream_->get(), cudaStreamCaptureModeGlobal),
            "cudaStreamBeginCapture refinement graph");
  bool capture_open = true;
  try {
    prepareNetworkInputs(poses, batch_size, image_width, image_height, intrinsics,
                         config_.depth_min);
    inference_->enqueueRefine(workspace_->network_rendered.as<float>(),
                              workspace_->network_observed.as<float>(),
                              workspace_->delta_translation.as<float>(),
                              workspace_->delta_rotation.as<float>(), batch_size,
                              stream_->get());
    applyPoseDeltasOnDevice(poses, workspace_->delta_translation.as<float>(),
                            workspace_->delta_rotation.as<float>(), batch_size,
                            mesh_.diameter, config_, stream_->get());
    checkCuda(cudaStreamEndCapture(stream_->get(), &refine_graph_),
              "cudaStreamEndCapture refinement graph");
    capture_open = false;
    checkCuda(cudaGraphInstantiate(&refine_graph_exec_, refine_graph_, 0),
              "cudaGraphInstantiate refinement graph");
  } catch (...) {
    if (capture_open) {
      cudaGraph_t abandoned_graph = nullptr;
      cudaStreamEndCapture(stream_->get(), &abandoned_graph);
      if (abandoned_graph != nullptr) {
        cudaGraphDestroy(abandoned_graph);
      }
    }
    destroyRefinementGraph();
    throw;
  }
  graph_batch_size_ = batch_size;
  graph_image_width_ = image_width;
  graph_image_height_ = image_height;
  graph_intrinsics_ = intrinsics;
}

void FoundationPose::destroyRefinementGraph() noexcept {
  if (refine_graph_exec_ != nullptr) {
    cudaGraphExecDestroy(refine_graph_exec_);
    refine_graph_exec_ = nullptr;
  }
  if (refine_graph_ != nullptr) {
    cudaGraphDestroy(refine_graph_);
    refine_graph_ = nullptr;
  }
  graph_batch_size_ = 0;
  graph_image_width_ = 0;
  graph_image_height_ = 0;
}

PoseEstimate FoundationPose::readBestResult() {
  DevicePoseResult result;
  checkCuda(cudaMemcpyAsync(&result, workspace_->best_result.data(), sizeof(result),
                            cudaMemcpyDeviceToHost, stream_->get()),
            "cudaMemcpyAsync best result");
  stream_->synchronize();
  PoseEstimate estimate;
  estimate.pose = result.pose;
  estimate.score = result.score;
  return estimate;
}

}  // namespace foundation_pose_nvidia
