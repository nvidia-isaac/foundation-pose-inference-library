// Temporary diagnostic instrumentation, installed by make_trace_patch.py.
#pragma once

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "foundation_pose_nvidia/cuda_pipeline.hpp"
#include "foundation_pose_nvidia/exception.hpp"

namespace foundation_pose_nvidia {

class FpNumericTrace {
 public:
  FpNumericTrace(const char* phase, int batch, int iterations, int width, int height,
                 CameraIntrinsics k, const PreprocessedMesh& mesh, const Config& config,
                 cudaStream_t stream)
      : stream_(stream), config_(config), previous_(current_) {
    const char* root = std::getenv("FP_NUMERIC_TRACE_DIR");
    if (root && *root) {
      if (config.capture_cuda_graph) {
        throw FoundationPoseError("Numeric tracing requires capture_cuda_graph=false");
      }
      const char* tensors = std::getenv("FP_NUMERIC_TRACE_TENSORS");
      tensors_ = tensors && std::string(tensors) == "1";
      std::filesystem::create_directories(root);
      do {
        directory_ = std::filesystem::path(root) /
            (std::string(phase) + "_" + std::to_string(getpid()) + "_" +
             std::to_string(counter_.fetch_add(1)));
      } while (!std::filesystem::create_directory(directory_));
      std::ofstream info(directory_ / "metadata.json");
      info << std::setprecision(std::numeric_limits<float>::max_digits10);
      info << "{\"schema_version\":1,\"phase\":\"" << phase << "\",\"batch\":" << batch
           << ",\"iterations\":" << iterations << ",\"image_width\":" << width
           << ",\"image_height\":" << height << ",\"network_tensors\":"
           << (tensors_ ? "true" : "false") << ",\"config\":{";
      bool first = true;
      const auto field = [&](const char* name, auto value) {
        if (!first) info << ',';
        first = false;
        info << '"' << name << "\":" << value;
      };
#define FP_TRACE_CONFIG(name) field(#name, config.name)
      FP_TRACE_CONFIG(n_hypotheses); FP_TRACE_CONFIG(n_refine_iters);
      FP_TRACE_CONFIG(n_track_iters); FP_TRACE_CONFIG(crop_ratio);
      FP_TRACE_CONFIG(input_width); FP_TRACE_CONFIG(input_height); FP_TRACE_CONFIG(n_channels);
      FP_TRACE_CONFIG(max_image_width); FP_TRACE_CONFIG(max_image_height);
      FP_TRACE_CONFIG(min_n_views); FP_TRACE_CONFIG(inplane_step_deg);
      FP_TRACE_CONFIG(cluster_threshold_deg); FP_TRACE_CONFIG(depth_min); FP_TRACE_CONFIG(depth_max);
      FP_TRACE_CONFIG(erosion_radius); FP_TRACE_CONFIG(bilateral_radius);
      FP_TRACE_CONFIG(depth_diff_threshold); FP_TRACE_CONFIG(erosion_ratio_threshold);
      FP_TRACE_CONFIG(bilateral_sigma_d); FP_TRACE_CONFIG(bilateral_sigma_r);
      FP_TRACE_CONFIG(rotation_normalizer_rad); FP_TRACE_CONFIG(normalize_xyz);
      FP_TRACE_CONFIG(znear); FP_TRACE_CONFIG(zfar); FP_TRACE_CONFIG(ambient_weight);
      FP_TRACE_CONFIG(diffuse_weight); FP_TRACE_CONFIG(mesh_sample_points);
      FP_TRACE_CONFIG(min_voxel_size); FP_TRACE_CONFIG(voxel_downsample_ratio);
      FP_TRACE_CONFIG(model_free_sample_stride); FP_TRACE_CONFIG(model_free_min_reference_images);
      FP_TRACE_CONFIG(model_free_max_vertices); FP_TRACE_CONFIG(model_free_depth_edge_threshold);
      FP_TRACE_CONFIG(tensorrt_workspace_bytes); FP_TRACE_CONFIG(capture_cuda_graph);
#undef FP_TRACE_CONFIG
      field("tensorrt_precision", static_cast<int>(config.tensorrt_precision));
      info << "}}\n";
      info.close();
      if (!info) throw FoundationPoseError("Cannot write numeric trace metadata");
      Mat4f centered_from_original = Mat4f::identity();
      centered_from_original(0, 3) = -mesh.center.x;
      centered_from_original(1, 3) = -mesh.center.y;
      centered_from_original(2, 3) = -mesh.center.z;
      host("centered_from_original", centered_from_original.values.data(), {4, 4});
      host("mesh_diameter_m", &mesh.diameter, {1});
      const auto matrix = k.matrix();
      host("K", matrix.values.data(), {3, 3});
    }
    current_ = this;
  }

  ~FpNumericTrace() { current_ = previous_; }
  FpNumericTrace(const FpNumericTrace&) = delete;
  FpNumericTrace& operator=(const FpNumericTrace&) = delete;
  static FpNumericTrace* current() { return current_; }

  void host(const std::string& name, const float* data, std::initializer_list<int> shape) {
    if (directory_.empty()) return;
    static_assert(sizeof(float) == 4 && std::endian::native == std::endian::little);
    std::ostringstream dims;
    std::size_t count = 1;
    for (int dim : shape) { dims << dim << ", "; count *= dim; }
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         dims.str() + "), }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    std::ofstream file(directory_ / (name + ".npy"), std::ios::binary);
    file.write("\x93NUMPY\x01\x00", 8);
    const std::uint16_t size = static_cast<std::uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(header.data(), header.size());
    file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    file.close();
    if (!file) throw FoundationPoseError("Cannot write numeric trace: " + name);
  }

  void device(const std::string& name, const float* data, std::initializer_list<int> shape) {
    if (directory_.empty()) return;
    std::size_t count = 1;
    for (int dim : shape) count *= dim;
    std::vector<float> copy(count);
    checkCuda(cudaMemcpyAsync(copy.data(), data, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_), "numeric trace copy");
    checkCuda(cudaStreamSynchronize(stream_), "numeric trace synchronize");
    host(name, copy.data(), shape);
  }

  void frame(const GpuWorkspace& work, int width, int height) {
    device("depth_raw_m", work.depth_raw.as<float>(), {height, width});
    device("depth_filtered_m", work.depth_filtered.as<float>(), {height, width});
    if (tensors_) device("xyz_camera_m", work.xyz_map.as<float>(), {height, width, 3});
  }

  void inputs(const std::string& stage, const GpuWorkspace& work, const float* poses, int batch) {
    device(stage + "_poses_centered", poses, {batch, 4, 4});
    device(stage + "_crop_xyxy", work.crop_boxes.as<float>(), {batch, 4});
    if (!tensors_) return;
    device(stage + "_inputA", work.network_rendered.as<float>(),
           {batch, config_.n_channels, config_.input_height, config_.input_width});
    device(stage + "_inputB", work.network_observed.as<float>(),
           {batch, config_.n_channels, config_.input_height, config_.input_width});
  }

  void finish(const Mat4f& pose, float score, int index) {
    host("selected_pose_original", pose.values.data(), {4, 4});
    host("selected_score", &score, {1});
    const float candidate = static_cast<float>(index);
    host("selected_candidate_id", &candidate, {1});
    if (!directory_.empty()) {
      std::ofstream complete(directory_ / "complete");
      complete << "ok\n";
      complete.close();
      if (!complete) throw FoundationPoseError("Cannot complete numeric trace");
    }
  }

 private:
  inline static std::atomic<unsigned long long> counter_{0};
  inline static thread_local FpNumericTrace* current_ = nullptr;
  std::filesystem::path directory_;
  cudaStream_t stream_;
  Config config_;
  bool tensors_ = false;
  FpNumericTrace* previous_;
};

}  // namespace foundation_pose_nvidia
