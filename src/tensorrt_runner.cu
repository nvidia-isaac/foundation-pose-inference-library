/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/tensorrt_runner.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "foundation_pose_nvidia/device_buffer.hpp"
#include "foundation_pose_nvidia/exception.hpp"

namespace foundation_pose_nvidia {
namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // TensorRT owns process-wide logging style; stderr keeps the C ABI quiet
      // unless a caller has a failing build or runtime issue to diagnose.
      std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

constexpr int kMaxTensorRtEngineBatch = 256;

enum class EngineKind { Refine, Score };

// Short tag used in the engine-cache filename so plans self-invalidate when the
// precision changes (the cache key must reflect the actual build precision).
const char* precisionTag(TensorrtPrecision precision) {
  switch (precision) {
    case TensorrtPrecision::kTF32:
      return "tf32";
    case TensorrtPrecision::kFP16:
      return "fp16";
    case TensorrtPrecision::kBF16:
      return "bf16";
    case TensorrtPrecision::kFP32:
    default:
      return "fp32";
  }
}

// Configure the TensorRT builder for the requested precision. TensorRT enables
// TF32 by default, so strict FP32 must explicitly clear it.
void applyPrecision(nvinfer1::IBuilderConfig& build_config, TensorrtPrecision precision) {
  switch (precision) {
    case TensorrtPrecision::kFP32:
      build_config.clearFlag(nvinfer1::BuilderFlag::kTF32);
      break;
    case TensorrtPrecision::kTF32:
      // TensorRT default already allows TF32; nothing to set.
      break;
    case TensorrtPrecision::kFP16:
      build_config.setFlag(nvinfer1::BuilderFlag::kFP16);
      break;
    case TensorrtPrecision::kBF16:
      build_config.setFlag(nvinfer1::BuilderFlag::kBF16);
      break;
  }
}

template <class T>
struct TrtDeleter {
  void operator()(T* value) const {
    delete value;
  }
};

template <class T>
using TrtUnique = std::unique_ptr<T, TrtDeleter<T>>;

std::vector<char> readFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw FoundationPoseError("Failed to open file: " + path.string());
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<std::size_t>(size));
  in.read(data.data(), size);
  if (!in) {
    throw FoundationPoseError("Failed to read file: " + path.string());
  }
  return data;
}

std::uint64_t fnv1a64(const std::vector<char>& data) {
  std::uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : data) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

nvinfer1::Dims4 inputDims(int batch, int channels, const Config& config) {
  return nvinfer1::Dims4{batch, channels, config.input_height, config.input_width};
}

void setProfileForInput(nvinfer1::IOptimizationProfile& profile,
                        const char* name,
                        int channels,
                        const Config& config,
                        int max_batch) {
  if (!profile.setDimensions(name, nvinfer1::OptProfileSelector::kMIN,
                             inputDims(1, channels, config))) {
    throw FoundationPoseError(std::string("Failed to set TensorRT min dims for ") + name);
  }
  if (!profile.setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
                             inputDims(max_batch, channels, config))) {
    throw FoundationPoseError(std::string("Failed to set TensorRT opt dims for ") + name);
  }
  if (!profile.setDimensions(name, nvinfer1::OptProfileSelector::kMAX,
                             inputDims(max_batch, channels, config))) {
    throw FoundationPoseError(std::string("Failed to set TensorRT max dims for ") + name);
  }
}

__global__ void splitCombinedRefineKernel(const float* combined,
                                          float* translation,
                                          float* rotation,
                                          int batch_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_size) {
    return;
  }
  translation[i * 3 + 0] = combined[i * 6 + 0];
  translation[i * 3 + 1] = combined[i * 6 + 1];
  translation[i * 3 + 2] = combined[i * 6 + 2];
  rotation[i * 3 + 0] = combined[i * 6 + 3];
  rotation[i * 3 + 1] = combined[i * 6 + 4];
  rotation[i * 3 + 2] = combined[i * 6 + 5];
}

}  // namespace

class SharedEngine {
 public:
  SharedEngine(EngineKind kind,
               const std::filesystem::path& onnx_path,
               const RuntimeOptions& options,
               const Config& config,
               int max_batch)
      : kind_(kind),
        config_(config),
        max_batch_(max_batch),
        runtime_(nvinfer1::createInferRuntime(logger())) {
    if (onnx_path.empty()) {
      throw FoundationPoseError(kind == EngineKind::Refine
                                   ? "refine_model_path is required"
                                   : "score_model_path is required");
    }
    if (!runtime_) {
      throw FoundationPoseError("Failed to create TensorRT runtime");
    }
    loadOrBuild(onnx_path, options);
  }

  nvinfer1::ICudaEngine& engine() const { return *engine_; }

  bool hasTensor(const std::string& name) const {
    if (name.empty()) {
      return false;
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      if (name == engine_->getIOTensorName(i)) {
        return true;
      }
    }
    return false;
  }

  std::string firstOutputName() const {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        return name;
      }
    }
    throw FoundationPoseError("TensorRT engine has no output tensors");
  }

 private:
  void loadOrBuild(const std::filesystem::path& onnx_path,
                   const RuntimeOptions& options) {
    const char* kind_str = kind_ == EngineKind::Refine ? "refine" : "score";
    const std::vector<char> onnx = readFile(onnx_path);
    const std::string hash = hex64(fnv1a64(onnx));
    std::filesystem::path cache_dir =
        options.engine_cache_dir.empty() ? std::filesystem::path("engine_cache")
                                         : std::filesystem::path(options.engine_cache_dir);
    std::filesystem::create_directories(cache_dir);
    const char* prefix = kind_ == EngineKind::Refine ? "refine" : "score";
    cache_path_ = cache_dir / (std::string(prefix) + "_" +
                               precisionTag(config_.tensorrt_precision) + "_b" +
                               std::to_string(max_batch_) + "_" +
                               std::to_string(config_.input_height) + "x" +
                               std::to_string(config_.input_width) + "_" + hash +
                               ".plan");

    if (std::filesystem::is_regular_file(cache_path_)) {
      const std::vector<char> plan = readFile(cache_path_);
      engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
      if (engine_) {
        return;
      }
    }

    TrtUnique<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
    if (!builder) {
      throw FoundationPoseError("Failed to create TensorRT builder");
    }
    const auto flags =
        1U << static_cast<unsigned int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUnique<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) {
      throw FoundationPoseError("Failed to create TensorRT network");
    }
    TrtUnique<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger()));
    if (!parser || !parser->parse(onnx.data(), onnx.size())) {
      std::string message = "Failed to parse ONNX model: " + onnx_path.string();
      if (parser) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
          message += "\n  ";
          message += parser->getError(i)->desc();
        }
      }
      throw FoundationPoseError(message);
    }

    TrtUnique<nvinfer1::IBuilderConfig> build_config(builder->createBuilderConfig());
    if (!build_config) {
      throw FoundationPoseError("Failed to create TensorRT builder config");
    }
    build_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                     config_.tensorrt_workspace_bytes);
    applyPrecision(*build_config, config_.tensorrt_precision);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) {
      throw FoundationPoseError("Failed to create TensorRT optimization profile");
    }
    for (int i = 0; i < network->getNbInputs(); ++i) {
      nvinfer1::ITensor* input = network->getInput(i);
      const int channels = input->getDimensions().nbDims >= 2 &&
                                   input->getDimensions().d[1] > 0
                               ? input->getDimensions().d[1]
                               : config_.n_channels;
      setProfileForInput(*profile, input->getName(), channels, config_, max_batch_);
    }
    if (build_config->addOptimizationProfile(profile) < 0) {
      throw FoundationPoseError("Failed to add TensorRT optimization profile");
    }

    TrtUnique<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *build_config));
    if (!plan) {
      throw FoundationPoseError("Failed to build TensorRT engine from " +
                               onnx_path.string());
    }
    {
      std::ofstream out(cache_path_, std::ios::binary);
      out.write(static_cast<const char*>(plan->data()), plan->size());
    }
    engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
      throw FoundationPoseError("Failed to deserialize freshly built TensorRT engine");
    }
  }

  EngineKind kind_;
  Config config_;
  int max_batch_ = 0;
  std::filesystem::path cache_path_;
  TrtUnique<nvinfer1::IRuntime> runtime_;
  TrtUnique<nvinfer1::ICudaEngine> engine_;
};

namespace {

std::string sharedEngineKey(EngineKind kind,
                            const std::filesystem::path& onnx_path,
                            const RuntimeOptions& options,
                            const Config& config,
                            int max_batch) {
  const std::filesystem::path cache_dir =
      options.engine_cache_dir.empty() ? std::filesystem::path("engine_cache")
                                       : std::filesystem::path(options.engine_cache_dir);
  return std::string(kind == EngineKind::Refine ? "refine" : "score") + "|" +
         std::filesystem::absolute(onnx_path).lexically_normal().string() + "|" +
         std::filesystem::absolute(cache_dir).lexically_normal().string() + "|" +
         std::to_string(max_batch) + "|" + std::to_string(config.input_height) + "x" +
         std::to_string(config.input_width) + "|" + precisionTag(config.tensorrt_precision);
}

std::shared_ptr<SharedEngine> sharedEngineFor(
    EngineKind kind,
    const std::filesystem::path& onnx_path,
    const RuntimeOptions& options,
    const Config& config,
    int max_batch) {
  // TensorRT owns CUDA-facing process state, and destroying cached engines from
  // static destructors can run after CUDA teardown in Python embedding scenarios.
  // Keep the process-wide cache alive until process exit instead.
  static auto* mutex = new std::mutex();
  static auto* engines =
      new std::unordered_map<std::string, std::shared_ptr<SharedEngine>>();
  const std::string key = sharedEngineKey(kind, onnx_path, options, config, max_batch);
  std::lock_guard<std::mutex> lock(*mutex);
  auto it = engines->find(key);
  if (it != engines->end()) {
    return it->second;
  }
  auto engine =
      std::make_shared<SharedEngine>(kind, onnx_path, options, config, max_batch);
  engines->emplace(key, engine);
  return engine;
}

}  // namespace

class TensorRtRunner::Engine {
 public:
  Engine(std::shared_ptr<SharedEngine> shared,
         RuntimeOptions options,
         Config config,
         int max_batch)
      : shared_(std::move(shared)),
        options_(std::move(options)),
        config_(config),
        max_batch_(max_batch),
        context_(shared_->engine().createExecutionContext()) {
    if (!context_) {
      throw FoundationPoseError("Failed to create TensorRT execution context");
    }
  }

  void enqueueRefine(const float* rendered,
                     const float* observed,
                     float* delta_translation,
                     float* delta_rotation,
                     float* combined_refine_output,
                     int batch_size,
                     cudaStream_t stream) {
    setInputs(rendered, observed, batch_size);
    const bool has_trans = shared_->hasTensor(options_.refine_translation_output_name);
    const bool has_rot = shared_->hasTensor(options_.refine_rotation_output_name);
    if (has_trans && has_rot) {
      setTensor(options_.refine_translation_output_name, delta_translation);
      setTensor(options_.refine_rotation_output_name, delta_rotation);
    } else {
      setTensor(shared_->firstOutputName(), combined_refine_output);
    }
    if (!context_->enqueueV3(stream)) {
      throw FoundationPoseError("TensorRT RefineNet enqueueV3 failed");
    }
    if (!(has_trans && has_rot)) {
      const int block = 256;
      splitCombinedRefineKernel<<<(batch_size + block - 1) / block, block, 0, stream>>>(
          combined_refine_output, delta_translation, delta_rotation, batch_size);
      checkCuda(cudaGetLastError(), "splitCombinedRefineKernel");
    }
  }

  void enqueueScore(const float* rendered,
                    const float* observed,
                    float* scores,
                    int batch_size,
                    cudaStream_t stream) {
    setInputs(rendered, observed, batch_size);
    const std::string output_name =
        shared_->hasTensor(options_.score_output_name) ? options_.score_output_name
                                                       : shared_->firstOutputName();
    setTensor(output_name, scores);
    if (!context_->enqueueV3(stream)) {
      throw FoundationPoseError("TensorRT ScoreNet enqueueV3 failed");
    }
  }

 private:
  void setTensor(const std::string& name, const void* ptr) {
    if (name.empty()) {
      throw FoundationPoseError("TensorRT tensor name is empty");
    }
    if (!context_->setTensorAddress(name.c_str(), const_cast<void*>(ptr))) {
      throw FoundationPoseError("Failed to bind TensorRT tensor: " + name);
    }
  }

  void setInputShapeIfPresent(const std::string& name, int batch_size, int channels) {
    if (!shared_->hasTensor(name)) {
      return;
    }
    if (!context_->setInputShape(name.c_str(), inputDims(batch_size, channels, config_))) {
      throw FoundationPoseError("Failed to set TensorRT input shape: " + name);
    }
  }

  void setInputs(const float* rendered, const float* observed, int batch_size) {
    if (batch_size <= 0 || batch_size > max_batch_) {
      throw FoundationPoseError("TensorRT batch size exceeds engine profile");
    }
    setInputShapeIfPresent(options_.rendered_input_name, batch_size, config_.n_channels);
    setInputShapeIfPresent(options_.observed_input_name, batch_size, config_.n_channels);
    setTensor(options_.rendered_input_name, rendered);
    setTensor(options_.observed_input_name, observed);
  }

  std::shared_ptr<SharedEngine> shared_;
  RuntimeOptions options_;
  Config config_;
  int max_batch_ = 0;
  TrtUnique<nvinfer1::IExecutionContext> context_;
};

TensorRtRunner::TensorRtRunner(RuntimeOptions options, Config config, int max_batch)
    : options_(std::move(options)),
      config_(config),
      max_batch_(max_batch),
      engine_batch_([&]() {
        const int bs = config.batch_size > 0 ? config.batch_size : max_batch;
        const int effective_bs = std::min(max_batch, bs);
        if (effective_bs > 0 && max_batch % effective_bs != 0) {
          throw FoundationPoseError("max_batch (" + std::to_string(max_batch) +
                                    ") must be divisible by effective_bs (" +
                                    std::to_string(effective_bs) + ")");
        }
        return effective_bs;
      }()) {
  if (max_batch_ <= 0 || engine_batch_ <= 0) {
    throw FoundationPoseError("TensorRT runner requires a positive batch size");
  }
  checkCuda(cudaSetDevice(options_.device_id), "cudaSetDevice");
  combined_refine_output_.allocate(static_cast<std::size_t>(max_batch_) * 6 *
                                   sizeof(float));
}

TensorRtRunner::~TensorRtRunner() = default;

TensorRtRunner::Engine& TensorRtRunner::refineEngine() {
  if (!refine_) {
    auto shared = sharedEngineFor(EngineKind::Refine, options_.refine_model_path,
                                  options_, config_, engine_batch_);
    refine_ = std::make_unique<Engine>(std::move(shared), options_, config_,
                                       engine_batch_);
  }
  return *refine_;
}

TensorRtRunner::Engine& TensorRtRunner::scoreEngine() {
  if (!score_) {
    auto shared = sharedEngineFor(EngineKind::Score, options_.score_model_path,
                                  options_, config_, engine_batch_);
    score_ = std::make_unique<Engine>(std::move(shared), options_, config_,
                                      engine_batch_);
  }
  return *score_;
}

void TensorRtRunner::prepareForBatch(int batch_size) {
  if (batch_size <= 0 || batch_size > max_batch_) {
    throw FoundationPoseError("TensorRT prepare batch size exceeds configured maximum");
  }
  (void)refineEngine();
  (void)scoreEngine();
}

void TensorRtRunner::enqueueRefine(const float* rendered,
                                   const float* observed,
                                   float* delta_translation,
                                   float* delta_rotation,
                                   int batch_size,
                                   cudaStream_t stream) {
  if (batch_size <= 0 || batch_size > max_batch_) {
    throw FoundationPoseError("TensorRT refine batch size exceeds configured maximum");
  }
  const int input_stride = config_.n_channels * config_.input_height * config_.input_width;
  for (int offset = 0; offset < batch_size; offset += engine_batch_) {
    const int chunk = std::min(engine_batch_, batch_size - offset);
    refineEngine().enqueueRefine(
        rendered + static_cast<std::size_t>(offset) * input_stride,
        observed + static_cast<std::size_t>(offset) * input_stride,
        delta_translation + static_cast<std::size_t>(offset) * 3,
        delta_rotation + static_cast<std::size_t>(offset) * 3,
        combined_refine_output_.as<float>() + static_cast<std::size_t>(offset) * 6,
        chunk, stream);
    if (offset + chunk < batch_size) {
      checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize TensorRT refine chunk");
    }
  }
}

void TensorRtRunner::enqueueScore(const float* rendered,
                                  const float* observed,
                                  float* scores,
                                  int batch_size,
                                  cudaStream_t stream) {
  if (batch_size <= 0 || batch_size > max_batch_) {
    throw FoundationPoseError("TensorRT score batch size exceeds configured maximum");
  }
  const int input_stride = config_.n_channels * config_.input_height * config_.input_width;
  for (int offset = 0; offset < batch_size; offset += engine_batch_) {
    const int chunk = std::min(engine_batch_, batch_size - offset);
    scoreEngine().enqueueScore(rendered + static_cast<std::size_t>(offset) * input_stride,
                               observed + static_cast<std::size_t>(offset) * input_stride,
                               scores + offset, chunk, stream);
    if (offset + chunk < batch_size) {
      checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize TensorRT score chunk");
    }
  }
}

}  // namespace foundation_pose_nvidia
