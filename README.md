# NVIDIA FoundationPose Inference Library

A **low-latency, GPU-resident** runtime for
[FoundationPose](https://arxiv.org/abs/2312.08344) — 6-DoF object pose **estimation**
and **tracking** from RGB-D streams. The entire pipeline runs on the GPU with
**CUDA acceleration and TensorRT-optimized inference**, and is exposed through a
**stable C ABI** designed for integration from any language or framework
(C/C++, Python, ROS, Rust, Go, ...). Python bindings are included.

- **Two modalities:** model-based (CAD mesh, OBJ/PLY) and model-free (RGB-D-mask reference views).
- **Two modes:** `register` (one-shot global pose search) and `track` (frame-to-frame refinement at hundreds of Hz).
- **Multi-object:** estimate poses for several objects on one shared frame concurrently.
- **Zero-copy input:** frames can be passed as CPU memory or CUDA device buffers (e.g. PyTorch CUDA tensors).
- **Selectable inference precision:** TF32 (default), strict FP32, FP16, or BF16 TensorRT engines.
- **Inference micro-batching:** configurable batch chunk size (`batch_size`) to evaluate pose hypotheses in smaller chunks, tailoring VRAM usage to memory-constrained GPUs.

## Architecture

The runtime wraps a small GPU pipeline behind two integration surfaces (C ABI and
Python) and keeps the two heavy stages — rendering and neural-network inference —
behind swappable interfaces (a built-in CUDA rasterizer and TensorRT by default):

```
  application ── C ABI ──┐            ┌── IRenderer ──────── built-in CUDA rasterizer
  application ── Python ─┼─ estimator ┤                      (optional: nvdiffrast plugin)
                         │  (per obj) └── IInferenceRunner ─ TensorRT
                         └─ persistent GPU workspace, mesh, CUDA stream
```

An estimator answers two calls: **register** (one-shot global pose search on
first sight of an object; needs a mask) and **track** (per-frame refinement
seeded by the previous pose; no mask). Inputs are RGB (`HWC uint8`), depth
(`HW float32`, meters), and `3×3` intrinsics; the output is a `4×4`
object-to-camera SE(3) pose plus a confidence score. Frames may live in **CPU
memory or CUDA device memory** (auto-detected; GPU input skips the
host-to-device copy), and multiple objects can be estimated concurrently on one
shared frame.

See [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md) for the detailed **register and
track data-flow diagrams**, the **C ABI / Python integration workflows**
(including multi-object `fp_group_*`), and how to plug in a **custom renderer or
inference backend** through the C++ interfaces.

## Requirements

- **NVIDIA driver ≥ 580** (the minimum driver line for CUDA 13).
- **FoundationPose ONNX weights** — `refiner_net.onnx`, `score_net.onnx` from NGC
  [`nvidia/tao/foundationpose:deployable_v1.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationpose)
  (or `scripts/download_weights.sh`).
- Everything runs **inside a container** (`nvcr.io/nvidia/pytorch:26.05-py3`, which
  bundles CUDA 13.2 and TensorRT 10.16); only Docker with the NVIDIA Container
  Toolkit is required on the host. The default build has no further
  dependencies — rendering uses the built-in CUDA rasterizer (see
  [`src/`](src/) for the optional nvdiffrast backend).

**Supported platforms:**

| Platform | GPU | Software stack | Status |
| --- | --- | --- | --- |
| x86_64 datacenter / workstation | Compute capability ≥ 7.5 (validated on RTX PRO 6000 Blackwell; the default container build targets every arch of the base image — see [`src/`](src/)) | driver 580+, `pytorch:26.05-py3` (CUDA 13.2, TensorRT 10.16) | Supported |
| Jetson AGX Thor | Blackwell iGPU | JetPack 7 / L4T R38, driver 580.00, `pytorch:26.05-py3` (aarch64) | Supported — see [`src/`](src/) |

## Getting Started

All containerized workflows go through [`run_dev.sh`](run_dev.sh), which
auto-detects the host architecture and selects the right Docker Compose overlay.
Docker Compose ≥ 2.30 and the NVIDIA Container Toolkit are required on the host.

### 1. Configure

```bash
cp .env.example .env    # set FP_DATA_DIR, FP_WEIGHTS_DIR, FP_UID for your machine
sed -i "s/^FP_UID.*/FP_UID=$(id -u)/" .env
sed -i "s/^FP_GID.*/FP_GID=$(id -g)/" .env
```

### 2. Pull the container image

```bash
./run_dev.sh build
```

### 3. Build the C++ library

```bash
./run_dev.sh run --rm build
```

Produces `build/libfoundation_pose_nvidia.so` and `build/foundation_pose_nvidia_cli`.
See [`src/`](src/) for GPU architecture targets, renderer backends, and platform-specific notes.

### 4. Download weights and dataset

```bash
scripts/download_weights.sh        # ONNX weights into $FP_WEIGHTS_DIR
scripts/download_bop_ycbv.sh       # BOP YCB-V dataset into $FP_DATA_DIR
```

### 5. Run the sample app

```bash
./run_dev.sh run --rm sample_app   # C++ and Python examples (BOP mode)
./run_dev.sh run --rm sample_app python_multi   # Python multi-object example
```

See [`example/`](example/) for all available examples, including synthetic mode
(no dataset required) and multi-object registration.

## Memory Optimization & Micro-Batching

FoundationPose registration mode evaluates candidate rotation hypotheses (default: `n_hypotheses = 252`) across RefineNet and ScoreNet. On memory-constrained GPUs or in multi-model perception pipelines (e.g. running alongside FoundationStereo and SAM), you can tune memory usage through configuration:

### Inference Micro-Batching (`batch_size`)

By default, all hypotheses are executed in a single TensorRT batch (`batch_size = 252`). You can configure `batch_size` (via C ABI `fp_config_t::batch_size`, C++ `Config::batch_size`, or Python `RuntimeConfig(batch_size=...)`) to chunk TensorRT execution into smaller mini-batches (e.g. 42, 63, or 126):

- **Lower VRAM footprint:** TensorRT engines and intermediate activation buffers are allocated to fit the smaller batch size, significantly reducing peak device memory during registration.
- **Full search coverage preserved:** The total search coverage (`n_hypotheses`) remains unchanged; hypotheses are evaluated sequentially across chunks with zero accuracy loss.
- **Must divide the hypothesis count:** when `batch_size` is smaller than the active hypothesis count, it must divide that count exactly. There is no partial final chunk — a non-divisible pair (e.g. `n_hypotheses = 100` with `batch_size = 42`) is rejected by `fp_register_frame` / `FoundationPose::registerFrame` with a `FoundationPoseError`. The suggested values below (42, 63, 126) all divide the default 252.
- **Not compatible with `capture_cuda_graph`:** chunked refinement must synchronize the CUDA stream between chunks, which is illegal during graph capture. Enabling both is rejected with a `FoundationPoseError` whenever `batch_size` is smaller than the active hypothesis count. This only affects registration; tracking runs at batch size 1, never chunks, and keeps working with graph capture enabled.

```python
from foundation_pose_nvidia import Estimator, EstimatorOptions, RuntimeConfig

config = RuntimeConfig(
    n_hypotheses=252,  # total rotation search grid
    batch_size=42,     # evaluate in micro-batches of 42
)
with Estimator(options, config) as est:
    ...
```

### Engine Build Workspace and Batch Size on Lower-Memory GPUs

When compiling TensorRT engines from ONNX (`refiner_net.onnx` and `score_net.onnx`), the builder allocates temporary workspace memory (default: 8 GB, `Config::tensorrt_workspace_bytes`). 

Lowering the workspace size on lower-memory GPUs (e.g. 8 GB–16 GB cards or embedded platforms) **requires lowering `batch_size` accordingly**. The workspace needed by TensorRT's builder scales directly with the optimization profile's batch dimension:
- In testing, building an engine with the default batch size of **252 requires at least ~6.1 GB** of builder workspace.
- To successfully compile engines under constrained workspace limits (e.g. 2–4 GB), pair the reduced workspace with a smaller micro-batch size (such as 42, 63, or 126). This prevents out-of-memory errors during initial build without affecting downstream accuracy.

## Performance at a Glance

Built-in CUDA rasterizer, FP16 precision, BOP YCB-V (4,123 targets), RTX PRO 6000 Blackwell:

| Mode | Mean latency | BOP19 AR |
| --------- | -------------- | -------- |
| Register  | 148.37 ms / target | 0.9165 |
| Tracking (video) | 2.01 ms / target (~498 Hz) | 0.9054 |

Register is one-shot per keyframe; tracking is frame-to-frame on the full video sequences (`test_all`).

See [`benchmarks/`](benchmarks/) for full latency and accuracy comparisons across TF32/FP16 precisions on both RTX PRO 6000 Blackwell and Jetson AGX Thor.

## License

This project is licensed under **[Apache-2.0][Apache-2.0]**. Every source and
script file carries an `SPDX-License-Identifier: Apache-2.0` header; see
[`LICENSE`](LICENSE) for the full terms.

## 3rd-Party Licenses

This project downloads and installs additional third-party open source software
at build and run time (nvdiffrast, TensorRT, CUDA libraries, BOP toolkit,
Python packages, and datasets/model weights fetched by the download scripts).
Review the license terms of these projects before use — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the component list and
license references. Note in particular that the optional **nvdiffrast**
renderer plugin is governed by the *NVIDIA Source Code License* and is
**disabled by default**: it is fetched and built only when explicitly enabled
with `WITH_NVDIFFRAST=1` / `-DFOUNDATION_POSE_WITH_NVDIFFRAST=ON`, as its own
dynamically loaded shared library, never into the Apache-2.0 main library;
the default build contains no nvdiffrast code.

## Contribution Guidelines

By contributing to this project, you agree that your contributions will be licensed under its [Apache License, Version 2.0](LICENSE). 
The product roadmap is managed by NVIDIA.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed/licensed under those terms without any additional conditions.

[Contribution Rules](CONTRIBUTING.md)

[Apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0
