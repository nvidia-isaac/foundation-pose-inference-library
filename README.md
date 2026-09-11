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

This project is currently not accepting contributions. The product roadmap is
managed internally by NVIDIA.

[Apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0
