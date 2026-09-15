<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Third-Party Notices

This repository does not redistribute third-party source code, but it
**downloads, installs, and links against** third-party software at build and
run time, and its benchmark/data scripts fetch third-party datasets and models.
Each component below remains governed by its own upstream license — review
those terms before use. Version pins live in `docker/docker_build.sh`,
`docker/benchmark.Dockerfile`, and `scripts/`.

## Fetched and linked at build time

### nvdiffrast

- **Upstream project:** https://github.com/NVlabs/nvdiffrast
- **Copyright:** NVIDIA Corporation & affiliates
- **License:** [NVIDIA Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt)
- **Usage:** optional renderer backend, disabled by default. When explicitly
  enabled (`WITH_NVDIFFRAST=1` / `-DFOUNDATION_POSE_WITH_NVDIFFRAST=ON`), it is
  fetched at a pinned commit by `docker/docker_build.sh` into
  `external/nvdiffrast/` (not committed), patched for standalone (non-PyTorch)
  compilation, and compiled into a separate shared library,
  `libfoundation_pose_nvdiffrast.so`, which the Apache-2.0 main library loads
  dynamically (dlopen) only when `FP_RENDERER=nvdiffrast` is requested. The
  default build contains no nvdiffrast code; rendering uses the built-in CUDA
  rasterizer.

### libpng

- **Upstream project:** http://www.libpng.org/pub/png/libpng.html
- **License:** [PNG Reference Library License version 2](http://www.libpng.org/pub/png/src/libpng-LICENSE.txt)
- **Usage:** linked by the core library (mesh texture decoding) and the C++
  example (frame loading); provided by the build container.

## NVIDIA software (container, SDKs, model weights)

### NGC PyTorch container (`nvcr.io/nvidia/pytorch`)

- **Upstream:** https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- **License:** [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf)
- **Usage:** base image for build, benchmark, test, and example services.

### CUDA Toolkit and TensorRT

- **Upstream:** https://developer.nvidia.com/cuda-toolkit , https://developer.nvidia.com/tensorrt
- **License:** NVIDIA Software License Agreement / [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html), [TensorRT SLA](https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html)
- **Usage:** GPU kernels, runtime, and inference engines; bundled in the container.

### FoundationPose ONNX weights

- **Upstream:** https://huggingface.co/nvidia/foundationpose
- **License:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (see the model card)
- **Usage:** RefineNet/ScoreNet weights downloaded by `scripts/download_weights.sh`.

## Downloaded by the benchmark / data scripts

### BOP toolkit

- **Upstream project:** https://github.com/thodan/bop_toolkit
- **License:** [MIT](https://github.com/thodan/bop_toolkit/blob/master/LICENSE)
- **Usage:** cloned by `scripts/download_bop_ycbv.sh`; used only for official
  BOP19 benchmark evaluation.

### YCB-V dataset (BOP edition)

- **Upstream:** https://bop.felk.cvut.cz/datasets/ (mirror: https://huggingface.co/datasets/bop-benchmark/ycbv)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); YCB objects courtesy of the YCB Object and Model Set
- **Usage:** downloaded by `scripts/download_bop_ycbv.sh`; used by the
  benchmark, tests, and BOP-mode examples.

## Python packages (benchmark/test image)

Installed by `docker/benchmark.Dockerfile` into an isolated virtualenv; each is
governed by its own license as published on PyPI:

| Package | License |
| --- | --- |
| numpy | BSD-3-Clause |
| opencv-python-headless | Apache-2.0 (OpenCV) / MIT (wrapper) |
| scipy | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| imageio | BSD-2-Clause |
| Pillow | MIT-CMU |
| pypng | MIT |
| vispy | BSD-3-Clause |
| PyOpenGL | BSD-style (PyOpenGL license) |
| pytz | MIT |
