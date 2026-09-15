# Build, Renderers, and Library Usage

## Build

```bash
./run_dev.sh run --rm build
```

Produces `build/libfoundation_pose_nvidia.so` and `build/foundation_pose_nvidia_cli`.

### GPU architecture

`docker/docker_build.sh` resolves the target architectures in this order:

1. An explicit `CUDA_ARCHITECTURES` env var (CMake format — e.g. `120`, `native`, or `75;80;86`)
2. The container-provided `CUDA_ARCH_LIST` (e.g. `7.5 8.0 8.6 9.0 10.0 12.0`), auto-converted
   to CMake format — so the default container build supports every GPU the base image does
3. Fallback `120`

For a faster single-arch build:

```bash
CUDA_ARCHITECTURES=120 ./run_dev.sh run --rm build   # Blackwell
CUDA_ARCHITECTURES=86  ./run_dev.sh run --rm build   # Ampere
```

Compile jobs: `BUILD_JOBS=<n>` (default 4). Native (non-container) builds are
supported — see `CMakeLists.txt`.

## Renderers

The runtime ships two `IRenderer` implementations:

- **Built-in CUDA rasterizer (default).** Self-contained, part of
  `libfoundation_pose_nvidia.so`, no extra dependencies or downloads. Used
  automatically — nothing to configure.
- **nvdiffrast (optional).** Uses the CUDA rasterizer core of
  [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast). nvdiffrast is
  licensed under the *NVIDIA Source Code License* (separate from this
  repository's Apache-2.0 code) and is **disabled by default** — it is not
  fetched or built unless you explicitly enable it with `WITH_NVDIFFRAST=1` /
  `-DFOUNDATION_POSE_WITH_NVDIFFRAST=ON`. It is packaged as its **own
  dynamically loaded shared library**, `libfoundation_pose_nvdiffrast.so` —
  the main library never contains or links nvdiffrast code.

### Enable the nvdiffrast backend

```bash
# 1. Build the plugin (fetches nvdiffrast at a pinned, validated commit,
#    applies the standalone-build patches, and produces
#    build/libfoundation_pose_nvdiffrast.so next to the main library):
WITH_NVDIFFRAST=1 ./run_dev.sh run --rm build

# 2. Select it at runtime (per process; default remains the built-in rasterizer):
./run_dev.sh run --rm -e FP_RENDERER=nvdiffrast cli --smoke --mode register ...
```

Native builds: configure with `-DFOUNDATION_POSE_WITH_NVDIFFRAST=ON
-DNVDIFFRAST_ROOT=/path/to/nvdiffrast`. The plugin is loaded with `dlopen`
from the main library's directory (`$ORIGIN`) or `LD_LIBRARY_PATH`; if
`FP_RENDERER=nvdiffrast` is requested without the plugin present, estimator
creation fails with a clear error.

## Library usage

The build produces the shared library **`libfoundation_pose_nvidia.so`** plus the
public headers under `include/foundation_pose_nvidia/`; link the library (or load it
via FFI) and integrate through the C ABI in
[`c_api.h`](../include/foundation_pose_nvidia/c_api.h):

`fp_default_config` → `fp_create` (or `fp_create_model_free`) → `fp_prepare` →
`fp_register_frame` / `fp_track_frame` → `fp_destroy`; multi-object via
`fp_group_*`. Worked C and Python workflows are in
[`doc/ARCHITECTURE.md`](../doc/ARCHITECTURE.md#integration-workflows).

Buffer conventions: RGB `HWC uint8`, depth `HW float32` in **meters**, mask
`HW uint8` (non-zero = object), intrinsics `3×3` row-major float32 (host). Image
buffers may be **host or CUDA device** pointers — the location is auto-detected,
and device input skips the host-to-device copy. Each estimator handle owns its
own CUDA stream and GPU workspace; use one handle per object. TensorRT engine
precision is selectable per estimator via `fp_config_t.tensorrt_precision`
(TF32 default; strict FP32, FP16, BF16 — see `fp_precision_t` in `c_api.h`).
Engines are cached per precision.

The **Python package** ([`python/`](../python/README.md)) layers a Pythonic
`Estimator` / `RgbdFrame` API (numpy or GPU tensors) over the same ABI
(precision via `RuntimeConfig(tensorrt_precision=Precision.FP32)`), and the
**C++ API** additionally accepts custom `IRenderer` / `IInferenceRunner`
backends (see [`doc/ARCHITECTURE.md`](../doc/ARCHITECTURE.md)).

## Jetson AGX Thor

`run_dev.sh` auto-detects `aarch64` and transparently uses `compose.thor.yaml` — no
flags needed. All commands in the root README work unchanged on Thor.

### Requirements

- **Jetson AGX Thor** flashed with **JetPack 7.0 GA / L4T R38**
  (verify: `head -1 /etc/nv_tegra_release` → `# R38 (release), REVISION: 2.0 ...`).
- **Docker + NVIDIA Container Toolkit** per the official
  [Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html).
- **Internet access** on the board (to pull the base image and fetch nvdiffrast).
- **FoundationPose ONNX weights** — same [Hugging Face source](https://huggingface.co/nvidia/foundationpose) as x86.
- Everything runs inside `nvcr.io/nvidia/pytorch:26.05-py3` (aarch64 variant).

### Build

`compose.thor.yaml` sets `CUDA_ARCHITECTURES=native` automatically so the Thor SM is
auto-detected (no need to override `CUDA_ARCHITECTURES`).

### GPU verification

After building, verify CUDA is accessible inside the container:

```bash
./run_dev.sh run --rm --entrypoint python3 cli \
  -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True Thor
```

### GPU access on Thor

On x86 you'd use `gpus: all`. On Thor that does **not** work, and even `runtime: nvidia`
alone leaves the container with no CUDA device (`CUDA error 100`).

Root cause: on JetPack 7 / L4T R38, `libnvidia-container` in `auto` mode mis-detects
Thor (it has both a Tegra release file *and* a dGPU-style driver), and the shipped
`devices.csv` lists the **old** Tegra node layout (`/dev/nvgpu/igpu0/*`,
`/dev/nvhost-*-gpu`) that does not exist on Thor. Thor's real iGPU nodes are:

- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-modeset`, `/dev/nvmap`
- `/dev/nvgpu/igpu-<pci>/*`
- `/dev/nvgpu-pci/card-<pci>*`

`compose.thor.yaml` therefore keeps `runtime: nvidia` (which still mounts the Tegra
driver libs via `drivers.csv`) and injects the real nodes explicitly via `devices:`,
`device_cgroup_rules:`, and bind-mounts of `/dev/nvgpu` + `/dev/nvgpu-pci`.

> **If you still hit `error 100`:** confirm the node paths on your board
> (`ls /dev/nvidia* ; ls -R /dev/nvgpu /dev/nvgpu-pci`) and adjust the `devices:` /
> `volumes:` lists in `compose.thor.yaml` if your BSP revision names them differently.
