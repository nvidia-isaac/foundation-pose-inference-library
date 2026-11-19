# foundation-pose-nvidia (Python)

Python bindings for `libfoundation_pose_nvidia.so`, structured like NVIDIA's
[cuda-python](https://github.com/NVIDIA/cuda-python) packages:

| Layer | Module | Role |
|---|---|---|
| Low-level | `foundation_pose_nvidia.bindings` | ctypes structures, library loader, raw C ABI |
| High-level | `foundation_pose_nvidia.core` | Pythonic estimator APIs (`Estimator`, `MultiObjectEstimator`, `RgbdFrame`, …) |

## Install

Inside the project container (or any env with the `.so` built):

```bash
pip install -e python/
```

The loader discovers `build/libfoundation_pose_nvidia.so` relative to the repo,
or honors `FP_LIBRARY`.

## Quick start

Requires the weights (`refiner_net.onnx`, `score_net.onnx`) — set the `FP_*`
env vars (as `compose.yaml` does) or pass the paths explicitly. Replace the
placeholder arrays below with your real RGB-D capture + segmentation.

```python
import numpy as np
from foundation_pose_nvidia import Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig

W, H = 640, 480

# 1. Inputs. 
options = EstimatorOptions.from_env(
    "/work/data/BOP_datasets/ycbv/models/obj_000001.ply",
    mesh_unit_scale=0.001,  # YCB-V PLY meshes are in millimeters
)
config = RuntimeConfig(max_image_width=W, max_image_height=H)

# 2. Buffers (HWC uint8 RGB, float32 depth in METERS, uint8 mask, 3x3 K).
rgb = np.zeros((H, W, 3), np.uint8)
depth_m = np.zeros((H, W), np.float32)
mask = np.zeros((H, W), np.uint8)
K = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], np.float32)

# 3. Register (needs a mask) then track (no mask). 
with Estimator(options, config) as est:
    result = est.register(RgbdFrame(rgb, depth_m, K, mask))
    print(result.pose, result.score, result.elapsed_s)

    result = est.track(RgbdFrame(rgb, depth_m, K))  # no mask on track
    print(result.pose, result.score, result.elapsed_s)
```

`RgbdFrame` buffers may be host numpy arrays **or GPU arrays** exposing
`__cuda_array_interface__` (torch CUDA tensors, CuPy, Numba) — GPU input skips
the host-to-device copy and can be mixed per buffer:

```python
frame = RgbdFrame(rgb=rgb_cuda_u8, depth_m=depth_cuda_f32, intrinsics=K, mask=mask_np)
result = est.track(frame)  # rgb/depth stay on the GPU
```

GPU buffers must be C-contiguous with the exact dtype (uint8 RGB/mask, float32
depth in meters); they are validated, never converted. `intrinsics` stays host.

> Note: this is a Python package, not compiled C++ — "compiling" isn't needed.
> `EstimatorOptions.from_env` raises `KeyError` if the `FP_*` weight env vars are
> unset and you didn't pass `refine_model_path` / `score_model_path`.

See [`example/python/fp_single_object_app/`](../example/python/fp_single_object_app/)
for a full, runnable synthetic-frame demo.

## Symmetry filtering

Enable temporal continuity for an object with a 180-degree symmetry around its
local z axis:

```python
options = EstimatorOptions.from_env(
    "plate.obj", mesh_unit_scale=0.01, symmetry_axes=["z_180"]
)
with Estimator(options, config) as est:
    first = est.register(RgbdFrame(rgb, depth_m, K, mask))
    following = est.track(RgbdFrame(rgb, depth_m, K))
    # Both calls return the selected equivalent and update the tracking seed.
    est.reset_tracking()  # Clear the anchor before a new episode.
```

Filtering defaults to disabled. `est.set_symmetry_axes([])` disables it;
`est.reset_tracking(last_accepted_pose)` restores an externally accepted pose
after a quality guard rejects an output. For multiple objects, configure
`ObjectSpec(..., symmetry_axes=["z_180"])` independently for each mesh.

See [Symmetry filter](../doc/SYMMETRY_FILTER.md) for all axis options, original
mesh origin handling, and limitations. Rebuild the native library to enable this
feature; existing C structure layouts are unchanged.
