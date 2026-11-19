# Tests

## Symmetry filter tests

Build with `BUILD_TESTING=ON` (default), then run:

```bash
ctest --test-dir build --output-on-failure
PYTHONPATH=python/src python3 tests/test_symmetry_api.py
FP_TEST_GPU=1 PYTHONPATH=python/src python3 tests/test_symmetry_api.py
```

`symmetry_filter` checks the CPU geometry, corner-origin correction, axis
configuration, invalid inputs, and randomized poses. `symmetry_tracking` uses
real CUDA buffers and pose-update kernels with a deterministic mock refiner and
renderer to force a flip, verify register/track continuity, and verify feedback
into the next tracking step. It skips if CUDA is unavailable.

The Python suite checks bindings, error handling, and cleanup without a GPU.
`FP_TEST_GPU=1` also exercises real C handles, per-object group configuration,
and seeded zero-iteration tracking. These tests do not need weights or datasets
and do not measure neural-network pose accuracy. Existing
`tests/test_python_wrapper.py` covers the default Python path.

For a CPU-only geometry check without configuring CUDA/CMake:

```bash
c++ -std=c++20 -Iinclude tests/test_symmetry_filter.cpp src/symmetry_filter.cpp \
  -o /tmp/test_symmetry_filter
/tmp/test_symmetry_filter
```

## Smoke test

Quick sanity check that the built library and TensorRT engines work end-to-end:

```bash
./run_dev.sh run --rm cli --smoke --mode register \
  --cad /work/data/BOP_datasets/ycbv/models/obj_000001.ply \
  --refine /work/weights/refiner_net.onnx --score /work/weights/score_net.onnx \
  --cache /work/engine_cache --mesh-unit-scale 1.0
```

Expected output: a scalar score and a valid row-major 4×4 pose.
`--mode track` and `--model-free-smoke` are also available; run `--help` for all options.

## End-to-end C API tests

End-to-end test scripts for the C API. Like the benchmark harness, they own
dataset I/O in Python and call `libfoundation_pose_nvidia.so` through ctypes,
reusing the helpers in `benchmarks/run_benchmark.py`.

## Requirements

Same as the benchmark harness: the built library (`build/`), the FoundationPose
ONNX weights, and a BOP YCB-V checkout. Run inside the benchmark container
(`fp-bench`, see the repository `README.md`).

## `test_group_api.py`

Exercises the `fp_group_*` multi-object API on one YCB-V frame that contains
several objects:

- group register of all objects in a single call, verified **exactly** against
  the single-object `fp_register_frame` path on the same estimators;
- group (concurrent) vs serial single-handle tracking over subsequent frames,
  reporting wall time per frame;
- per-object skip semantics (`NULL` mask -> `FP_OBJECT_SKIPPED`);
- borrowed-handle access via `fp_group_handle`.

```bash
./run_dev.sh run --rm test                          # all defaults from .env / compose.yaml
./run_dev.sh run --rm test tests/test_group_api.py --track-frames 10   # custom args
```

Dataset/weights/library/cache paths default to the `FP_*` env vars set in
`compose.yaml` / `.env` (see `.env.example`); explicit `--flag` arguments override them.
Without compose, run `tests/test_group_api.py` inside the `fp-bench` container
with `LD_LIBRARY_PATH` pointing at `build/` and pass the paths as flags.

The default scene (`--scene-id 48 --image-id 1 --obj-ids 1,6,14,19,20`) is the
first YCB-V test image with five annotated objects; any scene/image with
multiple `scene_gt` entries works.

> **GPU memory:** five concurrent estimators can use up to ~50 GB. On smaller GPUs
> (e.g. 40 GB), run fewer objects, or lower the hypothesis count instead
> (`FP_N_HYPOTHESES=84`, may drop accuracy):
>
> ```bash
> ./run_dev.sh run --rm test tests/test_group_api.py --obj-ids 1,6,14   # 3 objects
> ```

## `test_device_input.py`

Verifies CUDA **device-buffer frame input**: `fp_image_view_t` RGB/depth/mask
pointers may reference GPU memory (resolved via unified virtual addressing).
The test stages a YCB-V frame into device memory with raw libcudart (no
torch/cupy dependency) and asserts register and track results are bit-identical
to the host-pointer path.

```bash
./run_dev.sh run --rm test tests/test_device_input.py
```

## `test_torch_input.py`

Verifies GPU tensor input through the **Python wrapper**: `RgbdFrame` accepts
torch CUDA tensors (any `__cuda_array_interface__` array) for rgb/depth/mask,
asserting bit-identical results vs host numpy input, mixed host/GPU buffers,
and dtype/contiguity validation. Needs torch, which lives in the container's
system python (not the /opt/bench venv):

```bash
./run_dev.sh run --rm --entrypoint bash test -c \
  'PYTHONPATH=python/src python3 tests/test_torch_input.py'
```

## `test_python_wrapper.py`

Regression tests for the Python wrapper (`python/src/foundation_pose_nvidia`).
Covers four memory/safety properties using `unittest.mock` — **no GPU or real
model files required** for the main suite:

- **Exception-safety**: `fp_destroy` is called when `_prepare` raises during
  `Estimator.__init__`, so the C handle is never leaked.
- **Buffer lifetime**: `ImageView._keepalive` pins the numpy buffer copies
  (e.g. float64→float32 depth, non-contiguous rgb) to the view so they
  outlive the anonymous `RgbdFrame` temporary until the C call completes.
- **Intrinsics pointer stability**: documents that `to_image_view`'s local `k`
  array must remain a view (not a copy) of `self.intrinsics` to avoid a
  dangling pointer in `ImageView.k_row_major`.
- **Finalizer**: `Estimator.__del__` calls `close()` so GPU handles are
  released even when the caller forgets `close()` or a `with` block.

**Run inside the benchmark container (no GPU required for the mock suite):**

```bash
./run_dev.sh run --rm test tests/test_python_wrapper.py
```

**Run with VRAM leak detection** (requires the real library and model files):

```bash
FOUNDATION_POSE_TEST_MODEL_DIR=/work/weights \
./run_dev.sh run --rm test tests/test_python_wrapper.py
```

`FOUNDATION_POSE_TEST_MODEL_DIR` must point to a directory containing
`refiner.onnx`, `scorer.onnx`, and a `cache/` subdirectory for TensorRT
engines. When set and the library is loadable, `TestSanityVRAMLeak` activates
and asserts VRAM stays within 32 MiB of baseline across two
create/register/track/destroy cycles. Without it the VRAM test is skipped
automatically.
