# Numeric collection and replay

Commands below run from the library repository root:

```bash
SKILL=.agents/skills/foundationpose-troubleshooting
```

## Save exact API inputs

Store one NumPy NPZ per frame. Required keys are `rgb` (HWC uint8 RGB), `depth_m`
(HW float32 camera Z, meters), and `K` (3×3); registration also needs `mask` (HW
uint8, nonzero target). Optional arrays can remain in the archive. Save sensor
timestamps, calibration version, object ID, reference uncertainty, and the action
in a manifest. Keep source sensor data as well if the API inputs were converted.

Example inside the application, after explicit channel/unit conversion:

```python
import numpy as np
from pathlib import Path

# rgb, depth_m, K, mask are the exact host arrays passed to the API.
# For device input, synchronize the producer and copy to host for this capture.
with Path("frame_000000.npz").open("xb") as output:
    np.savez_compressed(output, rgb=rgb, depth_m=depth_m, K=K, mask=mask)
```

Use explicit manifest order; filenames and lexicographic ordering do not define
tracking history. Paths resolve relative to the manifest. Frame IDs are unique
strings, and the first action must register:

```json
{
  "object_id": "part-1",
  "notes": "Original sensor capture; no synthetic depth substitution",
  "frames": [
    {"id": "000000", "path": "frame_000000.npz", "action": "register",
     "rgb_timestamp_ns": 1000000000, "depth_timestamp_ns": 1000000000},
    {"id": "000001", "path": "frame_000001.npz", "action": "track"},
    {"id": "000002", "path": "frame_000002.npz", "action": "register",
     "reset_reason": "Application re-detection"}
  ]
}
```

For independent one-shot tests, set every action to `register`. Re-registration
replaces the previous pose; subsequent `track` uses it. Repeat the whole sequence
to assess variation, including the initial registration.

## CPU measurements

```bash
python3 "$SKILL/scripts/numeric.py" inspect /path/to/frame_000000.npz \
  --output /tmp/input_metrics.json
python3 "$SKILL/scripts/replay_capture.py" --manifest /path/to/manifest.json \
  --output /tmp/fp-input-check --validate-only
python3 "$SKILL/scripts/numeric.py" pose /path/to/estimate.npy /path/to/reference.npy \
  --grasp-point 0.02 0 0.01 --output /tmp/pose_error.json
python3 "$SKILL/scripts/numeric.py" compare /path/to/a.npy /path/to/b.npy \
  --atol 1e-6 --rtol 1e-5 --output /tmp/tensor_difference.json
```

All output paths must be new. `inspect` reports depth percentiles, nonfinite and
valid counts, mask coverage, bounding box, and a raw-depth center proxy. It does
not classify geometry as correct. Bad shape/dtype/intrinsics fail validation;
depth holes and NaNs remain intact for diagnosis. `pose` accepts NPY or text 4×4
poses in the same original mesh frame and meters. It rejects malformed SE(3) and
does not resolve symmetry. `compare` accepts aligned real-valued NPY or text arrays,
reports max/mean absolute difference and RMSE, and returns 1 for shape mismatch,
nonfinite values, or differences outside tolerance; usage/I/O errors return 2.
It does not infer candidate correspondences. Same-shaped arrays can still refer
to different candidates, frames, units, or preprocessing.

## Run the saved inputs through the library

In the environment where the repo's library and Python package are available:

```bash
python3 "$SKILL/scripts/replay_capture.py" \
  --manifest /path/to/manifest.json --output /tmp/fp-baseline \
  --cad /local/private/part.obj --mesh-unit-scale 1.0 \
  --refine-model /path/to/refiner_net.onnx --score-model /path/to/score_net.onnx \
  --precision tf32 --n-hypotheses 252 --n-refine 5 --n-track 2 --repeat 3
```

`FP_LIBRARY`, `FP_REFINE_MODEL_PATH`, `FP_SCORE_MODEL_PATH`, and
`FP_ENGINE_CACHE_DIR` provide defaults. `--repo` selects the checkout if the skill
was copied elsewhere. The runner imports that checkout's Python wrapper. To match
a nondefault baseline, pass `--config /path/to/runtime_overrides.json`, containing
`RuntimeConfig` fields such as `{"crop_ratio": 1.2, "input_width": 160, "input_height": 160}`.
Explicit CLI flags override this file. Unspecified fields use library defaults;
camera capacity follows the saved inputs and CUDA graph capture defaults to false.
Keep graphs false for stage tracing. The runner prepares
the estimator before frame calls. Use a fresh output for the strict FP32 control
or another variation. `--engine-cache /path/to/new_cache` preserves existing plans.

Configured workspace/engine capacity and actual registration count can differ.
Use `--n-hypotheses "$CAPACITY" --register-hypotheses "$ACTIVE_COUNT"`
with the values recorded by the application, and set `--n-refine` independently.
This preserves the configured engine profile while replaying the actual call count. Do not infer the
effective registration defaults from `RuntimeConfig` alone.

For a built `fp-bench` container with data under the repo's mounted `data/`:

```bash
./run_dev.sh run --rm --entrypoint /opt/bench/bin/python bench \
  .agents/skills/foundationpose-troubleshooting/scripts/replay_capture.py \
  --manifest /work/data/case/manifest.json --output /work/data/case/run_tf32 \
  --cad /work/data/case/part.obj --mesh-unit-scale 1.0 --repeat 3
```

Output includes copied lossless inputs and their source hashes, a replayable
manifest, input metrics, `run.json` provenance, `runtime.json` effective loaded
library/configuration, and per-repeat `records.json`, setup
timings, and exact `NNNNNN_pose.npy` matrices. The runner records model/CAD/library
hashes and engine-cache inventory; it does not copy CAD or weights. Referenced
textures/materials and custom runtime components need their own provenance.
Existing cache inventory can include plans unused by this run. Match against the
engine logs or use a fresh cache to identify the used plans.

A failed call stops the sequence, saves an error, and exits nonzero. An invalid
returned pose is retained before validation. It never continues tracking from a
fabricated identity pose. This helper is a CAD single-object replay, not a live
camera recorder, model-free runner, or multi-object concurrency test.

## Optional initialization/refinement/scoring traces

The stock library has no numeric dump flag. Generate the temporary instrumentation
patch only when final poses and input metrics cannot isolate the problem:

```bash
python3 "$SKILL/scripts/make_trace_patch.py" --output /tmp/fp-numeric.patch
git apply --check /tmp/fp-numeric.patch
git apply /tmp/fp-numeric.patch
./run_dev.sh run --rm build
```

Review the patch before using it. The generator validates exact source anchors;
if a revision differs, adapt the anchors after inspecting its control flow. It
does not guess or partially edit source. The patch adds one header and hooks in
`src/foundation_pose.cpp`; ABI and public APIs remain unchanged. Restore just this
instrumentation after the investigation using `git apply -R --check` then
`git apply -R /tmp/fp-numeric.patch`, and rebuild. If reversal fails because of
later edits, resolve the changes explicitly rather than resetting other work.

After applying and rebuilding, set these environment variables in the process
actually loading the instrumented library:

```bash
FP_NUMERIC_TRACE_DIR=/tmp/fp-stage-trace \
FP_NUMERIC_TRACE_TENSORS=1 \
python3 "$SKILL/scripts/replay_capture.py" \
  --manifest /path/to/one_frame_manifest.json --output /tmp/fp-traced-replay \
  --cad /local/private/part.obj --refine-model /path/to/refiner_net.onnx \
  --score-model /path/to/score_net.onnx
```

For Compose, pass `-e FP_NUMERIC_TRACE_DIR=/work/data/case/trace` and
`-e FP_NUMERIC_TRACE_TENSORS=1` before the service name; host variables are not
automatically forwarded. A folder's absence can mean the wrong `.so` was loaded.
Unset `FP_NUMERIC_TRACE_TENSORS` for compact poses/deltas/depth/crop dumps. Enabled
network input pairs require `2*N*6*Hcrop*Wcrop*4` bytes **per stage**: at 252×160×160,
about 295 MiB per pair and 1.73 GiB for five refinement iterations plus scoring,
before depth/XYZ and other arrays. Start with one frame and available disk space.

Each call creates a fresh `<register|track>_<pid>_<counter>/` directory. Counter is
process-wide across calls and threads; serial replay maps counters to manifest
order across repeats. For concurrent applications, add the app's object/frame IDs
at the call site; numeric order alone cannot associate calls with object handles.
Only directories with a `complete` file represent completed captures.

Use `analyze_candidates.py` to export stage errors, coverage and ranking, as
described in [dataset diagnosis and handoff](dataset-and-handoff.md). To replay
a saved network stage independently of CAD/rendering. Compile the small runner
inside the same CUDA/TensorRT environment (adjust include/library paths for your
installation); no Python TensorRT or PyTorch package is needed:

```bash
c++ -std=c++17 -O2 -I/usr/local/cuda/include \
  "$SKILL/assets/replay_network.cpp" -L/usr/local/cuda/lib64 \
  -lnvinfer -lcudart -o /tmp/fp-network-replay
python3 "$SKILL/scripts/replay_network.py" --trace /path/to/register_pid_counter \
  --runner /tmp/fp-network-replay --stage refine_0 --engine /path/to/matching_refine.plan --output /tmp/refine-replay
python3 "$SKILL/scripts/replay_network.py" --trace /path/to/register_pid_counter \
  --runner /tmp/fp-network-replay --stage score --engine /path/to/matching_score.plan --output /tmp/score-replay
```

The helper saves outputs and numeric comparisons against captured raw outputs.
A passing replay establishes that these network tensors/engine reproduce the
saved network outputs; it does not validate the pose, upstream crop, or GT.

All arrays are lossless little-endian float32 NPY, C-order:

- `metadata.json`: phase, actual candidate count, iteration count, dimensions,
  tensor flag, and C++ configuration. Precision values: 0 FP32, 1 TF32, 2 FP16,
  3 BF16. The caller's replay metadata supplies weights, renderer, and revisions.
- `K.npy`, `mesh_diameter_m.npy`, `centered_from_original.npy`: geometry conventions.
- `depth_raw_m.npy`, `depth_filtered_m.npy`, optional `xyz_camera_m.npy`:
  H×W depth and H×W×3 camera XYZ. Exact RGB and mask remain in the input NPZ.
- Registration: `init_translation_m.npy`, `initial_poses_centered.npy` (N×4×4).
  Tracking: `incoming_pose_centered.npy` (1×4×4), including the preceding state.
- Each `refine_<iteration>_...`: poses before and after the update, crop xyxy in
  source pixels, raw translation/rotation outputs (N×3), and, when requested,
  `inputA.npy`/`inputB.npy` (N×6×Hcrop×Wcrop).
- Registration scoring: `score_poses_centered.npy`, `score_crop_xyxy.npy`, optional
  `score_inputA.npy`/`score_inputB.npy`, and unsorted `scores.npy` (N).
- Selection: `selected_candidate_id.npy` (one integer-valued float32),
  `selected_score.npy`, and `selected_pose_original.npy` (4×4). Candidate ID is
  the original row throughout this call; arrays are never sorted. Tracking's
  sole ID is 0 and its score is a placeholder.

The patch copies from the estimator's CUDA stream after production and before
buffers are reused. It rejects tracing with CUDA graph capture enabled. It records
the ordered buffers passed to the runner; inspect `TensorRtRunner::enqueueRefine`
and `enqueueScore` for engine chunk boundaries if batch sizes exceed its cap.
It does not capture internal TensorRT activations. Compare initial candidates,
crop boxes, inputs, deltas, refined poses, then scores to locate the first material
divergence. Compare full candidate sets before attributing a winner change to
scoring. Do not interpret a tracked score of 1 as evidence that tracking succeeded.

## Checks for the helper implementation

Run the bundled CPU regression checks after modifying these scripts:

```bash
python3 "$SKILL/scripts/test_numeric_tools.py"
```

These tests verify measurements, input rejection, replay sequencing/error behavior,
and patch drift rejection. They do not certify GPU inference parity; validate
instrumented and uninstrumented output on the same saved case in the target runtime.
