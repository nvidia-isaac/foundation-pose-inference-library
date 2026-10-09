# Troubleshooting the inference library

## Preserve the baseline

Capture the exact arrays presented to `RgbdFrame` / `fp_image_view_t`, output pose,
object/frame IDs, individual sensor timestamps, registration/tracking action, and
reset history. Include a failing case and a successful control, application error
tolerances, and an independent reference pose with uncertainty if available.
Repeat with fixed inputs before evaluating small numerical changes.

Record the source commit and local changes, loaded `.so` identity/build info,
ONNX hashes, engine hashes, GPU/driver/CUDA/TensorRT versions, renderer, precision,
mesh scale, and effective configuration. `replay_capture.py` captures much of this;
retain container image/digest and CUDA/TensorRT versions separately. Store the
actual local source diff when working from a modified tree: a dirty flag alone
does not identify its behavior. Do not compare historical benchmark numbers as
though they were measured on the current failing capture.

## Inputs, segmentation, and depth

- RGB: H×W×3 uint8 **RGB**, not OpenCV BGR. Depth: H×W float32 camera-frame **Z in
  meters**, registered to the RGB optical frame. Mask: H×W uint8; every nonzero
  pixel is target. Intrinsics: finite host 3×3; the ABI consumes only fx/fy/cx/cy.
- Validate all shapes before passing buffers to the ABI. `RgbdFrame` converts host
  dtypes/contiguity, which can hide an upstream conversion; it is not a complete
  shape or calibration validator. Device arrays must already have exact dtypes,
  C-contiguous storage, and pointers accessible on the estimator's device. Preserve
  their lifetime until inference finishes. Reproduce first with saved host arrays
  to isolate zero-copy/stream/integration differences; this does not reproduce a
  live race automatically.
- BOP conversion is `depth_raw * depth_scale / 1000` meters; the camera file's
  `depth_scale` matters. Do not multiply already-meter-valued depth a second time.
  Preserve original floating-point samples rather than quantizing for convenience.
- Overlay masks on RGB and depth. Inspect holes, background leakage, fixture
  contact, thin edges, reflective areas, and depth/segmentation boundary alignment.
  Measure finite/valid depth under the mask, not only whole-image averages.
- Registration filters depth (erosion then bilateral filtering). In the inspected
  CUDA code, initialization uses the mask bounding-box midpoint and the median
  **histogram bin center** of filtered valid masked depth. With 65,536 bins over
  `[depth_min, depth_max)`, default bin width is about 1.526 mm. The CPU helper's
  raw-depth exact median is a diagnostic proxy, not a reconstruction of this stage.
  A biased seed alone does not establish the root cause; refinement may correct it.
- The registration mask affects initialization. It does not mask the full-frame
  XYZ/RGB data sampled into observed refinement/scoring crops. Background or
  adjacent-object depth can therefore affect the pose even with a perfect mask.

If references exist, substitute segmentation alone, depth alone, then both while
holding everything else fixed. Repeatable improvement implicates input processing;
one unchanged result does not rule it out. For synthetic references, render CAD at
an independently known pose using calibrated intrinsics, camera Z, and the visible
silhouette. Preserve occluders and measured background. Label synthetic depth
clearly: improvement may also reflect closer agreement with the runtime renderer.

## Calibration, mesh, and coordinate frames

Check final rectified/dewarped intrinsics, depth-to-color extrinsics, synchronization,
and any crop/resize. For a pixel-center resize, `fx'=sx*fx`, `fy'=sy*fy`,
`cx'=(cx+0.5)*sx-0.5`, `cy'=(cy+0.5)*sy-0.5`; use the convention of the actual
resampler. Apply crop offsets before resizing. Use nearest-neighbor depth/mask
resampling to avoid blending object and background. Compare native and reduced
resolution from the same acquisition.

Verify CAD dimensions, axes, origin, texture/colors, and object-to-grasp transform.
`mesh_unit_scale=0.001` is appropriate for millimeter mesh coordinates, not a
universal default. The library applies mesh scaling before preprocessing.

Internal poses transform the **centered** mesh to the camera. With mesh center c
in meters, `C = centered_from_original = [I, -c; 0, 1]`, and:

```text
T_camera_from_original = T_camera_from_centered @ C
```

The public pose is already in the original mesh frame. Do not center it twice or
invert it to compare against another object-to-camera pose. Translations are
meters, rotations are row-major matrix entries. Evaluate axis-specific errors in
the requested object/camera/world frame. The bundled pose helper reports a
camera-frame translation delta, geodesic rotation error, and optional grasp-point
error. That helper does not resolve symmetry. `mesh_metrics.py` provides explicit
vertex metrics with discrete symmetries, and `analyze_candidates.py` applies them
to traces; see [dataset diagnosis](dataset-and-handoff.md). Raw Euler differences
are misleading for symmetric objects.

Back-project measured depth with calibrated intrinsics, transform the original
CAD by the public pose, and inspect both in the same 3D camera frame. Use axes,
units, and wireframe/transparency. Retain local views or describe the mismatch;
the library itself has no debug-image exporter. CAD/posed mesh stays local.

## Registration, tracking, and inference

Read `src/foundation_pose.cpp`, `src/cuda_pipeline.cu`, `src/tensorrt_runner.cu`, and
`include/foundation_pose_nvidia/config.hpp` when local behavior differs.

- Defaults inspected here: 252 requested hypotheses, 5 registration refinement
  iterations, 2 tracking iterations, 160×160 network crops, crop ratio 1.2, TF32.
  Active hypotheses are clamped to the generated grid and workspace capacity.
  Match actual candidates/order rather than assuming requested counts match.
- Registration needs a mask, initializes candidates, refines, scores, and keeps
  the best pose. Tracking requires prior registration on the same estimator and
  refines one previous pose. Ordinary tracking does not use a mask or run scoring;
  `score=1` is a placeholder. Re-register explicitly after lost tracking when the
  application calls for it, and record this event.
- For drift, replay every intervening frame, initialization, drops, and resets.
  Sparse BOP keyframes are unsuitable substitutes for dense video. The benchmark's
  `--track-all-frames` requires the dense split; see `benchmarks/README.md`.
  Use one estimator per object; a wrong multi-object mask/handle association can
  appear as a numerical problem.
- Network inputA is rendered RGBXYZ and inputB observed RGBXYZ, float32 NCHW
  `[N,6,Hcrop,Wcrop]`. RGB channels are normalized by 255; XYZ is camera XYZ minus
  candidate translation divided by mesh radius. Out-of-range/invalid XYZ is zeroed.
  Refiner and scorer validity thresholds differ (`depth_min` versus 0.1 m here).
  Capture crop boxes and actual tensors to locate the first divergence.
- Raw translation deltas are scaled by mesh diameter/2 before adding to camera
  translation. Rotation logits go through componentwise tanh, the rotation
  normalizer, and the implementation's transposed exponential update. Do not label
  raw rotation output as degrees or directly applied axis-angle radians.
- Scores are raw network outputs, not calibrated probabilities. Preserve candidate
  order and scoring context. The library does not add NVlabs' wrapper offset of
  100. If a runner chunks inference, preserve each ordered engine batch as well;
  the bundled trace records arrays at the IInferenceRunner boundary, not private
  TensorRT layer activations or pre-split combined outputs.

## Runtime failures and controlled comparisons

- Import/load error: confirm `pip install -e python/` in the active environment,
  `FP_LIBRARY` or auto-discovered `build/libfoundation_pose_nvidia.so`, architecture,
  and shared-library dependencies. `EstimatorOptions.from_env` needs
  `FP_REFINE_MODEL_PATH` and `FP_SCORE_MODEL_PATH` unless supplied explicitly.
- Image capacity error: set `max_image_width`/`max_image_height` to cover the actual
  inputs. These limits do not resize images. Do not change 160×160 model input
  dimensions to match camera dimensions; engine shapes must fit the model.
- Engine/build error: inspect ONNX input/output names and shapes, precision, GPU
  support, memory, and cache permissions. Standard bindings use inputA/inputB,
  trans/rot, and score, with source-defined fallbacks for combined outputs.
- Cache filenames encode ONNX hash, precision, batch size, and crop resolution,
  but not GPU/driver/TensorRT identity. Compare with a **new cache directory** and
  fresh process after GPU/software changes; preserve the old cache as evidence.
  Do not assume a cache copied from another machine is portable.
- Cross-GPU discrepancy: compare strict FP32 (`Precision.FP32`, TF32 disabled)
  against the original precision with identical inputs and fresh runs. This tests
  sensitivity; it does not guarantee bitwise parity or higher task accuracy.
- Renderer discrepancy: built-in CUDA is default. `FP_RENDERER=nvdiffrast` needs
  the optional plugin built using `WITH_NVDIFFRAST=1` / the documented CMake option.
  See `src/README.md`; setting the environment variable alone cannot install it.
- CUDA/device access: use the repo's platform-specific `run_dev.sh`/Compose setup.
  For Thor, read `compose.thor.yaml` and verify actual device nodes on that board;
  do not assume x86 GPU container configuration applies unchanged.
- Performance: record estimator creation/engine preparation separately from frame
  latency, warm up comparable paths, and turn off trace dumps for timing. Tensor
  dumps synchronize the CUDA stream and perform disk I/O.
- Model-free failure: retain each reference RGB-D-mask, intrinsics, and
  camera-to-world transform; check reference alignment and reconstructed geometry.
  The bundled replay script targets CAD-based single-object inference. Adapt the
  existing model-free or multi-object APIs for those cases and state this limit.
