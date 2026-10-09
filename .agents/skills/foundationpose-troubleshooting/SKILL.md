---
name: foundationpose-troubleshooting
description: Diagnose registration errors, tracking drift, input/calibration problems, or numerical differences in foundation-pose-inference-library. Collect reproducible RGB-D replays, pose/depth metrics, and optional CUDA stage traces; compare with NVlabs FoundationPose when useful.
---

# FoundationPose troubleshooting

Use this skill in a `foundation-pose-inference-library` checkout. Resolve paths
from that checkout and this skill directory. Check the current source before
relying on implementation details.

Keep this public skill and its examples generic. Store private dataset names,
paths, case identifiers, results, and evidence links only in a separate local
investigation bundle. Use synthetic or explicitly public inputs for published
examples; derived images and tensors can reveal private geometry too.

## Investigation

1. Preserve one failing capture and a successful control. Identify registration
   versus tracking, frame/object IDs, resets, timestamps, implementation revision,
   renderer, weights, precision, and effective configuration. State the application
   tolerance and reference uncertainty; distinguish visual mismatch from measured
   pose error. Repeat the unchanged baseline before interpreting small differences.
   For dataset P90 failures, read [dataset diagnosis and handoff](references/dataset-and-handoff.md)
   to recover metric/symmetry definitions, camera IDs, reference provenance, fixed
   cohorts, visible GT masks, candidate attribution, and engineer evidence.
2. Read [library troubleshooting](references/library-troubleshooting.md) for
   input, calibration, mesh, runtime, and tracking checks. Inspect exact data at
   the library boundary. RGB is HWC uint8 RGB, depth is HW float32 camera Z in
   meters, the registration mask is HW uint8, and intrinsics are host 3×3.
   Public poses map the original mesh to the camera; internal candidates use
   the centered mesh. Tracking's score of 1 is a placeholder, not confidence.
3. Use [numeric collection](references/numeric-collection.md) and bundled
   scripts to capture arrays and measure input quality, pose error, and numerical
   deltas. CPU analysis requires only NumPy; replay additionally needs the built
   library, Python bindings, CUDA, and compatible ONNX models. Preserve originals;
   use new output directories for each experiment.
4. Change one factor at a time: mask, depth, calibration/units, resolution,
   precision, renderer, checkpoint, then algorithm settings as evidence warrants.
   Compare identical ordered tracking sequences, including initialization and
   re-registration. Single-frame registration cannot diagnose tracking history.
5. If needed, read [cross-implementation replay](references/cross-implementation.md)
   to compare saved inputs with NVlabs. Record unmatched defaults and
   model conversions; replay outcomes narrow causes but do not prove ownership.
6. Summarize the reproduced symptom, measurements, tested corrections, remaining
   hypotheses, and missing evidence. Retain inputs, poses, configuration, source
   and model identities, debug images, and available traces. Keep CAD and posed
   mesh exports local. Images, point clouds, and tensors can reveal geometry too;
   assembling local evidence does not imply authorization to publish it.

## Helpers

From the repository root, set `SKILL=.agents/skills/foundationpose-troubleshooting`:

```bash
python3 "$SKILL/scripts/numeric.py" inspect /path/to/frame.npz
python3 "$SKILL/scripts/numeric.py" pose /path/to/pose.npy /path/to/reference.npy
python3 "$SKILL/scripts/numeric.py" compare /path/to/a.npy /path/to/b.npy
python3 "$SKILL/scripts/replay_capture.py" --help
python3 "$SKILL/scripts/make_trace_patch.py" --help
python3 "$SKILL/scripts/prepare_bop.py" --help
python3 "$SKILL/scripts/analyze_candidates.py" --help
python3 "$SKILL/scripts/replay_network.py" --help
```

The public API exposes the final pose and score, not candidate arrays or network
inputs. PNG visualizations do not provide numeric tensors. The optional patch
generator adds temporary instrumentation for initialization, filtered depth,
each refinement update, scoring, and winner selection. It produces a reviewable
patch, refuses unrecognized source anchors, and does not edit the checkout. Apply
and rebuild when intermediate capture is needed. Disable CUDA graph capture for
these synchronous dumps, keep the capture short, and measure performance in a
separate uninstrumented run.
