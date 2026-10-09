# Replay between the library and NVlabs

Use the exact saved baseline in each implementation before changing it. Match
RGB channel order, original depth scale/precision, mask, calibrated intrinsics,
mesh scale/origin, checkpoints, refinement counts, precision, renderer, and pose
convention where possible. Document differences, including candidate generation
and candidate order. Registration comparisons and tracking comparisons are
separate experiments.

- Library fails and NVlabs succeeds: investigate integration, preprocessing,
  rendering, ONNX conversion, TensorRT precision, candidate construction, and
  frame handling. This does not by itself prove a library bug.
- Both fail: investigate shared inputs, mesh, initialization, refinement/scoring,
  symmetry, and weak observable cues.
- Saved replays pass but the live application fails: check capture fidelity,
  synchronization, drops, reset history, handle ownership, and downstream transforms.

## NVlabs saved-input replay

For an independent registration through `run_demo.py`, use one frame per scene:

```text
replay_scene/
  cam_K.txt
  rgb/000000.png
  depth/000000.png
  masks/000000.png
```

The cited reader expects 16-bit depth PNGs in millimeters and a single-channel
nonzero mask. Apply the original sensor scale before encoding and record integer
millimeter quantization; retain the float original. Prefer NVlabs' Python
`register()` with float depth in meters when measuring submillimeter differences.
The demo registers the first frame and tracks later frames, so a multi-frame
directory is not a sequence of independent registrations. Reproduce resets
explicitly when diagnosing tracking.

Use `--debug 3` with a fresh `--debug_dir`; the demo clears its debug directory.
In headless use, disable interactive display while retaining disk exports. Retain
`vis_refiner.png`, `vis_score.png`, `track_vis/`, and numeric
`ob_in_cam/<frame>.txt`. Refiner PNGs show the full call's before/after candidates,
not every iteration. Detailed tracking refiner visuals need the `extra` outputs
from `track_one()` saved separately.

`model_tf.obj` and `scene_complete.ply` from the first registration are in camera
coordinates. View them together locally using Open3D (mesh wireframe/transparent,
point cloud, and camera axes). Preserve original sensor depth: `scene_raw.ply`
uses filtered depth; the demo's `scene_complete.ply` uses reader depth.

NVlabs debug levels do **not** save a complete numeric trace. At the cited revision:

- Before refinement in `register()`, capture translation and all initial poses.
- Inside `PoseRefinePredictor.predict()`, capture each iteration's poses, crops,
  ordered network inputs, and raw/applied deltas with frame/candidate/batch IDs.
- Capture scorer network inputs and raw outputs in `ScorePredictor.predict()`.
  In `register()`, save the unsorted IDs, scores, and sort permutation.
- Caller-accessible `est.poses` and `est.scores` are sorted together, winner row 0.
  `est.best_id` is the winner's **unsorted** row ID. Do not use it to index the
  already-sorted arrays. Save `C = est.get_tf_to_centered_mesh()` and use
  `T_camera_from_original = T_camera_from_centered @ C`.
- The scorer wrapper adds 100 to raw outputs. Preserve both raw and wrapper values
  if comparing with the library, which returns raw scorer output. Scores depend
  on scoring context and are not confidence probabilities.
- `compute_add_err_to_gt_pose()` is a stub returning -1 at the cited revision.
  Measure reference errors independently; use appropriate symmetry-aware evaluation
  for the task. Tracking ordinarily has no initialization/scoring stage.

The bundled CUDA patch targets this inference library. For NVlabs, use local hooks
at the locations above and NumPy lossless arrays; report any missing stages explicitly.

## Source provenance

These cross-implementation notes refer to the public NVlabs revision
`a1b694b83e633c2cb6115b9063d940a687759392`. Inspect the tested checkout before relying
on these historical details:

- [NVlabs estimator](https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/estimater.py)
- [NVlabs refiner](https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/learning/training/predict_pose_refine.py)
- [NVlabs scorer](https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/learning/training/predict_score.py)
- [NVlabs demo](https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/run_demo.py)
- [NVlabs data reader](https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/datareader.py)
