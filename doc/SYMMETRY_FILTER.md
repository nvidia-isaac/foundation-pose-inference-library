# Symmetry filter

The optional temporal filter prevents switching between configured equivalent
orientations during registration and tracking. It ports the selection and
center correction from `nvlabs_symmetry_filter.py` into the native estimator,
so C++, C, Python, and group calls share the same behavior. It is disabled by
default and does not change registration hypotheses or confidence scores.

## Axis configuration

Use the Isaac ROS `symmetry_axes` string syntax. Axes are in the **object's
local mesh frame**, with rotation about the scaled mesh's bounding-box center.

| Configuration | Equivalent rotations considered |
| --- | --- |
| `[]` | Disabled |
| `["z_180"]` | Identity and 180 degrees around z (C2 plate) |
| `["x_180"]` or `["y_180"]` | Identity and 180 degrees around that axis |
| `["z_90", "z_180", "z_270"]` | All four quarter-turn orientations (C4) |
| `["z_full"]` | z rotations from 0 through 330 degrees, every 30 degrees |
| `["x_180", "y_180"]` | Cartesian combinations, composed as `Rz * Ry * Rx` |

Numeric suffixes are **explicit angles, not step sizes**: `z_90` alone adds
only 90 degrees alongside identity. `x_full` and `y_full` also use 30-degree
sampling; `full` is not an exact continuous-symmetry solution. Negative angles
and angles above 360 degrees are normalized. Invalid entries raise an error
without replacing the previous configuration. Limits are 64 input strings and
4096 combinations.

## Python

```python
from foundation_pose_nvidia import Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig

options = EstimatorOptions.from_env(
    "plate.obj", mesh_unit_scale=0.01, symmetry_axes=["z_180"]
)
config = RuntimeConfig(max_image_width=1280, max_image_height=720)
with Estimator(options, config) as est:
    first = est.register(RgbdFrame(rgb, depth_m, K, mask))
    following = est.track(RgbdFrame(next_rgb, next_depth_m, K))
    # Registration again keeps continuity with the preceding accepted output.
    est.reset_tracking(first.pose)  # Restore a known accepted pose/track seed.
    est.reset_tracking()            # New episode: next call must register.
    est.set_symmetry_axes([])       # Disable; retain the current pose if any.
```

Inputs above are application-provided RGB, depth in metres, intrinsics, and
registration mask. In `MultiObjectEstimator`, set
`ObjectSpec("plate", cad_path, mesh_unit_scale=0.01, symmetry_axes=["z_180"])`.
Each group object has its own filter and pose history.

## C++, C, and CLI

```cpp
// Existing foundation_pose_nvidia::FoundationPose instance:
estimator.setSymmetryAxes({"z_180"});
// registerFrame / trackFrame now return and store the selected equivalent.
estimator.resetTracking(accepted_pose);  // Optional original-frame Mat4f.
estimator.resetTracking();              // Clear history for a new episode.
```

```c
const char* axes[] = {"z_180"};
char error[4096];
int status = fp_set_symmetry_axes(handle, axes, 1, error, sizeof(error));
/* Check status before registration/tracking. */
/* Restore a row-major 4x4 host float pose, or pass NULL to clear history: */
status = fp_reset_tracking(handle, accepted_pose, error, sizeof(error));
```

For C groups, configure a borrowed `fp_group_handle(group, object_index)` before
registering. Do not change configuration or reset an object during concurrent
group calls. Reset with a pose requires the estimator's CUDA device to be current,
as do the existing frame calls. These are additive C functions: `fp_config_t`
and `fp_create_options_t` layouts are unchanged. Rebuild C++ clients against the
new headers and library; Python reports an explicit error if the loaded library
does not support the requested feature.

The smoke-test CLI accepts `--symmetry-axis z_180`. Repeat the flag to provide
multiple entries, e.g. `--symmetry-axis z_90 --symmetry-axis z_180
--symmetry-axis z_270` alongside the existing smoke-test arguments.

## Selection and tracking feedback

For original-frame pose `T` and scaled mesh center `c`, each configured rotation
`R` produces a symmetry transform `S = Trans(c) * R * Trans(-c)`. The filter
selects the candidate `T * S` closest in rotation to the previous filtered pose.
Identity wins ties. The first pose is retained when there is no previous pose.

This preserves the physical center while adjusting both orientation and the
reported mesh-origin translation. Rotating about a corner origin without the
center correction would move the object. After selection, the native estimator
stores `T_selected * Trans(c)` in its centered CUDA tracking buffer. This is the
equivalent of updating NVlabs `est.pose_last` in the original Python wrapper.

Re-registration retains the anchor. Clear history for a new object/episode, or
restore the last accepted pose if an external quality guard rejects an output.
Configuration changes retain history; use a new estimator when changing meshes.

The configured rotations must be acceptable symmetries for the application.
The filter cannot identify a plate's A1 corner, recover a wrong initial
orientation, correct depth-driven drift, or reliably distinguish real large
rotations between frames from symmetry flips. Approximate CAD symmetries can
change small visual details; scores are those of the pre-filter estimate and
are not recomputed.

## Provenance and verification

The continuity algorithm follows `nvlabs_symmetry_filter.py`, whose four geometry
helpers came from [Zheng Wang's symmetry runner](https://github.com/zwdoescode/FoundationPose/blob/da261a154f229a4f7a809ed1b9c97a514a838811/run_for_symmetry_objs.py).
Axis syntax, explicit-angle semantics, and `full` sampling follow
[Isaac ROS's symmetry construction](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/main/isaac_ros_foundationpose/src/foundationpose_impl/pose_sampler.cpp).
Isaac ROS uses these transforms when clustering registration hypotheses; this
implementation uses them for temporal selection and tracking feedback.

See [tests](../tests/README.md#symmetry-filter-tests) for geometry checks, CUDA
feedback checks, and C/Python/group API coverage.
