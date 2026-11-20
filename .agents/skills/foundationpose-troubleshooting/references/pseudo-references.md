# References when measured GT is missing

Use this workflow to test pose/depth consistency without claiming independently
measured ground truth. It complements, rather than replaces, the input and
calibration checks. Preserve the original capture and every reference's method,
initialization, constraints, correspondence IDs, calibration, and uncertainty.

## Choose the available evidence

- Measured pose, missing depth: render CAD at that pose. Call it **rendered depth
  from measured pose**, not measured depth. CAD/calibration errors still apply.
- Missing pose, useful RGB: fit a **RGB-derived pseudo pose**, then render a
  **pseudo depth** map. Do not use the suspect depth in the fitting residual.
- Existing depth used for ICP or pose fitting: disclose the circular dependence;
  that reference cannot independently test the same depth sensor's accuracy.
- Insufficient visible/asymmetric RGB evidence: report unobservable components
  or multiple solutions. Do not manufacture a unique six-DoF reference.

An FP output can initialize an RGB fit, but the resulting reference is not fully
independent of FP. Vary the initialization, include symmetry alternatives, and
inspect convergence. Rendering the unmodified FP estimate is a useful consistency
visualization (`estimate-render`), not a pseudo-GT accuracy measurement.

## RGB-edge alignment versus feature-pattern alignment

**Edge alignment** fits the projected CAD silhouette or selected structural edges
to actual image edges. Segmentation can select search regions, but a segmentation
boundary is not automatically a true RGB boundary. Exclude occluding tools,
fixtures, shadows, and CAD edges hidden by the object itself. Do not match an
amodal CAD perimeter to the outline of an occluder.

For a local six-DoF edge fit, parameterize a small SO(3) rotation about the physical
mesh center plus translation. Render or sample the current visible CAD contour,
associate it with nearby verified RGB edges, and minimize robust image-space
distances with bounds justified by the capture. Refresh visibility and associations
after pose updates. Use several exposed sides and internal edges when available;
one straight edge cannot constrain all six components. Compare a translation-only
fit with a full pose fit to expose rotation/translation coupling. Save the selected
edge pixels, objective, limits, fit residuals, and overlay before/after the fit.
This object-specific edge/occlusion association is not automated by the helper.

**Feature-pattern alignment** fits known CAD landmarks to detected RGB centers,
corners, or intersections, including repeated holes or grid patterns when present.
Metric CAD spacing constrains scale/distance; landmark layout constrains rotation.
Repeated patterns require fixed row/column identities and one-to-one associations:
a one-period shift can have a low reprojection error and the wrong origin. Use an
asymmetric landmark or exposed boundary to establish identity. Planar PnP can
have two plausible tilt solutions; small pixel errors do not establish metric
depth or angular accuracy.

Prefer agreement between differently constructed references. For example, fit
exposed boundary edges and evaluate unused internal landmarks, then fit those
landmarks and evaluate unused boundaries. These checks still share CAD, camera
calibration, and possibly initialization. If rotation disagrees while center
translation agrees, state that distinction rather than reporting a single
unqualified pose-accuracy number.

## Fit verified RGB correspondences

The bundled `fit` helper implements RANSAC PnP and nonlinear reprojection
refinement. It accepts **already verified correspondences**, not a raw image
detector. For planar points it retains IPPE alternatives. Candidate 0 has the
smallest inlier training residual; inspect all candidates before using one as a
reference. The helper does not estimate covariance or resolve symmetry labels.

Use the lossless capture schema from [numeric collection](numeric-collection.md).
RGB must be rectified with the matching pinhole K; lens distortion is not modeled.
Save a separate NPZ with `object_points_m` (N×3 in the original CAD frame,
metres), `image_points` (N×2, u/v pixels), and optional `fit_mask` (N bool or
0/1). True entries fit the pose; false entries are held out. Use more than the
four-point minimum and spread points across the visible object.

```bash
python3 "$SKILL/scripts/pseudo_reference.py" fit \
  --frame /path/to/frame.npz --points /path/to/rgb_correspondences.npz \
  --output /tmp/rgb-reference
```

Outputs are `pseudo_pose.npy`, `candidate_poses.npy`, `fit.json`, and `rgb_fit.png`
(green observed points, orange projected points). No measured depth is used by
the objective. Preserve the correspondence file with the outputs. Reject or flag
fits with implausible dimensions/distance, cheirality problems, ambiguous landmark
indices, inadequate coverage, poor held-out residuals, or high sensitivity.

Repeat with perturbed starts/associations where applicable, plausible intrinsics,
different visible regions, and nearby image/pose timing offsets. Record median
and tail reprojection residuals, fit coverage, reference-to-reference center and
rotation differences, and temporal consistency. A smooth sequence alone is not
proof of correctness. Do not tune the reference to minimize FP error.

## Render pseudo camera-Z depth

```bash
python3 "$SKILL/scripts/pseudo_reference.py" render \
  --frame /path/to/frame.npz --pose /tmp/rgb-reference/pseudo_pose.npy \
  --mesh /path/to/part.obj --mesh-unit-scale 1.0 \
  --reference-kind rgb-derived --output /tmp/rendered-reference
```

Set the mesh scale from actual units. The helper accepts triangulated OBJ,
binary little-endian PLY, or NPZ arrays `vertices`/`faces`; materials/textures are
not used. It z-buffers the **full CAD triangles**, including cavities and rims,
at the original-frame `cam_T_obj`. A bounding box or a plane is not a substitute
for object depth unless that approximation is itself the experiment.

`reference.npz` contains float32 `depth_m` (camera Z in metres, zero outside CAD),
`pose`, `K`, `reference_kind`, `pixel_center_offset`, and the source capture hash.
`render.json` records source identities, scale, and convention. Do not quantize
these maps to millimetre PNGs for submillimetre comparisons. A measured-pose render
uses `--reference-kind measured-pose-render`; an estimate render uses
`--reference-kind estimate-render`.

The default pixel centers are integer OpenCV u/v. `--pixel-center-offset 0.5`
supports half-offset sampling explicitly; use the same convention when
back-projecting or comparing. The existing BOP helper uses half-offset sampling.
This new adapter compensates for that inside its renderer. The CPU helper rejects
meshes crossing the near plane; use a renderer with clipping if that case matters.

Inspect an RGB silhouette overlay and a [3D cloud/CAD view](3d-inspection.md).
Validate renderer depth against an analytic plane or a second renderer before
trusting small differences. A pseudo cloud's agreement with its source CAD is
expected by construction and does not independently validate shape or pose.

## Prepare and replay a depth intervention

Build a reference for **every evaluated frame**, retaining the original manifest
order, IDs, timestamps, and registration/reset actions. Do not substitute sparse
keyframes for a continuous tracking sequence. A reference index looks like:

```json
{"frames": [
  {"id": "000000", "reference": "ref_000000/reference.npz", "visible_mask": "visible_000000.npy"},
  {"id": "000001", "reference": "ref_000001/reference.npz", "visible_mask": "visible_000001.npy"}
]}
```

Paths are relative to this index. `visible_mask` is a reviewed HW bool/uint8 NPY
of **visible target pixels**, obtained from RGB annotation/segmentation or a
validated scene-visibility model. A full CAD silhouette is amodal. Erode uncertain
boundaries when warranted and report the margin; no universal pixel margin fits
every resolution. Do not classify a large measured/rendered depth disagreement
as occlusion by itself: that would discard the very depth errors being tested.

```bash
python3 "$SKILL/scripts/pseudo_reference.py" intervene \
  --manifest /path/to/captures/manifest.json --references /path/to/references.json \
  --variant visible --erode-px 1 --preserve-registration-frames \
  --output /tmp/depth-visible
python3 "$SKILL/scripts/pseudo_reference.py" intervene \
  --manifest /path/to/captures/manifest.json --references /path/to/references.json \
  --variant valid-only --erode-px 1 --preserve-registration-frames \
  --output /tmp/depth-valid-only
```

The `visible` variant replaces the eroded visible mask intersected with positive
reference depth, including missing depth there. `valid-only` additionally requires
original finite positive depth, preserving the entire original invalid pattern.
RGB, K, registration masks, other capture arrays, and depth outside the replacement
region stay unchanged, including invalid sentinels/NaN bits. Save the replacement
mask and replacement fraction, not just an aggregate improvement. Every reference
must match the capture's shape, K, pixel-center convention and, when present, hash.

`--preserve-registration-frames` keeps registration/reset inputs identical across
runs so the tracking intervention starts from the baseline initialization policy.
Check saved registration poses against baseline run-to-run variability; identical
inputs alone do not guarantee an identical selected pose.
This flag intentionally tests tracking depth, not corrected registration depth.
For a registration-depth test omit it and state that initialization changes too.
The replay runner does not accept an arbitrary pose seed: exact paired seeded
one-step tests need a runtime-specific adapter, e.g. a verified native
`resetTracking`/`fp_reset_tracking` API when available. For NVlabs the seed is
`T_original @ Trans(mesh_center)` in `est.pose_last`. Never write an original-frame
pose directly into a centered-mesh seed. Separate these one-step tests from
continuous replays, where subsequent estimates legitimately diverge.

Run the unchanged baseline and each generated manifest with the same
[`replay_capture.py`](numeric-collection.md#run-the-saved-inputs-through-the-library)
arguments, fresh estimator instances, frames, CAD, weights, precision, iterations,
and reset policy. Use separate outputs and repeats. The intervention helper does
not execute FP or claim that pose accuracy improved; retain actual replay outputs
before making that claim.

## Measure and limit the conclusion

Measure camera-axis center translation, origin translation, SO(3) rotation with
the declared symmetry policy, and application-relevant point/mesh discrepancies.
For a corner-origin mesh, `R*c+t` center displacement separates physical motion
from the translation component's rotation coupling. Report medians, tails, failure
counts, per-frame paired changes, and uncertainty relative to reference sensitivity.

For depth, compare measured minus reference **camera Z** on a fixed reviewed ROI:
signed bias, median absolute residual, tail residual, missing fraction, and
predeclared outlier thresholds. Distinguish exposed surfaces from occluded areas
and cavities. Do not select only depths agreeing with the reference.

Repeatable improvement after changing only target depth supports the original
depth as a **contributor** under the reference/CAD/calibration assumptions. A
valid-only improvement shows hole filling alone is insufficient to explain it.
It does not uniquely identify stereo aliases, bias, alignment, CAD error, or a
renderer/refiner preference for synthetic inputs. Matching the same pseudo pose
used to render the intervention is partly circular; strengthen it with held-out
RGB edges/landmarks, an alternative reference, and a known-good control. Remaining
error can still arise from appearance, occlusion, initialization, or model bias.
Synthetic depth is an offline diagnostic, not a deployable sensor correction.
