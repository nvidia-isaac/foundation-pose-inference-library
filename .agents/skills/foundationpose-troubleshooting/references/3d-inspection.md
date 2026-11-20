# Interactive point-cloud and CAD inspection

Use 3D inspection to distinguish a misplaced CAD from missing/noisy depth,
occlusion, frame conventions, or a wrong model. Rendering a CAD at an FP output
shows the placement implied by that pose, not the network's internal features.

## Build a portable view

From the repository root, using the capture schema in
[numeric collection](numeric-collection.md):

```bash
python3 -m pip install numpy opencv-python plotly
python3 "$SKILL/scripts/build_3d_viewer.py" \
  --frame /path/to/frame.npz --pose /path/to/estimated_pose.npy \
  --mesh /path/to/part.obj --mesh-unit-scale 1.0 \
  --reference /path/to/rendered-reference/reference.npz \
  --region mask --stride 2 --output /tmp/pose-inspection
```

Omit `--reference` for a measured-depth/estimated-CAD view alone. If GT is absent,
generate the optional reference using the [pseudo-reference workflow](pseudo-references.md).
Generation uses CPU geometry and plotting, with no FP/GPU inference. Opening the
result needs only a modern WebGL browser; the dependency install is for the author.

The helper produces `index.html`, a settings/count summary, and camera-frame PLY
exports. Open the HTML directly. It embeds Plotly, the displayed point clouds, full
mesh geometry, poses as transformed geometry, and an RGB silhouette overlay; no
server, CDN, absolute data path, or external source dataset is needed to view it.
Zip the output directory to include downloadable PLY files as well. Test a freshly
extracted copy with networking disabled before sharing. `127.0.0.1` URLs refer to
each reviewer's own machine and do not share an author's running viewer.

This helper builds **one frame per artifact**; it does not promise a full recording
browser. For sequences, create an index of selected views or extend the viewer
with explicit frame/pose selection and lazy per-frame loading. Package all required
assets and displayed data; an app-only ZIP is not self-contained. Use neutral titles
and review embedded metadata/images/geometry before publishing anything.

## Read the two views

- Left: cloud from reference camera-Z depth plus CAD transformed by its reference
  pose. The title identifies RGB-derived, measured-pose-rendered, or estimate-rendered
  provenance. CAD-only synthetic depth contains modeled hidden surfaces.
- Right: cloud from recorded depth plus CAD at the estimate. To compare a replay
  pose, build another view with that saved pose while keeping recorded depth fixed.
  Changing a pose should not silently change the observed cloud.
- Both: shared metric axis ranges, synchronized camera orbit/pan/zoom, CAD opacity
  and point-size controls, legend toggles, and markers at the original CAD origin.
  RGB overlays use the projected full CAD silhouette; hidden parts can project
  through occluders and must be interpreted accordingly.

Geometry is in OpenCV camera coordinates (x right, y down, z forward), metres:

```text
X = (u + pixel_center_offset - cx) * Z / fx
Y = (v + pixel_center_offset - cy) * Z / fy
P_camera = R_camera_from_object * P_original_mesh + t_camera_from_object
```

Scale CAD exactly once, and apply the public original-mesh pose directly. Do not
recenter the CAD a second time. Keep camera-Z separate from Euclidean ray range.
The default offset is 0 (integer OpenCV pixel centers); use 0.5 only when the
capture/reference convention requires it. Any optional display coordinate rotation
must apply equally to CAD and clouds and be excluded from exported camera geometry.

`--region mask` uses the capture's mask and requires one; `--region scene` includes
the full image. This mask's accuracy is an input assumption, not proof of visibility.
`--stride 2` samples every second pixel in both dimensions. Use `--stride 1` for
full density. The default retains every finite positive depth in the region;
`--depth-range-m MIN MAX` applies an explicit display/export limit. Counts report
excluded pixels. Shared ranges include all displayed cloud/CAD points, so distant
returns can make the CAD small; change the range explicitly rather than deleting
outliers from the source. PLY clouds contain the selected region/sampling/range;
CAD PLYs retain all triangles. Stored source depth is never modified.

## What to inspect and what to verify

Inspect top, oblique, and side views; vary opacity and toggle layers. Compare the
outline, thickness, cavities, support contact, and nearby occluders. A constant
axial separation can indicate depth/calibration/pose bias; local false surfaces,
holes, or repeated structures suggest different hypotheses. The visualization
alone does not identify which component is wrong.

Compare actual RGB silhouettes and internal features to the model. Missing
materials/textures must be reported separately from shape: neutral CAD shading
is not the appearance fed to FP, and RGB-projected cloud color is not a supplied
CAD texture. A pseudo cloud and its generating CAD agree by construction and
cannot independently validate the CAD. Use reliable observed surfaces and RGB.

Before trusting/exporting a new viewer implementation, check back-projection
round trips, exact depth/RGB preservation, pose direction, physical dimensions,
original-origin handling, zero/NaN omission, triangle count, shared scales, and
export coordinates. Test controls and confirm that clipping/sampling are explicit.
For a shareable artifact, open it from a different directory and block HTTP/HTTPS
requests. Optional screenshot previews supplement the interactive artifact;
they do not replace numerical fidelity checks.

The bundled synthetic tests cover projection, rendering, PnP, intervention
invariants, and HTML/PLY generation:

```bash
python3 -m unittest discover -s "$SKILL/scripts" -p 'test_*.py'
```

Keep generated HTML, PLY, CAD, RGB, point clouds, fit points, and result metrics in
the local investigation bundle. The public skill contains generic procedures and
procedurally generated test geometry only.
