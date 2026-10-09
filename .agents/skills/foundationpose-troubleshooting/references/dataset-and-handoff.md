# Dataset diagnosis and engineer handoff

## Establish the evaluation contract first

Recover the evaluator, selected prediction-to-GT matches, visibility rule, model
vertices, pose direction, units, and effective per-call options. Recompute saved
metrics before attributing errors. A label such as “max-vertex” is insufficient:
maximum corresponding-vertex displacement, bidirectional nearest-vertex maximum
(vertex Hausdorff), and symmetry-minimized maximum displacement are different.
`mesh_metrics.mesh_error` reports all three plus ADD-S. Discrete symmetries and
their translations use meters. Continuous symmetries require explicit sampling.
Meshes sampled differently can change nearest-vertex metrics.

Use a fixed matched cohort for paired input ablations. Report unmatched GT
instances separately; better masks must not silently change the denominator.
State whether P90 is pooled across instances or aggregated across scenes, and
the percentile convention. Report failures, sample counts, visibility strata,
and repeats. Resample scenes rather than individual correlated instances for
bootstrap uncertainty. A small threshold crossing in one run is weak evidence.

Do not infer acquisition time from image IDs: depending on the dataset schema,
IDs may identify camera views rather than consecutive frames. Verify which
camera each RGB image, depth map, and annotation uses. Inspect reference
generation code and metadata to distinguish measured, estimated, and rendered
depth. Agreement between two renderers using the same GT is not independent
validation of calibration or annotations.

## Prepare visible GT masks

`prepare_bop.py` accepts a BOP scene, an explicit camera/image key, mesh directory,
RGB, and registered depth. It renders **all annotated objects into a shared depth
buffer** and selects one target's visible mask. Independent silhouettes are
amodal and include occluded pixels. Unannotated occluders are not modeled.

The paths and IDs below are illustrative placeholders. Select a scene, image,
and target instance from the dataset being investigated.

```bash
SCENE_DIR=/path/to/dataset/test/000001
MODEL_DIR=/path/to/dataset/models
python3 "$SKILL/scripts/prepare_bop.py" \
  --scene "$SCENE_DIR" --image-id 0 --gt-index 0 \
  --models "$MODEL_DIR" \
  --rgb "$SCENE_DIR/rgb/000000.png" \
  --depth /path/to/registered-depth.npy \
  --depth-scale-to-m 1 --depth-provenance stereo-estimated \
  --output /tmp/pose-case
```

Requires NumPy and OpenCV. The CPU adapter supports binary little-endian PLY,
millimeter meshes and BOP translations, pinhole intrinsics, and camera-Z depth.
Explicitly convert other layouts/formats. Pixel centers are at x+0.5/y+0.5;
near-plane-crossing triangles are omitted. Inspect mask overlays before use.
Output includes a lossless frame, replay manifest, mask PNG, reference pose,
mesh vertices and discrete symmetries in meters, rendered depth/instance IDs,
and hashes. These geometry-bearing artifacts stay local by default.

For depth intervention, copy the original full depth and replace only valid
reference pixels in the target's **visible GT region**. Keep other objects and
background unchanged; replacing the whole image with an object-only render
confounds depth quality, missing values, and occlusion. Run original mask/depth,
GT mask/original depth, original mask/reference target depth, and both. The
registration mask controls initialization; it does **not** remove other depth
from observed network crops. “GT mask + target reference” is not perfect full
scene input. Synthetic depth sharing the evaluation GT is an oracle control,
not a deployable correction or proof of physical depth accuracy.

## Analyze candidates, then package evidence

```bash
python3 "$SKILL/scripts/analyze_candidates.py" \
  --trace /path/to/register_pid_counter \
  --reference /tmp/pose-case/reference_pose.npy \
  --vertices-m /tmp/pose-case/vertices_m.npy \
  --symmetries-m /tmp/pose-case/symmetries_m.npy \
  --threshold-mm 5 --output /tmp/candidate-analysis --plot
```

This helper requires a completed registration trace. It converts centered poses
back to original-mesh coordinates, validates the selected pose, exports every
candidate/stage to CSV, and summarizes the selected trajectory and GT oracle.
The threshold classification uses vertex Hausdorff; discrete MSSD and ordinary
corresponding-vertex maximum remain in the CSV. Optional plots use Matplotlib.

A passing candidate with a failing winner identifies a ranking opportunity.
No passing final candidate implicates candidate generation/refinement/inputs
together, not the scorer alone. A large initial translation error is not proof
of failure: surface-depth initialization naturally differs from the mesh center.
Test additional iterations and hypothesis counts independently; either can worsen
selection. Do not recommend parameter increases based only on intuition.

An engineer handoff should contain:

- A local HTML or Markdown report stating the reproduced symptom, exact metric,
  cohorts, baseline parity, interventions, supported causes, counterexamples,
  limitations, and a short next-experiment list with acceptance criteria.
- Per-instance CSV, aggregate JSON, poses, input/reference provenance and hashes,
  source revision/diff, loaded library/engine/weight hashes, environment versions,
  command lines, and the temporary instrumentation patch.
- A failure and successful control with lossless RGB/depth/mask/K captures;
  RGB/GT/prediction overlays, depth residual maps, calibrated camera-frame CAD
  versus depth projections, error distributions, and candidate convergence/ranking.
- Completed traces for representative cases, exact network tensors when needed,
  and commands verified in a fresh process. `replay_network.py` can verify raw
  refiner/scorer outputs without loading CAD; it uses the bundled C++ runner
  compiled against CUDA and TensorRT 10, compatible trusted engine plans, and
  FP32 IO named inputA/inputB with trans/rot outputs and score (or one uniquely
  named scorer output). It does not test rendering or preprocessing.

Keep case-specific evidence out of the public skill, including private names,
paths, results, hashes, logs, images, CAD, and tensors. Store the investigation
bundle separately and link it only from its private handoff. For a public
example, use synthetic or explicitly public inputs and review generated
metadata and visualizations for identifying information. Identify which helpers
were actually executed, which analysis needed custom glue, and which hypotheses
remain unresolved; a well-organized report alone does not establish a root cause.
