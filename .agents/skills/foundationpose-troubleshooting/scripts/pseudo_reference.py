#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit RGB correspondences, render camera-Z references, or prepare depth ablations.

Requires NumPy/OpenCV, not FoundationPose or a GPU. RGB fitting uses manually
validated object-point/image-point associations; it is not an automatic detector.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from numeric import file_identity, load_array, load_frame, validate_pose, write_json
from replay_capture import read_manifest
from visual_geometry import load_mesh, load_reference, render_depth, transform


def fit_rgb_points(object_points_m, image_points, k, fit_mask=None, threshold_px=3.0):
    xyz, uv = np.asarray(object_points_m, np.float64), np.asarray(image_points, np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2) or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError('require finite corresponding Nx3 object_points_m and Nx2 image_points')
    keep = np.ones(len(xyz), bool) if fit_mask is None else np.asarray(fit_mask)
    if keep.shape != (len(xyz),) or not np.isin(keep, [0, 1]).all():
        raise ValueError('fit_mask must select fit points; remaining points are held out')
    keep = keep.astype(bool)
    if keep.sum() < 4 or not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError('need at least four fit points and a positive RANSAC threshold')
    x, y = np.ascontiguousarray(xyz[keep]), np.ascontiguousarray(uv[keep])
    singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
    if singular[0] <= 0 or singular[1] < singular[0] * 1e-6:
        raise ValueError('correspondences must not be coincident or collinear')
    planar = bool(singular[2] < singular[0] * 1e-5)
    cv2.setRNGSeed(0)
    ok, r, t, inliers = cv2.solvePnPRansac(x, y, k, None, iterationsCount=200,
        reprojectionError=threshold_px, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 4:
        raise ValueError('RGB PnP did not find four consistent correspondences')
    indices = inliers.ravel()
    if planar:
        solutions = cv2.solvePnPGeneric(x[indices], y[indices], k, None, flags=cv2.SOLVEPNP_IPPE)
        starts = list(zip(solutions[1], solutions[2]))
    else:
        starts = [(r, t)]
    candidates = []
    for r, t in starts:
        r, t = cv2.solvePnPRefineLM(x[indices], y[indices], k, None, r.copy(), t.copy())
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
        if np.any(transform(xyz, pose)[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(xyz, r, t, k, None)[0].reshape(-1, 2)
        error = np.linalg.norm(projected - uv, axis=1)
        fitted = error[keep][indices]
        candidates.append((pose, dict(inlier_rmse_px=float(np.sqrt(np.mean(fitted**2))),
            all_fit_median_px=float(np.median(error[keep])),
            heldout_median_px=float(np.median(error[~keep])) if np.any(~keep) else None,
            reprojection_error_px=error.tolist())))
    if not candidates:
        raise ValueError('no positive-depth RGB pose solution')
    candidates.sort(key=lambda item: item[1]['inlier_rmse_px'])
    return np.stack([item[0] for item in candidates]), dict(
        method='RGB correspondences: RANSAC PnP + LM; planar IPPE alternatives',
        used_measured_depth=False, planar=planar, fit_points=int(keep.sum()),
        heldout_points=int((~keep).sum()), inlier_indices=np.where(keep)[0][indices].tolist(),
        ransac_threshold_px=threshold_px, selected_candidate=0,
        candidates=[item[1] for item in candidates],
        limitation='Pseudo reference, not measured GT. Inspect planar/symmetry ambiguity and held-out cues; covariance is not estimated.')


def replace_depth(frame, ref, visible_mask, variant, erode_px=0):
    visible = np.asarray(visible_mask)
    if visible.shape != frame['depth_m'].shape or visible.dtype.kind not in 'bu' or not np.isin(visible, [0, 1, 255]).all():
        raise ValueError('visible mask must be matching HW bool/uint8 target visibility, not an amodal silhouette')
    if erode_px < 0 or variant not in ('visible', 'valid-only'):
        raise ValueError('invalid erosion or depth variant')
    visible = (visible != 0).astype(np.uint8)
    if erode_px:
        visible = cv2.erode(visible, np.ones((2 * erode_px + 1,) * 2, np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
    selected = (visible != 0) & (ref['depth_m'] > 0)
    original = frame['depth_m']
    valid = np.isfinite(original) & (original > 0)
    if variant == 'valid-only':
        selected &= valid
    depth = original.copy()
    depth[selected] = ref['depth_m'][selected]
    # Bitwise preservation includes NaN payloads and negative/zero sentinels.
    assert np.array_equal(depth.view(np.uint32)[~selected], original.view(np.uint32)[~selected])
    if variant == 'valid-only':
        assert np.array_equal(np.isfinite(depth) & (depth > 0), valid)
    return {**frame, 'depth_m': depth}, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='RGB-only PnP from verified point correspondences')
    fit.add_argument('--frame', type=Path, required=True)
    fit.add_argument('--points', type=Path, required=True, help='NPZ: object_points_m, image_points, optional fit_mask')
    fit.add_argument('--ransac-threshold-px', type=float, default=3)
    render = sub.add_parser('render', help='render full CAD at an explicitly identified reference pose')
    render.add_argument('--frame', type=Path, required=True)
    render.add_argument('--pose', type=Path, required=True)
    render.add_argument('--mesh', type=Path, required=True)
    render.add_argument('--mesh-unit-scale', type=float, required=True)
    render.add_argument('--reference-kind', choices=['rgb-derived', 'measured-pose-render', 'estimate-render'], required=True)
    render.add_argument('--pixel-center-offset', type=float, choices=[0, .5], default=0)
    intervene = sub.add_parser('intervene', help='copy an ordered capture sequence, changing only selected target depth')
    intervene.add_argument('--manifest', type=Path, required=True)
    intervene.add_argument('--references', type=Path, required=True, help='JSON frames: id, reference NPZ, visible_mask NPY')
    intervene.add_argument('--variant', choices=['visible', 'valid-only'], required=True)
    intervene.add_argument('--erode-px', type=int, default=0)
    intervene.add_argument('--pixel-center-offset', type=float, choices=[0, .5], default=0)
    intervene.add_argument('--preserve-registration-frames', action='store_true')
    for p in (fit, render, intervene):
        p.add_argument('--output', type=Path, required=True, help='new directory; never overwrites inputs')
    a = parser.parse_args()
    try:
        if a.output.exists():
            raise ValueError('output already exists')
        if a.command == 'fit':
            frame = load_frame(a.frame)
            with np.load(a.points, allow_pickle=False) as points:
                poses, report = fit_rgb_points(points['object_points_m'], points['image_points'],
                    frame['K'], points.get('fit_mask'), a.ransac_threshold_px)
                image_points = points['image_points'].copy()
                xyz = points['object_points_m'].copy()
            a.output.mkdir(parents=True)
            np.save(a.output / 'pseudo_pose.npy', poses[0])
            np.save(a.output / 'candidate_poses.npy', poses)
            report['inputs'] = [file_identity(a.frame), file_identity(a.points)]
            write_json(a.output / 'fit.json', report)
            overlay = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
            pose = poses[0]
            projected = cv2.projectPoints(xyz, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], frame['K'], None)[0].reshape(-1, 2)
            for observed, predicted in zip(image_points, projected):
                cv2.circle(overlay, tuple(np.rint(observed).astype(int)), 3, (0, 220, 0), 1)
                cv2.drawMarker(overlay, tuple(np.rint(predicted).astype(int)), (0, 80, 255), markerSize=5)
            cv2.imwrite(str(a.output / 'rgb_fit.png'), overlay)
        elif a.command == 'render':
            frame, pose = load_frame(a.frame), validate_pose(load_array(a.pose))
            vertices, faces = load_mesh(a.mesh, a.mesh_unit_scale)
            depth = render_depth(vertices, faces, pose, frame['K'], frame['depth_m'].shape, a.pixel_center_offset)
            if not np.any(depth):
                raise ValueError('reference CAD has no image support')
            a.output.mkdir(parents=True)
            np.savez_compressed(a.output / 'reference.npz', depth_m=depth, pose=pose, K=frame['K'],
                reference_kind=a.reference_kind, pixel_center_offset=a.pixel_center_offset,
                source_frame_sha256=file_identity(a.frame)['sha256'])
            write_json(a.output / 'render.json', dict(inputs=[file_identity(p) for p in (a.frame, a.pose, a.mesh)],
                mesh_unit_scale=a.mesh_unit_scale, reference_kind=a.reference_kind,
                pixel_center_offset=a.pixel_center_offset, depth='camera Z, float32 metres; zero outside full CAD',
                visibility='amodal target only; excludes no unmodeled occluders'))
        else:
            manifest = read_manifest(a.manifest)
            entries = json.loads(a.references.read_text())['frames']
            references = {entry['id']: entry for entry in entries}
            if len(entries) != len(references) or set(references) != {e['id'] for e in manifest['frames']}:
                raise ValueError('reference IDs must match the capture cohort exactly, without duplicates')
            a.output.mkdir(parents=True)
            results, frames = [], []
            for index, entry in enumerate(manifest['frames']):
                original = load_frame(entry['path'])
                item = references[entry['id']]
                refpath = a.references.parent / item['reference']
                maskpath = a.references.parent / item['visible_mask']
                ref = load_reference(refpath, original, a.pixel_center_offset)
                identity = file_identity(entry['path'])
                if 'source_frame_sha256' in ref and str(ref['source_frame_sha256']) != identity['sha256']:
                    raise ValueError(f'reference was rendered for a different capture: {entry["id"]}')
                modified, selected = replace_depth(original, ref, np.load(maskpath, allow_pickle=False), a.variant, a.erode_px)
                preserved = a.preserve_registration_frames and entry['action'] == 'register'
                if preserved:
                    modified, selected = original, np.zeros_like(selected)
                filename = f'{index:06d}.npz'
                np.savez_compressed(a.output / filename, **modified)
                np.save(a.output / f'{index:06d}_replacement_mask.npy', selected)
                frames.append({**entry, 'path': filename})
                valid = selected & np.isfinite(original['depth_m']) & (original['depth_m'] > 0)
                residual = (original['depth_m'][valid] - ref['depth_m'][valid]) * 1000
                results.append(dict(id=entry['id'], input=identity, reference=file_identity(refpath),
                    visible_mask=file_identity(maskpath), reference_kind=str(ref['reference_kind']),
                    replaced_pixels=int(selected.sum()), registration_preserved=preserved,
                    valid_comparison_pixels=int(valid.sum()),
                    signed_depth_median_mm=float(np.median(residual)) if residual.size else None,
                    absolute_depth_p95_mm=float(np.percentile(abs(residual), 95)) if residual.size else None))
            write_json(a.output / 'manifest.json', {**manifest, 'frames': frames,
                'depth_intervention': dict(variant=a.variant, erode_px=a.erode_px,
                    preserve_registration_frames=a.preserve_registration_frames,
                    depth_provenance='rendered; see each reference_kind in intervention.json')})
            write_json(a.output / 'intervention.json', results)
        print(a.output)
    except Exception as e:
        parser.exit(2, f'error: {e}\n')


if __name__ == '__main__':
    main()
