#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic geometry/reference/intervention checks; no captured investigation data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
import cv2
import numpy as np
from pseudo_reference import fit_rgb_points, replace_depth
from visual_geometry import load_mesh, load_reference, point_cloud, render_depth, transform

SCRIPT = Path(__file__).parent


def synthetic_frame():
    k = np.array([[160., 0, 48], [0, 155, 36], [0, 0, 1]])
    rgb = np.zeros((72, 96, 3), np.uint8)
    rgb[:, :, 0] = np.arange(96)
    rgb[:, :, 1] = np.arange(72)[:, None]
    return dict(rgb=rgb, depth_m=np.full((72, 96), .6, np.float32), K=k,
                mask=np.full((72, 96), 255, np.uint8))


def synthetic_mesh():
    # Off-center, sloped quadrilateral; no real object's geometry or dimensions.
    vertices = np.array([[0, 0, 0], [.08, 0, .02], [.08, .06, .02], [0, .06, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    pose = np.eye(4)
    pose[:3, 3] = [-.04, -.03, .5]
    return vertices, faces, pose


class VisualTools(unittest.TestCase):
    def test_tilted_camera_z_and_pixel_conventions(self):
        frame = synthetic_frame()
        v, f, pose = synthetic_mesh()
        for offset in [0., .5]:
            d = render_depth(v, f, pose, frame['K'], frame['depth_m'].shape, offset)
            yy, xx = np.where(d > 0)
            # Camera-space plane: Z = .51 + .25 X; independent ray/plane oracle.
            expected = .51 / (1 - .25 * (xx + offset - 48) / 160)
            np.testing.assert_allclose(d[yy, xx], expected, atol=6e-8, rtol=0)
            self.assertGreater(float(d[d > 0].max() - d[d > 0].min()), .01)
            xyz, rgb, _ = point_cloud(d, frame['rgb'], frame['K'], d > 0, 1, pixel_center_offset=offset)
            projected = np.column_stack((xyz[:, 0] / xyz[:, 2] * 160 + 48,
                                         xyz[:, 1] / xyz[:, 2] * 155 + 36))
            np.testing.assert_allclose(projected, np.column_stack((xx, yy)) + offset, atol=1e-5, rtol=0)
            np.testing.assert_array_equal(rgb, frame['rgb'][yy, xx])
        bad = pose.copy(); bad[2, 3] = 0
        with self.assertRaises(ValueError): render_depth(v, f, bad, frame['K'], (72, 96))

    def test_z_buffer_and_origin(self):
        frame = synthetic_frame(); v, f, pose = synthetic_mesh()
        front = render_depth(v, f, pose, frame['K'], (72, 96))
        # Add a second identical surface behind it; first surface must win.
        joined = np.concatenate([v, v + [0, 0, .05]])
        both = render_depth(joined, np.concatenate([f, f + len(v)]), pose, frame['K'], (72, 96))
        np.testing.assert_array_equal(both[front > 0], front[front > 0])
        np.testing.assert_array_equal(transform(v, pose)[0], pose[:3, 3])

    def test_rgb_pose_outlier_and_heldout(self):
        rng = np.random.default_rng(13)
        xyz = rng.uniform(-.08, .08, (24, 3))
        k = synthetic_frame()['K']
        r, t = np.array([.16, -.12, .08]), np.array([.01, -.02, .6])
        uv = cv2.projectPoints(xyz, r, t, k, None)[0].reshape(-1, 2)
        uv[2] += [15, -15]
        fit = np.arange(len(xyz)) < 18
        candidates, report = fit_rgb_points(xyz, uv, k, fit)
        np.testing.assert_allclose(candidates[0][:3, :3], cv2.Rodrigues(r)[0], atol=1e-6)
        np.testing.assert_allclose(candidates[0][:3, 3], t, atol=1e-6)
        self.assertNotIn(2, report['inlier_indices'])
        self.assertFalse(report['used_measured_depth'])
        changed = uv.copy(); changed[~fit] += 20
        other, changed_report = fit_rgb_points(xyz, changed, k, fit)
        np.testing.assert_array_equal(other, candidates)
        self.assertGreater(changed_report['candidates'][0]['heldout_median_px'], 20)

    def test_planar_candidates_and_degenerate_points(self):
        xyz = np.array([[x, y, 0] for y in [-.04, 0, .04] for x in [-.05, 0, .05]])
        k = synthetic_frame()['K']
        uv = cv2.projectPoints(xyz, np.array([.25, .12, .03]), np.array([0., 0, .5]), k, None)[0].reshape(-1, 2)
        poses, report = fit_rgb_points(xyz, uv, k)
        self.assertTrue(report['planar']); self.assertEqual(len(poses), 2)
        self.assertLess(report['candidates'][0]['inlier_rmse_px'], 1e-4)
        with self.assertRaises(ValueError): fit_rgb_points(np.zeros((5, 3)), np.zeros((5, 2)), k)

    def test_depth_intervention_preserves_background_and_holes(self):
        frame = synthetic_frame()
        frame['depth_m'][20, 20] = 0
        frame['depth_m'][21, 20] = np.nan
        frame['depth_m'][22, 20] = -1
        visible = np.zeros((72, 96), bool); visible[10:30, 10:30] = True
        visible[15:18, 15:18] = False  # A known occluder remains measured.
        ref = {'depth_m': np.full((72, 96), .5, np.float32)}
        for variant in ['visible', 'valid-only']:
            changed, selected = replace_depth(frame, ref, visible, variant)
            np.testing.assert_array_equal(changed['depth_m'].view('u4')[~selected], frame['depth_m'].view('u4')[~selected])
            for key in ['rgb', 'K', 'mask']: np.testing.assert_array_equal(changed[key], frame[key])
            self.assertEqual(bool(selected[20, 20]), variant == 'visible')
            self.assertFalse(selected[16, 16])
        with self.assertRaises(ValueError): replace_depth(frame, ref, visible.astype(float), 'visible')

    def test_mesh_scale_and_reference_validation(self):
        frame = synthetic_frame(); vertices, faces, pose = synthetic_mesh()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.savez(root / 'mesh.npz', vertices=vertices * 100, faces=faces)
            loaded, triangles = load_mesh(root / 'mesh.npz', .01)
            np.testing.assert_allclose(loaded, vertices); np.testing.assert_array_equal(triangles, faces)
            np.savez(root / 'ref.npz', pose=pose, K=frame['K'], depth_m=frame['depth_m'],
                     reference_kind='rgb-derived', pixel_center_offset=.5)
            with self.assertRaises(ValueError): load_reference(root / 'ref.npz', frame)
            self.assertEqual(str(load_reference(root / 'ref.npz', frame, .5)['reference_kind']), 'rgb-derived')

    def test_intervention_cli_preserves_order_and_registration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); frame = synthetic_frame()
            np.save(root / 'visible.npy', frame['mask'])
            np.savez(root / 'reference.npz', pose=np.eye(4), K=frame['K'],
                     depth_m=np.full((72, 96), .4, np.float32), reference_kind='rgb-derived')
            entries, refs = [], []
            for i, (name, action) in enumerate([('later', 'register'), ('earlier', 'track'), ('reset', 'register')]):
                np.savez(root / f'{i}.npz', **frame)
                entries.append(dict(id=name, action=action, path=f'{i}.npz', timestamp=i))
                refs.append(dict(id=name, reference='reference.npz', visible_mask='visible.npy'))
            (root / 'manifest.json').write_text(json.dumps(dict(frames=entries)))
            (root / 'references.json').write_text(json.dumps(dict(frames=refs[::-1])))
            command = [sys.executable, str(SCRIPT / 'pseudo_reference.py'), 'intervene',
                '--manifest', str(root / 'manifest.json'), '--references', str(root / 'references.json'),
                '--variant', 'visible', '--preserve-registration-frames', '--output', str(root / 'result')]
            subprocess.run(command, check=True, capture_output=True)
            saved = json.loads((root / 'result/manifest.json').read_text())['frames']
            self.assertEqual([e['id'] for e in saved], [e['id'] for e in entries])
            for i in range(3):
                with np.load(root / f'result/{i:06d}.npz') as a:
                    np.testing.assert_array_equal(a['depth_m'], np.full((72, 96), .4 if i == 1 else .6, np.float32))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    @unittest.skipUnless(importlib.util.find_spec('plotly'), 'Plotly is optional; install it to test viewer generation')
    def test_viewer_is_offline_and_exports_full_mesh(self):
        from build_3d_viewer import build
        class Scripts(HTMLParser):
            def __init__(self): super().__init__(); self.external = []
            def handle_starttag(self, tag, attrs):
                if tag == 'script' and dict(attrs).get('src'): self.external.append(dict(attrs)['src'])
        frame = synthetic_frame(); v, f, pose = synthetic_mesh()
        ref = dict(pose=pose, depth_m=render_depth(v, f, pose, frame['K'], (72, 96)), reference_kind='rgb-derived')
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'viewer'
            summary = build(frame, pose, v, f, out, ref)
            parser = Scripts(); parser.feed((out / 'index.html').read_text())
            self.assertEqual(parser.external, [])
            self.assertEqual(summary['mesh_triangles'], len(f))
            exported, triangles = load_mesh(out / 'observed_cad.ply', 1)
            np.testing.assert_allclose(exported, transform(v, pose), atol=1e-7)
            np.testing.assert_array_equal(triangles, f)


if __name__ == '__main__': unittest.main(verbosity=2)
