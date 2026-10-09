#!/usr/bin/env python3
"""Regression checks for metric semantics, occlusion and candidate attribution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from analyze_candidates import analyze
from mesh_metrics import mesh_error
from prepare_bop import render_visible
from replay_capture import run_sequence
from test_numeric_tools import example_frame


class DatasetTools(unittest.TestCase):
    def test_metric_definitions_and_declared_symmetry(self):
        vertices=np.array([[-.01,-.01,0],[.01,-.01,0],[.01,.01,0],[-.01,.01,0]])
        rotation=np.eye(4);rotation[:2,:2]=[[0,-1],[1,0]]
        result=mesh_error(rotation,np.eye(4),vertices)
        self.assertAlmostEqual(result['vertex_hausdorff_mm'],0)
        self.assertAlmostEqual(result['mssd_discrete_mm'],20)
        result=mesh_error(rotation,np.eye(4),vertices,np.array([np.eye(4),rotation]))
        self.assertAlmostEqual(result['mssd_discrete_mm'],0)
        shifted=np.eye(4);shifted[2,3]=.005
        self.assertAlmostEqual(mesh_error(shifted,np.eye(4),vertices)['vertex_hausdorff_mm'],5)

    def test_shared_depth_buffer_removes_hidden_target(self):
        mesh=(np.array([[-10,-10,0],[10,-10,0],[0,10,0]]),np.array([[0,1,2]]))
        entries=[dict(obj_id=1,cam_R_m2c=np.eye(3).ravel().tolist(),cam_t_m2c=[0,0,z]) for z in [200,100]]
        inst,depth=render_visible(entries,{1:mesh},np.array([[100,0,16],[0,100,16],[0,0,1]]),(32,32))
        self.assertFalse(np.any(inst==1))
        self.assertTrue(np.any(inst==2))
        np.testing.assert_allclose(depth[inst==2],.1)

    def test_ranking_and_center_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);(path/'complete').write_text('ok')
            (path/'metadata.json').write_text(json.dumps(dict(phase='register',iterations=0)))
            c=np.eye(4);c[0,3]=-.01
            poses=np.repeat(np.linalg.inv(c)[None],2,axis=0);poses[1,2,3]=.02
            for name,array in dict(centered_from_original=c,initial_poses_centered=poses,
                                   score_poses_centered=poses,scores=np.array([0,1]),
                                   selected_candidate_id=np.array([1]),selected_pose_original=poses[1]@c).items():
                np.save(path/f'{name}.npy',array)
            _,result=analyze(path,np.eye(4),np.array([[0,0,0],[.01,0,0]]),None)
            self.assertEqual(result['diagnosis'],'ranking_missed_passing_candidate')
            self.assertAlmostEqual(result['stages'][-1]['selected_trajectory_mm'],20)
            (path/'complete').unlink()
            with self.assertRaises(ValueError):analyze(path,np.eye(4),np.zeros((1,3)),None)

    def test_call_count_can_differ_from_capacity(self):
        counts=[]
        class Estimator:
            def register(self,frame,*,n_hypotheses):
                counts.append(n_hypotheses)
                return SimpleNamespace(pose=np.eye(4),score=1,elapsed_s=.1)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);np.savez(path/'frame.npz',**example_frame())
            run_sequence(Estimator(),lambda *args:args,[dict(id='x',path=path/'frame.npz',action='register')],path,64)
        self.assertEqual(counts,[64])


if __name__=='__main__':unittest.main(verbosity=2)
