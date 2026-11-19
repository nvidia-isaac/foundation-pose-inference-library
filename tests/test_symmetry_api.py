# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python/C symmetry API checks; FP_TEST_GPU=1 adds real CUDA handle tests."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python/src'))
from foundation_pose_nvidia import (Estimator, EstimatorOptions, FPLibrary,
    FoundationPoseError, MultiObjectEstimator, ObjectSpec, RegisterOptions,
    RgbdFrame, RuntimeConfig)
from foundation_pose_nvidia.bindings import Config


class TestBindings(unittest.TestCase):
    def setUp(self):
        self.native = MagicMock()
        self.native.fp_set_symmetry_axes.return_value = 0
        self.native.fp_reset_tracking.return_value = 0
        self.library = FPLibrary.__new__(FPLibrary)
        self.library._lib = self.native

    def test_axis_array_and_disable(self):
        self.library.set_symmetry_axes(123, ['z_180', 'x_full'])
        args = self.native.fp_set_symmetry_axes.call_args.args
        self.assertEqual((args[0], args[2]), (123, 2))
        self.assertEqual(list(args[1]), [b'z_180', b'x_full'])
        self.library.set_symmetry_axes(123, [])
        self.assertEqual(self.native.fp_set_symmetry_axes.call_args.args[2], 0)

    def test_string_is_not_an_axis_list(self):
        for invalid in ('z_180', b'z_180', None):
            with self.assertRaises(TypeError):
                self.library.set_symmetry_axes(123, invalid)
        for invalid in ([1], ['z_180\0garbage']):
            with self.assertRaises(ValueError):
                self.library.set_symmetry_axes(123, invalid)

    def test_reset_host_copy_lifetime(self):
        def reset(handle, pointer, *_):
            np.testing.assert_array_equal(np.ctypeslib.as_array(pointer, (16,)).reshape(4,4), np.eye(4))
            return 0
        self.native.fp_reset_tracking.side_effect = reset
        self.library.reset_tracking(123, np.eye(4, dtype=np.float64).T)
        self.native.fp_reset_tracking.side_effect = None
        self.library.reset_tracking(123)
        self.assertIsNone(self.native.fp_reset_tracking.call_args.args[1])

    def test_invalid_reset_and_closed_handle(self):
        for pose in (np.eye(3), np.full((4,4), np.nan)):
            with self.assertRaises(ValueError): self.library.reset_tracking(123, pose)
        with self.assertRaises(FoundationPoseError): self.library.reset_tracking(None)
        with self.assertRaises(FoundationPoseError): self.library.set_symmetry_axes(None, [])

    def test_old_library_has_explicit_feature_error(self):
        self.library._lib = SimpleNamespace()
        with self.assertRaisesRegex(FoundationPoseError, 'Rebuild'):
            self.library.set_symmetry_axes(123, ['z_180'])

    def test_configuration_failure_closes_new_handle(self):
        native = self.native
        native.fp_create.return_value = 123
        native.fp_set_symmetry_axes.return_value = 1
        options = EstimatorOptions(Path('mesh.obj'), Path('refine'), Path('score'), Path('cache'),
                                   prepare=False, symmetry_axes=['bad'])
        with self.assertRaises(FoundationPoseError):
            Estimator(options, Config(), library=self.library)
        native.fp_destroy.assert_called_once_with(123)

    def test_default_does_not_require_new_native_symbols(self):
        self.native.fp_create.return_value = 123
        options = EstimatorOptions(Path('mesh.obj'), Path('refine'), Path('score'), Path('cache'), prepare=False)
        with Estimator(options, Config(), library=self.library) as estimator:
            self.assertEqual(estimator.symmetry_axes, ())
        self.native.fp_set_symmetry_axes.assert_not_called()


@unittest.skipUnless(os.environ.get('FP_TEST_GPU') == '1', 'set FP_TEST_GPU=1 for native CUDA tests')
class TestNativeCudaBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cad = Path(cls.tmp.name) / 'tetrahedron.obj'
        cls.cad.write_text('v 0 0 0\nv .12 0 0\nv 0 .08 0\nv 0 0 .015\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n')
        cls.library = FPLibrary(os.environ.get('FP_LIBRARY', str(ROOT/'build/libfoundation_pose_nvidia.so')), print_build_info=False)
        cls.K = np.array([[180,0,80],[0,180,80],[0,0,1]],np.float32)
        cls.rgb = np.full((160,160,3),128,np.uint8)
        cls.depth = np.full((160,160),.7,np.float32)
        cls.pose = np.eye(4,dtype=np.float32);cls.pose[:3,3]=[.01,.02,.7]

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def estimator(self):
        # Engines load lazily. Zero-iteration tracking after an explicit seed
        # exercises the real C API/CUDA buffers without a neural model.
        options = EstimatorOptions(self.cad,Path('unused'),Path('unused'),Path(self.tmp.name),
                                   prepare=False,symmetry_axes=['z_180'])
        return Estimator(options,RuntimeConfig(n_hypotheses=1,batch_size=1,max_image_width=160,max_image_height=160),library=self.library)

    def test_native_reset_track_validation_and_disable(self):
        frame=RgbdFrame(self.rgb,self.depth,self.K)
        with self.estimator() as estimator:
            self.assertEqual(estimator.symmetry_axes,('z_180',))
            estimator.reset_tracking(self.pose)
            result=estimator.track(frame,n_refine=0)
            np.testing.assert_allclose(result.pose,self.pose,atol=1e-7)
            self.assertEqual(result.score,1)
            with self.assertRaises(FoundationPoseError): estimator.set_symmetry_axes(['z_180junk'])
            self.assertEqual(estimator.symmetry_axes,('z_180',))
            bad=self.pose.copy();bad[0,0]=-1
            with self.assertRaises(FoundationPoseError): estimator.reset_tracking(bad)
            np.testing.assert_allclose(estimator.track(frame,n_refine=0).pose,self.pose,atol=1e-7)
            estimator.set_symmetry_axes([])
            np.testing.assert_allclose(estimator.track(frame,n_refine=0).pose,self.pose,atol=1e-7)
            estimator.reset_tracking()
            with self.assertRaisesRegex(FoundationPoseError,'No previous pose'): estimator.track(frame,n_refine=0)

    def test_group_per_object_configuration(self):
        options=RegisterOptions(self.library.path,Path('unused'),Path('unused'),Path(self.tmp.name),n_hypotheses=1,prepare=False)
        objects=[ObjectSpec('plate',self.cad,symmetry_axes=['z_180']),ObjectSpec('other',self.cad,symmetry_axes=['x_180'])]
        frame=SimpleNamespace(rgb=self.rgb,k=self.K)
        with MultiObjectEstimator(options) as group:
            group._create_group(frame,objects)
            for i in range(2):
                self.library.reset_tracking(group.lib.fp_group_handle(group.group,i),self.pose)
            results=group.track(frame,self.depth,n_refine=0)
            for result in results:
                self.assertEqual(result.status,0)
                np.testing.assert_allclose(result.pose,self.pose,atol=1e-7)
            objects[0].symmetry_axes=['y_180']
            with self.assertRaisesRegex(ValueError,'symmetry settings'): group._create_group(frame,objects)

    def test_c_api_rejects_bad_pointers_and_counts(self):
        native=self.library.cdll;err=C.create_string_buffer(256)
        self.assertNotEqual(native.fp_set_symmetry_axes(None,None,0,err,len(err)),0)
        with self.estimator() as estimator:
            self.assertNotEqual(native.fp_set_symmetry_axes(estimator.handle,None,1,err,len(err)),0)
            array=(C.c_char_p*1)(None)
            self.assertNotEqual(native.fp_set_symmetry_axes(estimator.handle,array,1,err,len(err)),0)
            self.assertNotEqual(native.fp_set_symmetry_axes(estimator.handle,array,65,err,len(err)),0)


if __name__ == '__main__': unittest.main(verbosity=2)
