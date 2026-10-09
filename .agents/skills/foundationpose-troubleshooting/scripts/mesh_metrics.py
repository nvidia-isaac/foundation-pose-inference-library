#!/usr/bin/env python3
"""Explicit mesh error metrics; vertices, poses, and symmetry translations use meters."""
import numpy as np
from numeric import validate_pose


def nearest_distances(a, b):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(b).query(a)[0]
    except ImportError:
        # Bound temporary pairwise storage for a NumPy-only installation.
        result = np.full(len(a), np.inf)
        for i in range(0, len(a), 256):
            for j in range(0, len(b), 256):
                d = np.linalg.norm(a[i:i+256, None] - b[None, j:j+256], axis=2)
                result[i:i+len(d)] = np.minimum(result[i:i+len(d)], d.min(axis=1))
        return result


def mesh_error(pose, reference, vertices_m, symmetries=None):
    pose, reference = validate_pose(pose), validate_pose(reference)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError('vertices must be a nonempty finite Nx3 array in meters')
    syms = np.eye(4)[None] if symmetries is None else np.asarray(symmetries)
    if syms.ndim != 3 or syms.shape[1:] != (4, 4) or not len(syms):
        raise ValueError('symmetries must be nonempty Sx4x4 rigid transforms in meters, including identity')
    if not any(np.allclose(s, np.eye(4), atol=1e-6) for s in syms):
        raise ValueError('symmetries must include identity')
    p = vertices @ pose[:3, :3].T + pose[:3, 3]
    g = vertices @ reference[:3, :3].T + reference[:3, 3]
    forward, reverse = nearest_distances(p, g), nearest_distances(g, p)
    errors = []
    for sym in syms:
        equivalent = reference @ validate_pose(sym)
        points = vertices @ equivalent[:3, :3].T + equivalent[:3, 3]
        errors.append(np.linalg.norm(p - points, axis=1).max())
    return dict(vertex_hausdorff_mm=float(max(forward.max(), reverse.max())*1000),
                mssd_discrete_mm=float(min(errors)*1000),
                max_corresponding_vertex_mm=float(np.linalg.norm(p-g, axis=1).max()*1000),
                adds_mm=float(forward.mean()*1000),
                translation_mm=float(np.linalg.norm(pose[:3, 3]-reference[:3, 3])*1000))
