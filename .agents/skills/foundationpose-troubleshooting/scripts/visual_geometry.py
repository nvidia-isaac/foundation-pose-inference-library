# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometry shared by RGB reference generation and offline 3D inspection."""
from pathlib import Path
import numpy as np
from bop_geometry import rasterize_mesh, read_binary_little_endian_ply
from numeric import validate_pose


def load_mesh(path, scale):
    path = Path(path)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('mesh scale must be positive and explicit')
    if path.suffix.lower() == '.ply':
        vertices, faces = read_binary_little_endian_ply(path)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as a:
            vertices, faces = a['vertices'], a['faces']
    elif path.suffix.lower() == '.obj':
        vertices, faces = [], []
        for line in path.read_text().splitlines():
            fields = line.split('#', 1)[0].split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == 'f':
                if len(fields) != 4:
                    raise ValueError('OBJ must be triangulated; do not fan-triangulate concave faces')
                ids = [int(x.split('/')[0]) for x in fields[1:]]
                if 0 in ids:
                    raise ValueError('OBJ vertex indices are one-based')
                faces.append([i - 1 if i > 0 else len(vertices) + i for i in ids])
        vertices, faces = np.asarray(vertices), np.asarray(faces)
    else:
        raise ValueError('use triangulated OBJ, binary little-endian PLY, or NPZ(vertices, faces)')
    vertices = np.asarray(vertices, dtype=np.float64) * scale
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('mesh requires finite Nx3 vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu':
        raise ValueError('mesh requires integer triangular faces')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('mesh face index out of range')
    return vertices, faces.astype(np.int32)


def transform(vertices_m, pose):
    pose = validate_pose(pose)
    return vertices_m @ pose[:3, :3].T + pose[:3, 3]


def render_depth(vertices_m, faces, pose, k, shape, pixel_center_offset=0.0):
    """Camera Z in float32 metres, zero outside CAD, original mesh frame.

    The shared BOP rasterizer samples at u+.5/v+.5. Shift K only inside that
    renderer so the public default corresponds to integer OpenCV pixel centers.
    """
    if pixel_center_offset not in (0.0, 0.5):
        raise ValueError('pixel center offset must be 0 or 0.5')
    camera = transform(vertices_m, pose)
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError('mesh crosses the near plane; this CPU helper does not clip triangles')
    adjusted = np.array(k, dtype=float, copy=True)
    adjusted[:2, 2] += 0.5 - pixel_center_offset
    depth = np.full(shape, np.inf, np.float64)
    ids = np.zeros(shape, np.int32)
    instances = np.zeros(shape, np.int32)
    rasterize_mesh(vertices_m * 1000, faces, pose[:3, :3], pose[:3, 3] * 1000,
                   adjusted, depth, ids, instances, 1, 1, 0.001)
    depth[~np.isfinite(depth)] = 0
    return (depth / 1000).astype(np.float32)


def load_reference(path, frame, pixel_center_offset=0.0):
    with np.load(path, allow_pickle=False) as a:
        ref = {key: a[key] for key in a.files}
    validate_pose(ref['pose'])
    d = ref['depth_m']
    if d.dtype != np.float32 or d.shape != frame['depth_m'].shape or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('reference depth must be matching HW float32 metres, zero outside support')
    if ref['K'].shape != (3, 3) or not np.allclose(ref['K'], frame['K'], atol=1e-6, rtol=0):
        raise ValueError('reference and capture intrinsics differ')
    if float(ref.get('pixel_center_offset', 0.0)) != pixel_center_offset:
        raise ValueError('reference and capture pixel-center conventions differ')
    if str(ref.get('reference_kind', '')) not in ('measured-pose-render', 'rgb-derived', 'estimate-render'):
        raise ValueError('reference_kind must identify measured-pose-render, rgb-derived, or estimate-render')
    return ref


def point_cloud(depth, rgb, k, region, stride=1, limits=(0, np.inf), pixel_center_offset=0.0):
    if stride < 1 or not 0 <= limits[0] < limits[1]:
        raise ValueError('invalid stride or depth range')
    valid = region & np.isfinite(depth) & (depth > 0)
    selected = valid & (depth >= limits[0]) & (depth <= limits[1])
    v, u = np.where(selected[::stride, ::stride])
    v, u = v * stride, u * stride
    z = depth[v, u]
    xyz = np.column_stack(((u + pixel_center_offset - k[0, 2]) * z / k[0, 0],
                           (v + pixel_center_offset - k[1, 2]) * z / k[1, 1], z)).astype(np.float32)
    return xyz, rgb[v, u], dict(valid_pixels=int(valid.sum()), clipped_pixels=int((valid & ~selected).sum()),
                                sampled_points=len(z), stride_xy=stride)


def write_ply(path, xyz, rgb=None, faces=None):
    fields = [(axis, '<f4') for axis in 'xyz']
    if rgb is not None:
        fields += [(channel, 'u1') for channel in ('red', 'green', 'blue')]
    records = np.empty(len(xyz), dtype=fields)
    for i, axis in enumerate('xyz'):
        records[axis] = xyz[:, i]
    header = f'ply\nformat binary_little_endian 1.0\ncomment OpenCV camera frame, metres\nelement vertex {len(xyz)}\n'
    header += ''.join(f'property float {axis}\n' for axis in 'xyz')
    if rgb is not None:
        for i, channel in enumerate(('red', 'green', 'blue')):
            records[channel] = rgb[:, i]
            header += f'property uchar {channel}\n'
    triangles = b''
    if faces is not None:
        header += f'element face {len(faces)}\nproperty list uchar int vertex_indices\n'
        data = np.empty(len(faces), dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
        data['count'], data['indices'] = 3, faces
        triangles = data.tobytes()
    with Path(path).open('xb') as f:
        f.write((header + 'end_header\n').encode() + records.tobytes() + triangles)
