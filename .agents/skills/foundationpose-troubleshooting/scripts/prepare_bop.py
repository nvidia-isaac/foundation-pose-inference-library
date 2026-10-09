#!/usr/bin/env python3
"""Make one lossless registration capture with an occlusion-aware BOP GT mask."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from bop_geometry import read_binary_little_endian_ply, rasterize_mesh
from numeric import file_identity, validate_frame, validate_pose, write_json


def render_visible(entries, meshes, k, shape):
    depth = np.full(shape, np.inf, np.float64)
    objects = np.zeros(shape, np.int32)
    instances = np.zeros(shape, np.int32)
    for index, entry in enumerate(entries):
        vertices, faces = meshes[entry['obj_id']]
        rasterize_mesh(vertices, faces, np.array(entry['cam_R_m2c']).reshape(3, 3),
                       np.array(entry['cam_t_m2c']), k, depth, objects, instances,
                       entry['obj_id'], index+1, 1.0)
    depth[~np.isfinite(depth)] = 0
    return instances, (depth/1000).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True, help='BOP scene directory containing scene_gt.json')
    p.add_argument('--image-id', type=int, default=0)
    p.add_argument('--gt-index', type=int, required=True, help='zero-based index in scene_gt.json')
    p.add_argument('--models', type=Path, required=True, help='obj_NNNNNN.ply, mesh coordinates in mm')
    p.add_argument('--rgb', type=Path, required=True)
    p.add_argument('--depth', type=Path, required=True, help='NPY or single-channel image; camera Z')
    p.add_argument('--depth-scale-to-m', type=float, required=True, help='explicit multiplier; 1 for NPY meters, .001 for mm PNG')
    p.add_argument('--depth-provenance', required=True, help='measured, stereo-estimated, or GT-rendered; describe source')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        if not np.isfinite(a.depth_scale_to_m) or a.depth_scale_to_m <= 0: raise ValueError('invalid depth scale')
        camera_path, gt_path = a.scene/'scene_camera.json', a.scene/'scene_gt.json'
        entries = json.loads(gt_path.read_text())[str(a.image_id)]
        if not 0 <= a.gt_index < len(entries): raise ValueError('gt-index out of range')
        k = np.array(json.loads(camera_path.read_text())[str(a.image_id)]['cam_K'], np.float32).reshape(3, 3)
        bgr = cv2.imread(str(a.rgb), cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError('cannot read RGB')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw = np.load(a.depth, allow_pickle=False) if a.depth.suffix == '.npy' else cv2.imread(str(a.depth), cv2.IMREAD_UNCHANGED)
        if raw is None: raise ValueError('cannot read depth')
        depth = (raw.astype(np.float32)*a.depth_scale_to_m).astype(np.float32)
        frame = dict(rgb=rgb, depth_m=depth, K=k)
        validate_frame(frame)
        paths = {oid: a.models/f'obj_{oid:06d}.ply' for oid in {e['obj_id'] for e in entries}}
        meshes = {oid: read_binary_little_endian_ply(path) for oid, path in paths.items()}
        inst, rendered = render_visible(entries, meshes, k, depth.shape)
        frame['mask'] = ((inst == a.gt_index+1)*255).astype(np.uint8)
        validate_frame(frame, require_mask=True)
        entry = entries[a.gt_index]
        truth = np.eye(4); truth[:3, :3] = np.array(entry['cam_R_m2c']).reshape(3, 3)
        truth[:3, 3] = np.array(entry['cam_t_m2c'])/1000
        validate_pose(truth)
        a.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(a.output/'frame.npz', **frame)
        np.save(a.output/'reference_pose.npy', truth)
        np.save(a.output/'vertices_m.npy', meshes[entry['obj_id']][0]/1000)
        info_path = a.models/'models_info.json'
        if info_path.exists():
            info = json.loads(info_path.read_text())[str(entry['obj_id'])]
            if info.get('symmetries_continuous'):
                raise ValueError('continuous symmetries need an explicit discretization; partial output retained')
            syms = [np.eye(4)] + [np.array(s).reshape(4, 4) for s in info.get('symmetries_discrete', [])]
            syms = np.array(syms); syms[:, :3, 3] /= 1000
            np.save(a.output/'symmetries_m.npy', syms)
        np.savez_compressed(a.output/'visible_geometry.npz', instance_id=inst, rendered_depth_m=rendered)
        cv2.imwrite(str(a.output/'visible_mask.png'), frame['mask'])
        write_json(a.output/'manifest.json', dict(object_id=entry['obj_id'], scene=str(a.scene),
                   image_id=a.image_id, gt_index=a.gt_index, depth_provenance=a.depth_provenance,
                   frames=[dict(id=f'{a.scene.name}:{a.image_id}:{a.gt_index}', path='frame.npz', action='register')]))
        write_json(a.output/'provenance.json', dict(inputs=[file_identity(x) for x in [a.rgb,a.depth,camera_path,gt_path,*paths.values()]],
                   depth_scale_to_m=a.depth_scale_to_m, depth_provenance=a.depth_provenance,
                   mask='shared depth buffer over all annotated instances; no unannotated occluders',
                   mesh_and_gt_input_units='mm', output_pose_units='meters'))
        print(a.output)
    except Exception as e: p.exit(2, f'error: {e}\n')


if __name__ == '__main__': main()
