#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a self-contained per-frame HTML point-cloud/CAD viewer and PLY exports.

Generation needs NumPy, OpenCV and Plotly. Viewing needs only a WebGL browser;
no server, CDN, inference, Python installation, or external data is required.
"""
import argparse
import base64
import html
from pathlib import Path
import cv2
import numpy as np
from numeric import file_identity, load_array, load_frame, validate_pose, write_json
from visual_geometry import load_mesh, load_reference, point_cloud, render_depth, transform, write_ply


def build(frame, estimated_pose, vertices, faces, output, reference=None, region_name='mask',
          stride=2, limits=(0, np.inf), pixel_center_offset=0.0):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if region_name not in ('mask', 'scene'):
        raise ValueError('region must be mask or scene')
    if region_name == 'mask' and 'mask' not in frame:
        raise ValueError('capture has no mask; select --region scene explicitly')
    region = frame['mask'] != 0 if region_name == 'mask' else np.ones(frame['depth_m'].shape, bool)
    views = [('Recorded depth + CAD at estimate', frame['depth_m'], estimated_pose, 'observed')]
    if reference is not None:
        label = {'rgb-derived': 'RGB-derived pseudo reference', 'measured-pose-render': 'CAD render at measured pose',
                 'estimate-render': 'CAD render at estimated pose'}[str(reference['reference_kind'])]
        views.insert(0, (label, reference['depth_m'], reference['pose'], 'reference'))
    fig = make_subplots(rows=1, cols=len(views), specs=[[{'type': 'scene'} for _ in views]],
                        subplot_titles=[v[0] for v in views])
    point_traces, mesh_traces, all_points, exports, summary = [], [], [], [], {}
    overlay = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
    for column, (_, depth, pose, role) in enumerate(views, 1):
        xyz, colors, counts = point_cloud(depth, frame['rgb'], frame['K'], region, stride, limits, pixel_center_offset)
        cad = transform(vertices, pose)
        all_points.extend([xyz, cad])
        summary[role] = counts
        point_traces.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(),
            mode='markers', name=f'{role}: cloud', marker=dict(size=2, color=[f'rgb({r},{g},{b})' for r, g, b in colors]),
            hovertemplate='X %{x:.5f} m<br>Y %{y:.5f} m<br>Z %{z:.5f} m<extra></extra>'), row=1, col=column)
        mesh_traces.append(len(fig.data))
        fig.add_trace(go.Mesh3d(x=cad[:, 0].tolist(), y=cad[:, 1].tolist(), z=cad[:, 2].tolist(),
            i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
            color='#15a0a6' if role == 'reference' else '#d67a32', opacity=.35, flatshading=True,
            name=f'{role}: CAD', showlegend=True, hoverinfo='skip'), row=1, col=column)
        # A marker at the public mesh origin makes corner-origin changes visible.
        fig.add_trace(go.Scatter3d(x=[pose[0, 3]], y=[pose[1, 3]], z=[pose[2, 3]], mode='markers',
            marker=dict(size=4, symbol='cross', color='#222222'), name=f'{role}: CAD origin',
            hovertemplate='Original mesh origin<extra></extra>'), row=1, col=column)
        exports.extend([(f'{role}_cloud.ply', xyz, colors, None), (f'{role}_cad.ply', cad, None, faces)])
        rendered = render_depth(vertices, faces, pose, frame['K'], depth.shape, pixel_center_offset)
        contours = cv2.findContours((rendered > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(overlay, contours, -1, (166, 160, 21) if role == 'reference' else (50, 122, 214), 1)
    combined = np.concatenate(all_points)
    lo, hi = combined.min(0), combined.max(0)
    span = max(float((hi - lo).max()), .001)
    for column in range(1, len(views) + 1):
        scene = 'scene' if column == 1 else f'scene{column}'
        axes = {f'{axis}axis': dict(title=f'{axis.upper()} (m)', range=[lo[i] - .03 * span, hi[i] + .03 * span])
                for i, axis in enumerate('xyz')}
        fig.update_layout(**{scene: dict(**axes, aspectmode='data',
            camera=dict(eye=dict(x=.8, y=-.7, z=-1.8), up=dict(x=0, y=-1, z=0)))})
    fig.update_layout(height=660, margin=dict(l=10, r=10, t=60, b=20), template='plotly_white',
        legend=dict(orientation='h'), uirevision='same-camera')
    # Synchronize camera changes, never duplicate transforms or alter data units.
    post = '''const plot = document.getElementById('inspection');
let synchronizing = false;
plot.on('plotly_relayout', event => {
  if (synchronizing || !plot.layout.scene2) return;
  const source = event['scene.camera'] ? 'scene' : event['scene2.camera'] ? 'scene2' : null;
  if (!source) return;
  synchronizing = true;
  Plotly.relayout(plot, {[source === 'scene' ? 'scene2.camera' : 'scene.camera']: event[source + '.camera']})
    .finally(() => { synchronizing = false; });
});'''
    graph = fig.to_html(full_html=False, include_plotlyjs=True, div_id='inspection', post_script=post)
    ok, encoded = cv2.imencode('.png', overlay)
    if not ok:
        raise ValueError('could not encode RGB overlay')
    image = base64.b64encode(encoded).decode()
    counts_text = '; '.join(f'{role}: {v["sampled_points"]} displayed points, {v["clipped_pixels"]} valid pixels excluded by depth range'
                            for role, v in summary.items())
    report = dict(views=summary, region=region_name, stride_xy=stride,
        depth_range_m=[limits[0], limits[1] if np.isfinite(limits[1]) else None],
        mesh_vertices=len(vertices), mesh_triangles=len(faces), pixel_center_offset=pixel_center_offset,
        coordinates='OpenCV camera XYZ in metres; original-frame cam_T_obj; no mesh decimation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, xyz, rgb, triangles in exports:
        write_ply(output / name, xyz, rgb, triangles)
    links = ' · '.join(f'<a href="{name}">{name}</a>' for name, *_ in exports)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Pose / depth inspection</title><style>body{{font:15px system-ui;margin:24px;color:#243640}}img{{max-width:100%}}label{{margin-right:24px}}.note{{max-width:1100px}}#inspection{{width:100%}}</style>
<h1>Pose / depth inspection</h1><p class="note">Orbit, zoom and pan in the 3D views. Cameras are synchronized. Click legend entries to toggle clouds, CAD and origins.</p>
<p>{html.escape(counts_text)}. Region: {region_name}; sampling: every {stride} pixels in x/y. Invalid depth is omitted. Coordinates: metres.</p>
<label>CAD opacity <input id="opacity" type="range" min="0" max="1" value=".35" step=".05"></label>
<label>Point size <input id="point-size" type="range" min="1" max="8" value="2"></label>
{graph}<h2>RGB with projected CAD outlines</h2><p>Orange: estimate. Teal: reference, when supplied. This is a full mesh silhouette and can include surfaces behind occluders.</p>
<img alt="RGB with projected CAD silhouettes" src="data:image/png;base64,{image}">
<p class="note">A cloud rendered from this CAD agrees with the reference CAD by construction. This is not independent proof of shape, calibration, or pose accuracy. RGB-derived poses are pseudo references, not measured ground truth. CAD uses neutral shading; projected RGB cloud colors are not CAD texture. Recorded depth is shown before FP preprocessing.</p>
<p>PLY exports (camera metres): {links}. HTML alone contains the visualization; include these files to share exports too.</p>
<script>document.getElementById('opacity').oninput=e=>Plotly.restyle('inspection',{{opacity:Number(e.target.value)}},{mesh_traces});
document.getElementById('point-size').oninput=e=>Plotly.restyle('inspection',{{'marker.size':Number(e.target.value)}},{point_traces});</script></html>'''
    (output / 'index.html').write_text(document, encoding='utf-8')
    write_json(output / 'inspection.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--pose', type=Path, required=True, help='estimated original-frame cam_T_obj')
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--mesh-unit-scale', type=float, required=True)
    p.add_argument('--reference', type=Path, help='reference NPZ from pseudo_reference.py render')
    p.add_argument('--region', choices=['mask', 'scene'], default='mask')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--depth-range-m', nargs=2, type=float, default=[0, np.inf])
    p.add_argument('--pixel-center-offset', type=float, choices=[0, .5], default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        frame, pose = load_frame(a.frame), validate_pose(load_array(a.pose))
        vertices, faces = load_mesh(a.mesh, a.mesh_unit_scale)
        reference = load_reference(a.reference, frame, a.pixel_center_offset) if a.reference else None
        if reference is not None and 'source_frame_sha256' in reference:
            if str(reference['source_frame_sha256']) != file_identity(a.frame)['sha256']:
                raise ValueError('reference was rendered for a different capture')
        build(frame, pose, vertices, faces, a.output, reference, a.region, a.stride, a.depth_range_m, a.pixel_center_offset)
        print(a.output / 'index.html')
    except Exception as e:
        p.exit(2, f'error: {e}\n')


if __name__ == '__main__':
    main()
