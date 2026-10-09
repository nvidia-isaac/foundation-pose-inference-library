#!/usr/bin/env python3
"""Attribute a completed registration trace to candidate coverage or ranking."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from mesh_metrics import mesh_error
from numeric import load_array, validate_pose, write_json


def plot_analysis(rows, report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    stages = report['stages'][:-1]
    axes[0].plot([s['stage'] for s in stages], [s['selected_trajectory_mm'] for s in stages], 'o-', label='Selected candidate trajectory')
    axes[0].plot([s['stage'] for s in stages], [s['oracle_mm'] for s in stages], 'o-', label='Best candidate at each stage')
    axes[0].set(xlabel='Refinement stage', ylabel='Vertex Hausdorff error (mm)', title='Convergence and candidate coverage')
    axes[0].legend(fontsize=8)
    final = [r for r in rows if r['stage'] == 'scored']
    axes[1].scatter([r['vertex_hausdorff_mm'] for r in final], [r['final_score'] for r in final], s=18, label='Final candidates')
    selected = next(r for r in final if r['selected'])
    axes[1].scatter([selected['vertex_hausdorff_mm']], [selected['final_score']], color='red', marker='x', s=70, label='Selected')
    axes[1].axvline(report['threshold_mm'], color='black', ls='--')
    axes[1].set(xlabel='Vertex Hausdorff error (mm)', ylabel='Raw scorer output', title='Ranking (larger score wins)')
    axes[1].legend(fontsize=8)
    axes[0].axhline(report['threshold_mm'], color='black', ls='--')
    fig.savefig(path, dpi=160); plt.close(fig)


def analyze(trace, reference, vertices, symmetries, threshold_mm=5):
    trace = Path(trace)
    if not (trace/'complete').is_file():
        raise ValueError('trace has no completion marker')
    metadata = json.loads((trace/'metadata.json').read_text())
    if metadata['phase'] != 'register':
        raise ValueError('candidate ranking analysis requires registration')
    c = validate_pose(np.load(trace/'centered_from_original.npy', allow_pickle=False))
    winner = int(np.load(trace/'selected_candidate_id.npy', allow_pickle=False).item())
    scores = np.load(trace/'scores.npy', allow_pickle=False).reshape(-1)
    if not np.isfinite(scores).all() or not 0 <= winner < len(scores):
        raise ValueError('invalid candidate ID or nonfinite scores')
    stages = [('initial', 'initial_poses_centered.npy')]
    stages += [(f'after_{i+1}', f'refine_{i}_poses_after_centered.npy') for i in range(metadata['iterations'])]
    stages += [('scored', 'score_poses_centered.npy')]
    rows, stage_summary = [], []
    final = None
    for stage, filename in stages:
        poses = np.load(trace/filename, allow_pickle=False)
        if poses.shape != (len(scores), 4, 4):
            raise ValueError(f'{filename}: candidate count mismatch')
        current = []
        for index, centered in enumerate(poses):
            row = dict(stage=stage, candidate_id=index, selected=index == winner,
                       final_score=float(scores[index]), **mesh_error(centered @ c, reference, vertices, symmetries))
            rows.append(row); current.append(row)
        oracle = min(current, key=lambda r: r['vertex_hausdorff_mm'])
        stage_summary.append(dict(stage=stage, oracle_id=oracle['candidate_id'],
                                  oracle_mm=oracle['vertex_hausdorff_mm'],
                                  selected_trajectory_mm=current[winner]['vertex_hausdorff_mm'],
                                  candidates_within_threshold=sum(r['vertex_hausdorff_mm'] <= threshold_mm for r in current)))
        if stage == 'scored': final = poses[winner] @ c
    selected = np.load(trace/'selected_pose_original.npy', allow_pickle=False)
    if not np.allclose(final, selected, atol=2e-6, rtol=1e-6):
        raise ValueError('selected output differs from centered candidate converted to original mesh')
    last = stage_summary[-1]
    conclusion = ('selected_within_threshold' if last['selected_trajectory_mm'] <= threshold_mm else
                  'ranking_missed_passing_candidate' if last['oracle_mm'] <= threshold_mm else
                  'no_passing_final_candidate')
    return rows, dict(trace=str(trace), threshold_mm=threshold_mm, metric='vertex_hausdorff_mm',
                      selected_candidate=winner, selected_score=float(scores[winner]),
                      selected_is_argmax=bool(scores[winner] == scores.max()), stages=stage_summary,
                      diagnosis=conclusion,
                      caveat='Oracle uses GT only for diagnosis. No passing candidate does not isolate initialization, input, or refiner by itself.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--vertices-m', type=Path, required=True)
    p.add_argument('--symmetries-m', type=Path)
    p.add_argument('--threshold-mm', type=float, default=5)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot', action='store_true', help='also export convergence/ranking PNG; requires matplotlib')
    a = p.parse_args()
    try:
        if not np.isfinite(a.threshold_mm) or a.threshold_mm <= 0: raise ValueError('threshold must be positive')
        rows, report = analyze(a.trace, load_array(a.reference), load_array(a.vertices_m),
                               load_array(a.symmetries_m) if a.symmetries_m else None, a.threshold_mm)
        a.output.mkdir(parents=True, exist_ok=False)
        write_json(a.output/'summary.json', report)
        with (a.output/'candidates.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        if a.plot: plot_analysis(rows, report, a.output/'candidate_diagnostics.png')
        print(json.dumps(report, indent=2))
    except Exception as e:
        p.exit(2, f'error: {e}\n')


if __name__ == '__main__': main()
