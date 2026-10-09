#!/usr/bin/env python3
"""Replay captured network inputs through an existing TensorRT engine, without CAD."""
import argparse
import subprocess
from pathlib import Path
import numpy as np
from numeric import compare_arrays, file_identity, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--stage', required=True, help='refine_0, refine_1, ... or score')
    p.add_argument('--runner', type=Path, required=True, help='compiled assets/replay_network.cpp executable')
    p.add_argument('--engine', type=Path, required=True, help='trusted local engine from the captured run')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-4)
    a = p.parse_args()
    try:
        if not (a.trace/'complete').is_file(): raise ValueError('incomplete trace')
        if a.stage != 'score' and not (a.stage.startswith('refine_') and a.stage[7:].isdigit()):
            raise ValueError('stage must be score or refine_N')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(a.runner.resolve()), str(a.engine),
                        str(a.trace/f'{a.stage}_inputA.npy'), str(a.trace/f'{a.stage}_inputB.npy'),
                        str(a.output)], check=True)
        report = dict(engine=file_identity(a.engine), runner=file_identity(a.runner),
                      trace=str(a.trace), stage=a.stage, outputs={})
        expected = {'score': 'scores'} if a.stage == 'score' else {'trans': f'{a.stage}_delta_translation_raw', 'rot': f'{a.stage}_delta_rotation_raw'}
        if a.stage == 'score' and not (a.output/'score.npy').exists():
            outputs = list(a.output.glob('*.npy'))
            if len(outputs) != 1: raise ValueError('ambiguous scorer output mapping')
            expected = {outputs[0].stem: 'scores'}
        for name in expected:
            value = np.load(a.output/f'{name}.npy', allow_pickle=False)
            reference = np.load(a.trace/f'{expected[name]}.npy', allow_pickle=False)
            if value.size != reference.size: raise ValueError('output element count mismatch')
            report['outputs'][name] = compare_arrays(value.reshape(reference.shape), reference, a.atol, a.rtol)
        report['passes'] = bool(report['outputs']) and all(r['passes'] for r in report['outputs'].values())
        write_json(a.output/'comparison.json', report)
        print(f"{'PASS' if report['passes'] else 'FAIL'}: {a.output}")
        return 0 if report['passes'] else 1
    except Exception as e: p.exit(2, f'error: {e}\n')


if __name__ == '__main__': raise SystemExit(main())
