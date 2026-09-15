"""Bounded synthetic integration check; never a paper accuracy experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compat-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(12)
    classes = np.array(['a', 'b', 'c', 'd', 'e'])
    x = rng.normal(size=(30, 288, 2)).astype('float32')
    np.savez(args.output / 'source.npz', x=x, y=np.arange(30) % 5, classes=classes)
    np.savez(args.output / 'target.npz', x=x * 0.8 + 0.1, y=np.arange(30) % 5, classes=classes)
    np.savez(args.output / 'query.npz', x=x[:5], classes=classes)
    results = []
    for paper in ['poster', 'radionet']:
        for stage, data in [('train', 'source'), ('tune', 'target'), ('predict', 'query')]:
            out = args.output / f'{paper}_{stage}'
            command = [sys.executable, str(Path(__file__).with_name('li_author_server.py')),
                       '--compat-root', args.compat_root, '--paper', paper,
                       '--stage', stage, '--data', str(args.output / f'{data}.npz'),
                       '--output', str(out), '--epochs', '1', '--batch-size', '10']
            if stage != 'train':
                parent = 'train' if stage == 'tune' else 'tune'
                command += ['--checkpoint', str(args.output / f'{paper}_{parent}' / 'final.keras')]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (args.output / f'{paper}_{stage}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            results.append(dict(paper=paper, stage=stage, returncode=result.returncode))
            (args.output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            if result.returncode:
                print(result.stderr[-4000:])
                raise RuntimeError(f'{paper}/{stage} failed')
            assert (out / 'completed.json').is_file()
        with np.load(out / 'predictions.npz') as pred:
            assert pred['probabilities'].shape == (5, 5)
            assert np.isfinite(pred['probabilities']).all()
            np.testing.assert_allclose(pred['probabilities'].sum(axis=1), 1, atol=1e-5)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
