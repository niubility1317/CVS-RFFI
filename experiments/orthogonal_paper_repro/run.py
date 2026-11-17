"""Train fixed stages, then predict, then independently score a paper-style row."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True)
    parser.add_argument('--training-indices', required=True)
    parser.add_argument('--truth-indices', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026092701)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    python = [sys.executable, '-X', 'utf8', '-m']
    steps = [
        ('train', python + ['experiments.orthogonal_paper_repro.train', '--training', args.training,
            '--output', str(root / 'training'), '--seed', str(args.seed), '--device', args.device]),
        ('prepare_query', python + ['experiments.orthogonal_paper_repro.query_pack',
            '--training-output', str(root / 'training'), '--training-indices', args.training_indices,
            '--truth-indices', args.truth_indices, '--output', str(root / 'query_pack')]),
        ('predict', python + ['experiments.orthogonal_paper_repro.predict',
            '--checkpoints', str(root / 'training'), '--inputs', str(root / 'query_pack/prediction_input.npy'),
            '--stage-counts', str(root / 'query_pack/stage_counts.json'), '--output', str(root / 'prediction'),
            '--device', args.device]),
        ('score', python + ['experiments.orthogonal_paper_repro.score', '--predictions', str(root / 'prediction'),
            '--truth', str(root / 'query_pack/truth_side.npz'), '--output', str(root / 'score.json')])]
    elapsed = {}
    for name, command in steps:
        start = time.perf_counter()
        with (root / f'{name}.log').open('x', encoding='utf8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed[name] = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}; retained {root}')
    (root / 'completion.json').write_text(json.dumps({'status': 'SCORED', 'steps_seconds': elapsed},
                                                    indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
