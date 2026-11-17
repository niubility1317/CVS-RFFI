"""Cross-day source/support training, followed by frozen prediction and scoring."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026092701)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = {'purpose': 'paired WiSig 256-point cross-day diagnostic; not paper reproduction',
        'source_train_per_class': 200, 'source_validation_per_class': 40,
        'support_per_class': [5, 10], 'target_query_per_class': 100,
        'source_and_target_receiver': '14-7', 'source_day': '2021-03-01', 'target_day': '2021-03-08',
        'seed': args.seed, 'device': args.device,
        'models': ['author source', 'legacy source', 'paper last layer', 'repository behavior', 'legacy project'],
        'caveat': 'Author Keras 2.2.4 compile-order behavior inferred statically; no paper-data claim.'}
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    python = [sys.executable, '-X', 'utf8', '-m']
    steps = [
        ('train', python + ['experiments.poster_paper_repro.diagnostic_train', '--training', args.training,
            '--output', str(root / 'models'), '--seed', str(args.seed), '--device', args.device]),
        ('prepare_query', python + ['experiments.poster_paper_repro.diagnostic_query',
            '--models', str(root / 'models'), '--indices', args.indices, '--output', str(root / 'query_pack')]),
        ('predict', python + ['experiments.poster_paper_repro.diagnostic_predict',
            '--models', str(root / 'models'), '--query', str(root / 'query_pack/query_input.npy'),
            '--output', str(root / 'prediction'), '--device', args.device]),
        ('score', python + ['experiments.poster_paper_repro.diagnostic_score',
            '--predictions', str(root / 'prediction'), '--truth', str(root / 'query_pack/truth_side_labels.npy'),
            '--output', str(root / 'score.json')])]
    elapsed = {}
    for name, command in steps:
        start = time.perf_counter()
        with (root / f'{name}.log').open('x', encoding='utf8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed[name] = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}; partial artifacts retained in {root}')
    (root / 'completion.json').write_text(json.dumps({'status': 'SCORED', 'steps_seconds': elapsed},
                                                    indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
