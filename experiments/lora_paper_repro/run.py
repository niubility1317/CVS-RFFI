"""Run the author-checkpoint residential benchmark and truth-last scoring."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--author-preprocessing', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = {'method': 'LoRa author pretrained Extractor 1', 'own_training': False,
        'dataset': str(Path(args.dataset).resolve()), 'checkpoint': str(Path(args.checkpoint).resolve()),
        'author_preprocessing': str(Path(args.author_preprocessing).resolve()),
        'device': args.device, 'device_numbers': list(range(31, 41)),
        'enrollment_packet_indices': list(range(100)), 'query_packet_indices': list(range(100, 200)),
        'neighbors': 15, 'scenario': 'residential enrollment and residential identification',
        'paper_target': .984, 'paper_target_source': 'Figure 9, Extractor 1, DUT 31-40',
        'paper_exact_packet_indices': 'not independently specified'}
    (root / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    python = [sys.executable, '-X', 'utf8', '-m']
    steps = [
        ('smoke', python + ['experiments.lora_paper_repro.smoke', '--checkpoint', args.checkpoint,
            '--dataset', args.dataset, '--author-preprocessing', args.author_preprocessing,
            '--output', str(root / 'smoke.json'), '--device', args.device]),
        ('prepare', python + ['experiments.lora_paper_repro.prepare', '--dataset', args.dataset,
            '--author-preprocessing', args.author_preprocessing, '--output', str(root / 'prepared')]),
        ('predict', python + ['experiments.lora_paper_repro.predict', '--checkpoint', args.checkpoint,
            '--enrollment-input', str(root / 'prepared/enrollment_input.npy'),
            '--enrollment-labels', str(root / 'prepared/enrollment_labels.npy'),
            '--query-input', str(root / 'prepared/query_input.npy'), '--output', str(root / 'prediction'),
            '--device', args.device]),
        ('score', python + ['experiments.lora_paper_repro.score', '--predictions', str(root / 'prediction'),
            '--truth', str(root / 'prepared/truth_side_labels.npy'), '--output', str(root / 'score.json')])]
    durations = {}
    for name, command in steps:
        start = time.perf_counter()
        with (root / f'{name}.log').open('x', encoding='utf8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        durations[name] = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}; artifacts retained in {root}')
    (root / 'completion.json').write_text(json.dumps({'status': 'SCORED', 'steps_seconds': durations},
                                                    indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
