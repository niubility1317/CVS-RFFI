"""Build reviewable argv from the existing baseline; never execute training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/phase1_adv3b02_pair_reform_pending.json'


def build_commands(*, root: str, python: str, dataset: str, checkpoint: str,
                   output_root: str, rows: list[str] | None = None) -> dict:
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    worker = (ROOT / config['baseline_worker']).read_text(encoding='utf-8')
    # Parse only literal assignments and the literal TRAIN_CMD argument array.
    # This is data parsing, not a shell interpreter or Bash invocation.
    defaults = dict(re.findall(r'^(\w+)="\$\{\1:-(.*)\}"$', worker, re.M))
    env = {**defaults, **config['inherited_environment'], 'ROOT': root, 'CODE_ROOT': root,
           'WISIG_PKL': dataset, 'BASE_CKPT': checkpoint, 'level': 'M1'}
    lines = worker.split('    --wisig_pkl ', 1)[1].split('\n  )', 1)[0]
    template = shlex.split('    --wisig_pkl ' + lines)
    chosen = rows or list(config['rows'])
    unknown = set(chosen) - set(config['rows']) - set(config['aliases'])
    if unknown:
        raise ValueError(f'unknown rows: {sorted(unknown)}')
    result = []
    for requested in chosen:
        row = config['aliases'].get(requested, requested)
        candidate = f'PAIR_{row}_S{env["SEED"]}'
        values = {**env, 'candidate_root': str(Path(output_root)/candidate),
                  'candidate_id': candidate, 'RUN_ID': 'adv3b02_pair_reform_pending'}
        def expand(token):
            for _ in range(4):
                if '${' not in token:
                    return token
                token = re.sub(r'\$\{(\w+)\}', lambda m: str(values[m[1]]), token)
            raise ValueError(f'unsupported inherited shell expression: {token}')
        args = [expand(token) for token in template]
        overrides = {**config['common_overrides'], **config['rows'][row]}
        # Replace option values instead of relying on duplicate CLI precedence.
        for key, value in overrides.items():
            flag = '--' + key
            value = str(value).lower() if isinstance(value, bool) else str(value)
            if flag in args:
                args[args.index(flag)+1] = value
            else:
                args += [flag, value]
        args += ['--from_scratch', 'false', '--baseline_ckpt', checkpoint]
        result.append({'row': row, 'requested': requested, 'status': 'PENDING_NOT_RUN',
                       'argv': [python, '-u', str(Path(root)/'code/SSDG/train_ssdg.py'), *args],
                       'environment': {'PYTHONPATH': str(Path(root)/'code')},
                       'cwd': root})
    return {'schema': config['schema'], 'status': 'PENDING_NOT_RUN',
            'baseline_worker': config['baseline_worker'], 'claims': config['claims'],
            'commands': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'dataset', 'checkpoint', 'output-root'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--rows', help='comma-separated subset; A_POINT aliases M0')
    args = parser.parse_args()
    result = build_commands(root=args.root, python=args.python, dataset=args.dataset,
        checkpoint=args.checkpoint, output_root=args.output_root,
        rows=args.rows.split(',') if args.rows else None)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
