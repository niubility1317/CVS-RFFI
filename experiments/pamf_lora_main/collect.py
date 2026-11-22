"""Collect a frozen N607 prediction matrix and score against the local archives."""
import argparse
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time

from experiments.cric_dura_comparison.io import read
from .scoring import score


def wait_for_predictions(connection, root):
    retries = 0
    while True:
        try:
            result = subprocess.check_output([*connection, f'cat {root}/state.json'],
                                             timeout=20).decode('utf-8')
            status = json.loads(result)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            retries += 1
            if retries >= 3:
                raise
            time.sleep(30)
            continue
        retries = 0
        if status['status'] == 'FAILED':
            raise RuntimeError(status['error'])
        if status['status'] == 'PREDICTIONS_FROZEN':
            return
        time.sleep(60)

def receive(connection, root, output, main, rows=()):
    files = 'predictions source_decision.json' if main else 'predictions'
    if rows:
        files = ' '.join(shlex.quote('predictions/'+row) for row in rows)
    raw = subprocess.check_output([*connection, f'tar -czf - -C {root} {files}'], timeout=120)
    with tarfile.open(fileobj=BytesIO(raw), mode='r:gz') as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if (name.is_absolute() or '..' in name.parts or name.parts[0] not in
                    ('predictions', 'source_decision.json')):
                raise ValueError('Unexpected N607 scoring artifact path')
            if entry.isdir():
                continue
            if not entry.isfile() or name.suffix not in ('.json', '.npy'):
                raise ValueError('Unexpected N607 scoring artifact type')
            path = output.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(archive.extractfile(entry).read())


def collect(submit, ssh_config, dataset, output, supplement=None, control=None):
    root = shlex.quote(read(submit)['run_root'])
    connection = ['ssh', '-F', str(ssh_config), '-T', '-o', 'BatchMode=yes', '-o',
                  'ConnectTimeout=10', 'N607']
    supplement_root = read(supplement)['run_root'] if supplement else None
    extra = shlex.quote(supplement_root) if supplement_root else None
    score_options = {}
    control_root = None
    if extra:
        from .phase1 import LONG_PL_RUN, LONG_SUP_RUN, OPTIMIZATION_RUN, RUN, registered_rows
        run_id = PurePosixPath(supplement_root).name
        if run_id == LONG_SUP_RUN:
            matched = registered_rows(run_id)
            score_options = dict(seeds=tuple(dict.fromkeys(s for _,s in matched)))
        if run_id == LONG_PL_RUN and not control:
            raise ValueError('Long PL scoring requires the matched long supervised run')
    if control:
        if not extra:
            raise ValueError('A comparison requires a supplement')
        control_root = read(control)['run_root']
        controls = {OPTIMIZATION_RUN: (RUN, 'r02'), LONG_PL_RUN: (LONG_SUP_RUN, 'sup')}
        if run_id not in controls or PurePosixPath(control_root).name != controls[run_id][0]:
            raise ValueError('Wrong LoRa comparison control')
        suffix = controls[run_id][1]
        control_root = shlex.quote(control_root)
        matched = registered_rows(run_id)
        score_options = dict(seeds=tuple(dict.fromkeys(s for _,s in matched)),
                             paired_controls={family: f'{family}_{suffix}' for family,_ in matched})
    if extra:
        wait_for_predictions(connection, extra)
    wait_for_predictions(connection, root)
    if control_root:
        wait_for_predictions(connection, control_root)
    output.mkdir(parents=True, exist_ok=False)
    receive(connection, root, output, main=True)
    methods = ()
    if extra:
        receive(connection, extra, output, main=False)
        if control_root:
            rows = [f'{family}-s{seed}' for family,seed in matched]
            control_output = output/f'{suffix}_control'
            receive(connection, control_root, control_output, main=False, rows=rows)
            for family, seed in matched:
                name = f'{family}-s{seed}'
                dest = output/'predictions'/f'{family}_{suffix}-s{seed}'
                if dest.exists():
                    raise FileExistsError(dest)
                (control_output/'predictions'/name).rename(dest)
            methods = tuple(score_options['paired_controls']) + tuple(score_options['paired_controls'].values())
        else:
            from .phase1 import METHODS as methods
    score(output/'predictions', dataset, output/'scores.json', output, extended=methods, **score_options)
    print('LORA_SCORED', output/'scores.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('submit', 'ssh-config', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    collect(args.submit, args.ssh_config, args.dataset, args.output, supplement=args.supplement, control=args.control)
