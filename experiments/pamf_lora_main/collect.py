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


def collect(submit, ssh_config, dataset, output):
    launch = read(submit)
    root = shlex.quote(launch['run_root'])
    connection = ['ssh', '-F', str(ssh_config), '-T', '-o', 'BatchMode=yes', '-o',
                  'ConnectTimeout=10', 'N607']
    retries = 0
    while True:
        try:
            result = subprocess.check_output([*connection, f'cat {root}/state.json'],
                                             timeout=20).decode('utf-8')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            retries += 1
            if retries >= 3:
                raise
            time.sleep(30)
            continue
        retries = 0
        status = json.loads(result)
        if status['status'] == 'FAILED':
            raise RuntimeError(status['error'])
        if status['status'] == 'PREDICTIONS_FROZEN':
            break
        time.sleep(60)
    raw = subprocess.check_output([*connection, f'tar -czf - -C {root} predictions source_decision.json'],
                                  timeout=120)
    output.mkdir(parents=True, exist_ok=False)
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
    score(output/'predictions', dataset, output/'scores.json', output)
    print('LORA_SCORED', output/'scores.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('submit', 'ssh-config', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    collect(args.submit, args.ssh_config, args.dataset, args.output)
