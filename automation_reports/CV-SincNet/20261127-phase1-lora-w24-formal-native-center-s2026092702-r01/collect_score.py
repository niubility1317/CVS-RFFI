"""Collect this fixed Full run and score only after all predictions are frozen."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from time import monotonic, sleep


RUN = '20261127-phase1-lora-w24-formal-native-center-s2026092702-r01'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'submit', 'ssh-config', 'dataset', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    submit = json.loads(args.submit.read_text(encoding='utf-8-sig'))
    remote = PurePosixPath(submit['run_root'])
    if remote.name != RUN or not re.fullmatch(r'/[a-zA-Z0-9_./-]+', str(remote)):
        raise ValueError('Expected this immutable Full test run')
    flags = ['-F', str(args.ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    deadline = monotonic() + 43200
    previous = None
    while monotonic() < deadline:
        try:
            result = subprocess.run(['ssh', *flags, '-T', 'N607', 'cat', str(remote/'state.json')],
                                    capture_output=True, timeout=25, check=False)
            state = json.loads(result.stdout) if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            state = None
        if state is not None:
            if state['status'] != previous:
                print(state, flush=True)
                previous = state['status']
            if state['status'] == 'FAILED':
                raise RuntimeError(state['error'])
            if state['status'] == 'PREDICTIONS_FROZEN':
                break
        sleep(40)
    else:
        raise TimeoutError('Frozen prediction matrix not observed')

    cases = [f'full_native-{i}' for i in range(3)]
    if state != dict(status='PREDICTIONS_FROZEN', cases=cases, target_truth_read=False):
        raise ValueError('Incomplete or unexpected formal Full state')
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run(['scp', *flags, '-r', f'N607:{remote}/predictions', str(args.output)],
                   check=True, timeout=240)
    for part in ('state.json', 'smoke.json', 'logs'):
        subprocess.run(['scp', *flags, '-r', f'N607:{remote}/{part}', str(args.output)], check=True, timeout=90)
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'experiments.pamf_lora_main.w24_formal_score',
                    '--native-center-direction', '--predictions', str(args.output/'predictions'),
                    '--dataset', str(args.dataset), '--output', str(args.output/'scores')],
                   cwd=args.repo, check=True)


if __name__ == '__main__':
    main()
