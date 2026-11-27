"""Collect and independently score the peer-spectral Full evaluation."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from time import monotonic, sleep

from .w24_formal import (
    LIMIT, PEER_CPU_RUNS, PEER_SPECTRAL_KIND, RUN_PEER_SPECTRAL_CPU, RUN_PEER_SPECTRAL_FULL, VIEWS, cases, receipt,
)
from .w24_formal_score import score


def collect(submit, dataset, output, ssh_config):
    remote = PurePosixPath(json.loads(submit.read_text(encoding='utf-8-sig'))['run_root'])
    if remote.name not in PEER_CPU_RUNS or not re.fullmatch(r'/[a-zA-Z0-9_./-]+', str(remote)):
        raise ValueError('Expected this immutable peer spectral Full run')
    flags = ['-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    matrix = cases(PEER_SPECTRAL_KIND)
    expected = dict(status='PREDICTIONS_FROZEN', cases=list(matrix), target_truth_read=False)
    prior = None
    deadline = monotonic() + 48*60*60
    while monotonic() < deadline:
        try:
            result = subprocess.run(['ssh', *flags, '-T', 'N607', 'cat', str(remote/'state.json')],
                                    timeout=25, capture_output=True, check=False)
            state = json.loads(result.stdout) if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            state = None
        if state is not None:
            status = state['status']
            if status != prior:
                print(state, flush=True)
                prior = status
            if status == 'FAILED':
                raise RuntimeError(state['error'])
            if status == 'SOURCE_NOT_QUALIFIED':
                if remote.name != RUN_PEER_SPECTRAL_CPU:
                    raise ValueError('Only the original conditional run may decline source qualification')
                return
            if status == 'MATCHED_FULL_EXISTS':
                if remote.name != RUN_PEER_SPECTRAL_FULL or state != dict(
                        status=status, previous_run=RUN_PEER_SPECTRAL_CPU, target_truth_read=False):
                    raise ValueError('Invalid peer spectral Full reuse marker')
                # The original run retains its own already-started independent collector.
                return
            if status == 'PREDICTIONS_FROZEN':
                if state != expected:
                    raise ValueError('The peer spectral Full matrix is not complete')
                break
        sleep(40)
    else:
        raise TimeoutError('Closed peer spectral Full matrix has not been observed')

    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(['scp', *flags, '-r', f'N607:{remote}/predictions', str(output)], check=True, timeout=240)
    for part in ('state.json', 'smoke.json', 'logs'):
        subprocess.run(['scp', *flags, '-r', f'N607:{remote}/{part}', str(output)], check=True, timeout=90)

    parity = {}
    for case in matrix:
        proof = receipt(case, True, third_prior=PEER_SPECTRAL_KIND)
        marker = output/'predictions'/case/'cpu-direct-parity.json'
        data = json.loads(marker.read_text(encoding='utf-8'))
        keys = {f'{rx}/{view}' for rx in proof['receivers'] for view in VIEWS}
        if data.keys() != keys or any(type(value) is not int or not 0 <= value <= LIMIT for value in data.values()):
            raise ValueError(f'Invalid CPU direct comparison for {case}')
        parity.update(data)
    score(output/'predictions', dataset, output/'scores', third_prior=PEER_SPECTRAL_KIND, by_transmitter=True)
    (output/'cpu-direct-parity.json').write_text(json.dumps(parity, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('submit', 'dataset', 'output', 'ssh-config'):
        parser.add_argument('--'+name, type=Path, required=True)
    collect(**vars(parser.parse_args()))
