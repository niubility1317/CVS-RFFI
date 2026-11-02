"""Predict all preregistered Phase1 ground scenes for one fixed source arm."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.phase1_dura_posterior.resources import wait_for_gpu
from experiments.phase1_dura_posterior.source_controller import ARMS, ROOT, RELEASE, RUN, PYTHON, environment


GROUND = {
    'mean': '20261102-phase1-dura-posterior-internal-ground-test-s2026092701-r01',
    'nonlocal': '20261102-phase1-dura-nonlocal-internal-ground-test-s2026092701-r01',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(GROUND), required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--source-pid', type=int, required=True)
    args = parser.parse_args()
    name = GROUND[args.arm]
    controller = ROOT/'runs'/(name+'-controller')
    ground = ROOT/'runs'/name
    source = ROOT/'runs'/ARMS[args.arm]
    source_controller = ROOT/'runs'/(RUN+'-source-controller')/'status.json'
    if not controller.is_dir() or any(controller.iterdir()) or ground.exists():
        raise ValueError('New immutable controller and ground output roots are required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('Registered release commit required')

    def status(state, **extra):
        temp = controller/'status.tmp'
        temp.write_text(json.dumps(dict(state=state, arm=args.arm, **extra), indent=2)+'\n', encoding='utf8')
        os.replace(temp, controller/'status.json')

    try:
        while not (source/'prediction_complete.json').exists():
            if source_controller.exists():
                data = json.loads(source_controller.read_text(encoding='utf8'))
                if data.get('state') == 'FAILED' or args.arm in data.get('failures', {}):
                    raise RuntimeError('The matching source arm failed technically')
            proc = Path('/proc')/str(args.source_pid)/'cmdline'
            if (not proc.is_file() or 'phase1_dura_posterior.source_controller' not in proc.read_bytes().decode(errors='replace')) and not (source/'prediction_complete.json').exists():
                raise RuntimeError('Source controller stopped before this arm completed')
            status('WAITING_FIXED_SOURCE', source_pid=args.source_pid)
            time.sleep(60)
        gpu = wait_for_gpu(lambda state, **extra: status(state, **extra))
        common = ['--root', str(ROOT), '--out', str(ground), '--commit', args.commit,
                  '--device', 'cuda:0', '--source-run', ARMS[args.arm]]
        for mode in ('smoke', 'predict'):
            status('RUNNING_SOURCE_REPLAY' if mode == 'smoke' else 'RUNNING_SEVEN_VIEW_PREDICTION', gpu=gpu)
            command = [PYTHON, '-X', 'utf8', '-u', '-m', 'experiments.phase1_dura_posterior.ground', mode, *common]
            with (controller/(mode+'.log')).open('x', encoding='utf8') as log:
                result = subprocess.run(command, cwd=RELEASE, env=environment(gpu), stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Ground '+mode+' exited '+str(result.returncode))
        manifest = json.loads((ground/'flow/test_prediction/complete.json').read_text(encoding='utf8'))
        if (manifest.get('status') != 'PREDICTIONS_COMPLETE' or manifest.get('count') != 168000
                or manifest.get('commit') != args.commit or manifest.get('TX_truth_reads') != 0
                or manifest.get('model_updates') != 0):
            raise ValueError('All seven fixed ground prediction arrays are required')
        status('PREDICTIONS_COMPLETE_AWAITING_AUTO_AUDIT')
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
