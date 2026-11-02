"""Wait for the fixed independent model and predict all Phase1 ground views."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.phase1_dura_posterior.resources import gpu_lease
from experiments.phase1_equivariant_dictionary.source_controller import ROOT, RUN, RELEASE, GPU_LOCK, PYTHON, environment


GROUND = '20261102-phase1-equivariant-dictionary-ground-test-s2026092701-r02'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--source-pid', type=int, required=True)
    args = parser.parse_args()
    controller = ROOT/'runs'/(GROUND+'-controller')
    output = ROOT/'runs'/GROUND
    source = ROOT/'runs'/RUN
    source_status = ROOT/'runs'/(RUN+'-controller')/'status.json'
    if not controller.is_dir() or any(controller.iterdir()) or output.exists():
        raise ValueError('Unique ground controller and output roots are required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('Registered release commit required')

    def status(state, **extra):
        temporary = controller/'status.tmp'
        temporary.write_text(json.dumps(dict(state=state, **extra), indent=2)+'\n', encoding='utf8')
        os.replace(temporary, controller/'status.json')

    try:
        while not (source/'prediction_complete.json').exists():
            if source_status.exists():
                upstream = json.loads(source_status.read_text(encoding='utf8'))
                if upstream.get('state') == 'FAILED':
                    raise RuntimeError('Independent source training failed technically')
            process = Path('/proc')/str(args.source_pid)/'cmdline'
            if (not process.is_file() or 'phase1_equivariant_dictionary.source_controller' not in process.read_bytes().decode(errors='replace')) and not (source/'prediction_complete.json').exists():
                raise RuntimeError('Source controller stopped before fixed predictions')
            status('WAITING_FIXED_SOURCE', source_pid=args.source_pid)
            time.sleep(60)
        common = ['--root', str(ROOT), '--out', str(output), '--commit', args.commit, '--device', 'cuda:0']
        with gpu_lease(GPU_LOCK, status) as (gpu, lock_fd):
            for mode in ('smoke', 'predict'):
                status('RUNNING_SOURCE_REPLAY' if mode == 'smoke' else 'RUNNING_SEVEN_VIEW_PREDICTION', gpu=gpu)
                command = [PYTHON, '-X', 'utf8', '-u', '-m', 'experiments.phase1_equivariant_dictionary.ground', mode, *common]
                with (controller/(mode+'.log')).open('x', encoding='utf8') as log:
                    result = subprocess.run(command, cwd=RELEASE, env=environment(gpu), stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
                if result.returncode:
                    raise RuntimeError('Ground '+mode+' exited '+str(result.returncode))
        manifest = json.loads((output/'flow/test_prediction/complete.json').read_text(encoding='utf8'))
        if (manifest.get('status') != 'PREDICTIONS_COMPLETE' or manifest.get('count') != 168000
                or manifest.get('commit') != args.commit or manifest.get('TX_truth_reads') != 0
                or manifest.get('model_updates') != 0):
            raise ValueError('All seven fixed Phase1 ground predictions are required')
        status('PREDICTIONS_COMPLETE_AWAITING_AUTO_AUDIT')
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
