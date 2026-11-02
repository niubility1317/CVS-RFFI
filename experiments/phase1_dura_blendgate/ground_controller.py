"""CPU-only source replay and seven-view ground prediction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.phase1_dura_posterior.resources import wait_for_cpu
from experiments.phase1_dura_blendgate.source_controller import ROOT, RUN, RELEASE, PYTHON, environment

GROUND = '20261103-phase1-dura-blendgate-ground-test-s2026092701-r01'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--commit', required=True); p.add_argument('--source-pid', type=int, required=True)
    a = p.parse_args(); controller = ROOT/'runs'/(GROUND+'-controller'); output = ROOT/'runs'/GROUND
    source = ROOT/'runs'/RUN; source_status = ROOT/'runs'/(RUN+'-controller')/'status.json'
    if not controller.is_dir() or any(controller.iterdir()) or output.exists(): raise ValueError('Unique ground paths required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != a.commit: raise ValueError('Registered release required')
    def status(state, **kw):
        tmp = controller/'status.tmp'; tmp.write_text(json.dumps(dict(state=state, **kw)), encoding='utf8'); os.replace(tmp, controller/'status.json')
    try:
        while not (source/'prediction_complete.json').exists():
            if source_status.exists() and json.loads(source_status.read_text()).get('state') == 'FAILED':
                raise RuntimeError('Source training failed technically')
            process = Path('/proc')/str(a.source_pid)/'cmdline'
            if (not process.is_file() or 'phase1_dura_blendgate.source_controller' not in process.read_bytes().decode(errors='replace')) and not (source/'prediction_complete.json').exists():
                raise RuntimeError('Source controller stopped before fixed predictions')
            status('WAITING_FIXED_SOURCE', source_pid=a.source_pid); time.sleep(60)
        for mode in ('smoke', 'predict'):
            wait_for_cpu(lambda state, **extra: status(state, phase=mode, **extra))
            status('RUNNING_SOURCE_REPLAY' if mode == 'smoke' else 'RUNNING_SEVEN_VIEW_PREDICTION')
            command = [PYTHON, '-X', 'utf8', '-u', '-m', 'experiments.phase1_dura_blendgate.ground', mode,
                       '--root', str(ROOT), '--out', str(output), '--commit', a.commit, '--device', 'cpu']
            with (controller/(mode+'.log')).open('x', encoding='utf8') as log:
                result = subprocess.run(command, cwd=RELEASE, env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Ground '+mode+' exited '+str(result.returncode))
        fixed = json.loads((output/'flow/test_prediction/complete.json').read_text())
        if (fixed.get('status') != 'PREDICTIONS_COMPLETE' or fixed.get('count') != 168000
                or fixed.get('commit') != a.commit or fixed.get('TX_truth_reads') != 0 or fixed.get('model_updates') != 0):
            raise ValueError('All seven fixed Phase1 ground predictions required')
        status('PREDICTIONS_COMPLETE_AWAITING_AUTO_AUDIT')
    except Exception as exc:
        status('FAILED', error=str(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
