"""One-owner background chain for the fixed physical-line ground evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def cpu_environment():
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    control = args.out.parent/(args.out.name+'-controller')
    if args.out.exists() or control.exists():
        raise FileExistsError('Unique controller and ground output roots required')
    control.mkdir(parents=True)
    def status(state, **extra):
        (control/'status.json').write_text(json.dumps(dict(state=state, **extra))+'\n', encoding='utf8')
    try:
        for stage in ('smoke', 'predict', 'score'):
            status('RUNNING_'+stage.upper())
            command = [sys.executable, '-X', 'utf8', '-u', '-m',
                       'experiments.phase1_dura_orthogonal.physical_line_ground', stage,
                       '--root', str(args.root), '--out', str(args.out), '--commit', args.commit]
            with (control/(stage+'.log')).open('x', encoding='utf8') as log:
                subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True, env=cpu_environment())
        summary = json.loads((args.out/'flow/test_summary.json').read_text(encoding='utf8'))
        paired = json.loads((args.out/'flow/paired_dura.json').read_text(encoding='utf8'))
        if summary.get('status') != 'ANALYZED' or not summary.get('all_predictions_before_truth') or paired.get('status') != 'PAIRED_COMPLETE':
            raise ValueError('Complete truth-last paired scoring required')
        status('GROUND_SCORE_COMPLETE')
    except Exception as exc:
        status('FAILED', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
