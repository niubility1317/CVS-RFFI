"""Wait for one fixed source model, then smoke, predict seven views and score."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = '20261107-phase1-dura-antithetic-flow-source-s2026092701-r01'
SOURCE_COMMIT = '0a8a4ad1c326fda05b7f49eeb3b9b6b310051a77'
MODE = 'antithetic_probability_mean'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    controller = args.out.with_name(args.out.name+'-controller')
    parent_status = args.root/'runs'/(SOURCE+'-controller')/'status.json'
    if args.out.exists() or controller.exists() or not parent_status.is_file():
        raise ValueError('New output roots and existing registered source controller required')
    controller.mkdir(parents=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def status(state, **extra):
        temporary = controller/'status.tmp'
        temporary.write_text(json.dumps(dict(state=state, source=SOURCE, source_commit=SOURCE_COMMIT,
            commit=args.commit, inference_mode=MODE, **extra))+'\n', encoding='utf8')
        os.replace(temporary, controller/'status.json')
    def execute(mode):
        status('RUNNING_'+mode.upper())
        command = [sys.executable, '-X', 'utf8', '-u', '-m', 'experiments.phase1_dura_orthogonal.ground', mode,
            '--root', str(args.root), '--out', str(args.out), '--source', SOURCE, '--source-commit', SOURCE_COMMIT,
            '--commit', args.commit, '--inference-mode', MODE]
        with (controller/(mode+'.log')).open('x', encoding='utf8') as log:
            subprocess.run(command, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        status('WAITING_FOR_FIXED_SOURCE')
        while not (args.root/'runs'/SOURCE/'source_summary.json').is_file():
            parent = json.loads(parent_status.read_text(encoding='utf8'))
            if parent.get('state') in ('FAILED', 'GROUND_SCORE_COMPLETE'):
                raise RuntimeError('Source controller ended without the fixed source model')
            time.sleep(30)
        execute('smoke')
        execute('predict')
        from experiments.pamf_phase1_flow import experiment as core
        done = core.read(args.out/'flow/test_prediction/complete.json')
        core.validate_prediction_manifest(done, count=168000, commit=args.commit)
        inference = core.read(args.out/'flow/inference.json')
        if (inference.get('source') != SOURCE or inference.get('source_commit') != SOURCE_COMMIT
                or inference.get('inference_mode') != MODE or inference.get('commit') != args.commit):
            raise ValueError('All predictions must use the registered fixed source and two-path readout')
        execute('score')
        summary = core.read(args.out/'flow/test_summary.json')
        paired = core.read(args.out/'flow/paired_dura.json')
        if (summary.get('status') != 'ANALYZED' or not summary.get('all_predictions_before_truth')
                or summary.get('target_feedback') is not False or paired.get('status') != 'PAIRED_COMPLETE'):
            raise ValueError('Independent completed seven-view score required')
        status('GROUND_SCORE_COMPLETE')
    except Exception as error:
        status('FAILED', error=str(error))
        raise


if __name__ == '__main__':
    main()
