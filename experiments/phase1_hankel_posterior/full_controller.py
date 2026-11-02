"""CPU full-source replay, fixed prediction, truth-last scoring and independent audit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261102-phase1-hankel-posterior-full-source-s2026092701-r01'
SMOKE = '20261102-phase1-hankel-posterior-full-smoke-s2026092701-r01'
SCORE = '20261102-phase1-hankel-posterior-full-score-s2026092701-r01'
RELEASE = ROOT/'releases'/(RUN+'-release')
CONTROL = ROOT/'runs'/(RUN+'-controller')
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'


def status(state, **kw):
    tmp = CONTROL/'status.tmp'
    tmp.write_text(json.dumps(dict(state=state, **kw), indent=2)+'\n', encoding='utf8')
    os.replace(tmp, CONTROL/'status.json')


def wait_capacity(phase):
    while os.getloadavg()[0] > os.cpu_count()-24:
        status('WAITING_CPU_CAPACITY', phase=phase, load=os.getloadavg()[0])
        time.sleep(60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    source, smoke, score = (ROOT/'runs'/name for name in (RUN, SMOKE, SCORE))
    if (not CONTROL.is_dir() or any(CONTROL.iterdir())
            or any(path.exists() for path in (source, smoke, score))):
        raise ValueError('Unique immutable source outputs are required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('Registered release commit is required')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONPATH=':'.join(map(str, (RELEASE, ROOT/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',
                                             ROOT/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'))))

    def run(module, options, name):
        wait_capacity(name)
        status('RUNNING_'+name.upper())
        with (CONTROL/(name+'.log')).open('x', encoding='utf8') as f:
            proc = subprocess.run([PYTHON, '-X', 'utf8', '-u', '-m', module, *options], cwd=RELEASE, env=env,
                                  stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(name+' exited '+str(proc.returncode))

    try:
        common = ['--root', str(ROOT), '--commit', args.commit]
        run('experiments.phase1_hankel_posterior.replay_full', [*common, '--out', str(smoke), '--smoke'], 'smoke')
        first = json.loads((smoke/'result.json').read_text())
        if first.get('status') != 'PASS' or first.get('records') != 64:
            raise ValueError('Actual source model smoke is required')
        run('experiments.phase1_hankel_posterior.replay_full', [*common, '--out', str(source)], 'predict')
        fixed = json.loads((source/'prediction_complete.json').read_text())
        if (fixed.get('status') != 'SOURCE_PREDICTIONS_FIXED' or fixed.get('commit') != args.commit
                or fixed.get('count') != 11761 or fixed.get('pilot_replayed') != 672 or any(fixed.get(k) != 0 for k in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads', 'model_updates'))):
            raise ValueError('Fixed source predictions are required')
        report = score/'source_score.json'
        run('experiments.phase1_hankel_posterior.score_full', [*common, '--source', str(source), '--out', str(report)], 'score')
        run('experiments.phase1_hankel_posterior.audit_full', ['--root', str(ROOT), '--source', str(source),
            '--score', str(report), '--out', str(score/'independent_audit.json')], 'audit')
        result = json.loads((score/'independent_audit.json').read_text())
        if result.get('status') != 'PASS' or result.get('records') != 11761 or result.get('fields_recomputed') != 736:
            raise ValueError('Independent source audit is required')
        status('SOURCE_SCORE_AND_AUDIT_COMPLETE', L_s_scored=result['L_s_scored'])
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__': main()
