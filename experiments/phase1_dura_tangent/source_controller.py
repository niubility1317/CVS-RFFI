"""CPU-only source training, frozen prediction and truth-last source scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

from experiments.phase1_dura_posterior.resources import wait_for_cpu

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261103-phase1-dura-tangent-source-s2026092701-r01'
RELEASE = ROOT/'releases'/(RUN+'-release')
CONTROL = ROOT/'runs'/(RUN+'-controller')
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'


def status(state, **kw):
    tmp = CONTROL/'status.tmp'; tmp.write_text(json.dumps(dict(state=state, **kw)), encoding='utf8')
    os.replace(tmp, CONTROL/'status.json')


def environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONPATH=':'.join(map(str, (RELEASE,
                 ROOT/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',
                 ROOT/'releases/pamf_stage2_inference_20261021_r01',
                 ROOT/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'))))
    return env


def run(module, args, name):
    wait_for_cpu(lambda state, **extra: status(state, phase=name, **extra))
    status('RUNNING_'+name.upper())
    with (CONTROL/(name+'.log')).open('x', encoding='utf8') as log:
        result = subprocess.run([PYTHON, '-X', 'utf8', '-u', '-m', module, *args], cwd=RELEASE,
                                env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(name+' exited '+str(result.returncode))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--commit', required=True); a = p.parse_args()
    source = ROOT/'runs'/RUN; smoke = ROOT/'runs'/(RUN+'-training-smoke')
    score = ROOT/'runs'/(RUN+'-score')/'source_score.json'
    if (not CONTROL.is_dir() or any(CONTROL.iterdir()) or any(x.exists() for x in (source, smoke, score.parent))):
        raise ValueError('Unique immutable source experiment paths are required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != a.commit:
        raise ValueError('Registered release commit is required')
    try:
        common = ['--root', str(ROOT), '--commit', a.commit, '--device', 'cpu']
        run('experiments.phase1_dura_tangent.train_source', [*common, '--out', str(smoke), '--smoke'], 'smoke')
        first = json.loads((smoke/'smoke.json').read_text())
        expected = dict(status='SOURCE_TRAINING_SMOKE_PASS', fit_ids=64, eval_records=28,
                        F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=True, DURA_frozen=True,
                        TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0)
        if any(first.get(k) != v for k, v in expected.items()) or first.get('adaptive_gradient_batches', 0) < 1:
            raise ValueError('Actual source training smoke did not pass')
        run('experiments.phase1_dura_tangent.train_source', [*common, '--out', str(source)], 'train')
        fixed = json.loads((source/'prediction_complete.json').read_text())
        expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=a.commit, count=672, fit_ids=8192,
                        F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=True, DURA_frozen=True,
                        TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
        if any(fixed.get(k) != v for k, v in expected.items()) or not (source/'checkpoint.pt').is_file():
            raise ValueError('Frozen source checkpoint and predictions required')
        run('experiments.phase1_dura_tangent.score_source',
            ['--root', str(ROOT), '--source', str(source), '--out', str(score), '--commit', a.commit], 'score')
        result = json.loads(score.read_text())
        if result.get('status') != 'SOURCE_SCORE_COMPLETE': raise ValueError('Complete source score required')
        status('SOURCE_SCORE_COMPLETE', L_s_scored=result['L_s_scored'])
    except Exception as exc:
        status('FAILED', error=str(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
