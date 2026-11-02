"""CPU source diagnostic, followed by pseudo-target scoring after fixation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261102-phase1-continuous-fusion-source-s2026092701-r02'
RELEASE = ROOT/'releases'/(RUN+'-release')
CONTROL = ROOT/'runs'/(RUN+'-controller')
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'


def status(state, **kw):
    tmp = CONTROL/'status.tmp'; tmp.write_text(json.dumps(dict(state=state, **kw)), encoding='utf8')
    os.replace(tmp, CONTROL/'status.json')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--commit', required=True); a = p.parse_args()
    source = ROOT/'runs'/RUN; smoke = ROOT/'runs'/(RUN+'-smoke'); score = ROOT/'runs'/(RUN+'-score')/'score.json'
    if (not CONTROL.is_dir() or any(CONTROL.iterdir()) or any(x.exists() for x in (source, smoke, score.parent))):
        raise ValueError('Unique immutable experiment paths required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != a.commit:
        raise ValueError('Registered release commit required')
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTHONPATH=':'.join(map(str, (RELEASE,
        ROOT/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',
        ROOT/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'))))
    def run(mode, path, name, extra=()):
        while os.getloadavg()[0] > 72:
            status('WAITING_CPU_CAPACITY', phase=name, load=os.getloadavg()[0]); time.sleep(60)
        status('RUNNING_'+name.upper())
        command = [PYTHON, '-X', 'utf8', '-u', '-m', 'experiments.phase1_source_continuous_fusion.diagnostic',
                   mode, '--root', str(ROOT), '--out', str(path), '--commit', a.commit, *extra]
        with (CONTROL/(name+'.log')).open('x', encoding='utf8') as log:
            result = subprocess.run(command, cwd=RELEASE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(name+' exited '+str(result.returncode))
    try:
        run('predict', smoke, 'smoke', ('--smoke',))
        if json.loads((smoke/'prediction_complete.json').read_text()).get('count') != 28:
            raise ValueError('28 actual source smoke records required')
        run('predict', source, 'predict')
        run('score', score, 'score', ('--source', str(source)))
        result = json.loads(score.read_text())
        if result.get('status') != 'SOURCE_PSEUDO_SCORE_COMPLETE' or result.get('count') != 448:
            raise ValueError('Complete source pseudo score required')
        status('SOURCE_DIAGNOSTIC_COMPLETE', score=str(score))
    except Exception as exc:
        status('FAILED', error=str(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
