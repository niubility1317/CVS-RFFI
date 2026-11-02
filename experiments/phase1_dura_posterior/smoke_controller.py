"""Wait for CPU capacity, then run one real-source physical DURA smoke."""

import json
import os
from pathlib import Path
import subprocess
import time
import traceback


ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261102-phase1-dura-posterior-internal-smoke-s2026092701-r01'
RELEASE = ROOT/'releases'/(RUN+'-release')
OUTPUT = ROOT/'runs'/RUN


def status(state, **extra):
    temporary = OUTPUT/'status.tmp'
    temporary.write_text(json.dumps(dict(state=state, **extra), indent=2)+'\n', encoding='utf8')
    os.replace(temporary, OUTPUT/'status.json')


def main():
    if not OUTPUT.is_dir() or any(OUTPUT.iterdir()):
        raise ValueError('One empty, immutable run root is required')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    releases = ROOT/'releases'
    env['PYTHONPATH'] = ':'.join(map(str, (RELEASE,
        releases/'20261101-phase1-physical-rich-router-s2026092701-r01-release',
        releases/'20261027-phase1-pamf-repl-shuffled-views-m16-r02-release')))
    try:
        while os.getloadavg()[0] > os.cpu_count()-24:
            status('WAITING_CPU_CAPACITY', load=os.getloadavg()[0])
            time.sleep(60)
        status('RUNNING_REAL_SOURCE_SMOKE')
        args = ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-X', 'utf8', '-u', '-m',
                'experiments.phase1_dura_posterior.physical_source_smoke', '--root', str(ROOT)]
        with (OUTPUT/'smoke.log').open('x', encoding='utf8') as log:
            result = subprocess.run(args, cwd=RELEASE, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Real source smoke exited '+str(result.returncode))
        lines = (OUTPUT/'smoke.log').read_text(encoding='utf8').splitlines()
        payload = json.loads(lines[-1])
        if (payload.get('status') != 'REAL_PHYSICAL_DURA_INSERTION_PASS'
                or payload.get('active') != 2 or payload.get('closed') != 2
                or any(payload.get(k) != 0 for k in ('optimizer_updates', 'TX_labels_read',
                                                       'U_s_truth_reads', 'V_reads', 'test_reads'))):
            raise ValueError('Real-source smoke invariants did not pass')
        status('REAL_SOURCE_SMOKE_COMPLETE', result=payload)
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
