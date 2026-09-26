"""One immutable launch; independent sequential GPU lanes; no automatic retry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    spec = json.loads(a.spec.read_text(encoding='utf-8'))
    runroot = Path(spec['execution']['remote_run_root'])
    logs = Path(spec['execution']['remote_log_root'])
    if a.detach:
        if runroot.exists() or logs.exists():
            raise FileExistsError('Run/log root exists; reconcile before launching')
        if Path.cwd().resolve() != ROOT or str(ROOT) != spec['code']['cwd']:
            raise ValueError('Wrong release checkout')
        guard = ROOT / 'launch_guard.json'
        with guard.open('x', encoding='utf-8') as f:
            json.dump(dict(owner=spec['execution']['launch_owner'], created=now(), commit=a.commit), f)
        with (ROOT / 'launcher.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--spec', str(a.spec.resolve()),
                                      '--commit', a.commit], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / 'detached.json').write_text(json.dumps(dict(pid=child.pid, commit=a.commit,
            spec=str(a.spec.resolve()), time=now())), encoding='utf-8')
        print(json.dumps(dict(pid=child.pid, status='DISPATCHED_NOT_YET_VERIFIED')))
        return
    runroot.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    launch = dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv, commit=a.commit, started=now(),
                  owner=spec['execution']['launch_owner'], python=sys.executable)
    (runroot / 'launch.json').write_text(json.dumps(launch, indent=2), encoding='utf-8')
    state = {r['row_id']: dict(status='QUEUED', gpu=r['gpu']) for r in spec['rows']}
    lock = threading.Lock()

    def update(row_id, **values):
        with lock:
            state[row_id].update(values, updated=now())
            temp = runroot / 'state.tmp'
            temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
            temp.replace(runroot / 'state.json')

    def lane(gpu):
        rows = [r for r in spec['rows'] if r['gpu'] == gpu]
        for row in rows:
            if Path(row['output_root']).exists():
                update(row['row_id'], status='TECHNICAL_FAILURE', error='output_exists')
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', COMPARISON_RELEASE_COMMIT=a.commit,
                       PYTHONPATH=str(ROOT / 'code')+os.pathsep+str(ROOT))
            started = time.time()
            with Path(row['log_path']).open('x', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, 'tools/run_practical_baseline.py', '--config', row['config_ref']],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                update(row['row_id'], status='RUNNING', pid=child.pid, started=now(), log=row['log_path'])
                code = child.wait()
            completion = Path(row['output_root']) / 'completion.json'
            ok = code == 0 and completion.is_file()
            update(row['row_id'], status='SOURCE_TRAINED' if ok else 'TECHNICAL_FAILURE',
                   exit_code=code, wall_seconds=time.time()-started)
            if not ok:
                return
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lane, sorted({r['gpu'] for r in spec['rows']})))
    summary = dict(finished=now(), counts={s:sum(r['status']==s for r in state.values())
                    for s in sorted({r['status'] for r in state.values()})})
    (runroot / 'dispatcher_complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
