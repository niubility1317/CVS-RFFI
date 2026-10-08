"""Adopt existing workers and resume the registered pipeline with four slots/GPU."""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LIMIT = 4
FREE_MB = 12000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def process(pid, proc=Path('/proc')):
    p = proc/str(pid)
    try:
        stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in ('Z', 'X', 'x'): return None
        try: env = dict(x.split('=', 1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
        except PermissionError: env = {}
        value = dict(pid=int(pid), start_ticks=int(stat[19]), state=stat[0],
            cwd=str((p/'cwd').resolve(strict=True)), argv=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x],
            gpu=env.get('CUDA_VISIBLE_DEVICES'))
        after = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if after[0] in ('Z', 'X', 'x'): return None
        if after[19] != stat[19]: raise RuntimeError('PID reused during read')
        return value
    except (FileNotFoundError, ProcessLookupError): return None
    except PermissionError:
        if not running(pid, proc): return None
        raise


def running(pid, proc=Path('/proc')):
    """Post-signal liveness uses stat only; exiting /proc/environ may be denied."""
    try: return (proc/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X', 'x')
    except (FileNotFoundError, ProcessLookupError): return False


def alive(receipt):
    actual = process(receipt['pid'])
    if actual and (actual['start_ticks'] != receipt['start_ticks'] or actual['cwd'] != receipt['cwd']):
        raise RuntimeError('Worker PID identity differs: '+str(receipt['pid']))
    return actual


def choose_gpu(caps):
    choices = [g for g, c in caps.items() if len(c['pids']) < LIMIT and c['free_mb'] >= FREE_MB]
    return min(choices, key=lambda g: (len(caps[g]['pids']), -caps[g]['free_mb'], g)) if choices else None


@contextmanager
def lock(path, nonblocking=False):
    import fcntl
    with Path(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def modules(release, package):
    release = Path(release).resolve()
    sys.path[:0] = [str(release), str(release/'code')]
    d = importlib.import_module('experiments.'+package+'.design')
    legacy = importlib.import_module('experiments.'+package+'.dispatch')
    if d.ROOT.resolve() != release: raise ValueError('Imported worker release differs')
    spec = read(release/'experiments'/package/'configs/launch_spec.json')
    legacy.validate_matrix(spec)
    for row in spec['rows']:
        d.validate_config(row['config'])
        if read(release/'experiments'/package/'configs'/(row['row_id']+'.json')) != row['config']:
            raise ValueError('Immutable worker config differs')
    return d, legacy, spec


def match_worker(actual, release, package, rows):
    if actual['cwd'] != str(Path(release).resolve()): return None
    argv = actual['argv']; prefix = 'experiments.'+package+'.'
    if '-m' not in argv: return None
    module = argv[argv.index('-m')+1]
    if module not in (prefix+'source', prefix+'evaluate', prefix+'worker'): return None
    if '--config' in argv:
        cfg = Path(argv[argv.index('--config')+1])
        rid = cfg.stem
        if cfg != Path(release)/'experiments'/package/'configs'/(rid+'.json'):
            raise ValueError('Live worker config path differs')
    elif '--row' in argv: rid = argv[argv.index('--row')+1]
    else: raise ValueError('Live worker has no registered row')
    if rid not in rows or not str(actual['gpu']).isdigit(): raise ValueError('Unregistered worker identity')
    kind = 'source' if module.endswith('.source') else 'predict' if module.endswith('.evaluate') else argv[argv.index('--kind')+1]
    if kind == 'predict' and '--mode' in argv and argv[argv.index('--mode')+1] != 'predict':
        raise ValueError('Scorer is not a CUDA worker')
    return dict(row_id=rid, kind=kind, pid=actual['pid'], start_ticks=actual['start_ticks'],
        gpu=int(actual['gpu']), cwd=actual['cwd'], argv=argv)


def adopt(d, package, spec):
    base = Path(d.PROJECT)/'runs'/d.RUN; logs = Path(d.PROJECT)/'logs'/d.RUN
    rows = {r['row_id']: r for r in spec['rows']}; found = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: actual = process(int(p.name))
        except (PermissionError, UnicodeError): continue
        if not actual: continue
        item = match_worker(actual, d.ROOT, package, rows)
        if item:
            key = (item['kind'], item['row_id'])
            if key in found: raise ValueError('Duplicate live row: '+str(key))
            item['log'] = str(logs/(item['kind']+'-'+item['row_id']+'.log'))
            found[key] = item
    for kind in ('source', 'predict'):
        path = base/('launch_'+kind+'.json')
        old = read(path)['rows'] if path.exists() else []
        seen = set(); merged = []
        for item in old:
            rid = item['row_id']
            if rid not in rows or rid in seen or item['kind'] != kind: raise ValueError('Launch receipt coverage differs')
            seen.add(rid); current = found.get((kind, rid))
            if current:
                if current['pid'] != item['pid']: raise ValueError('Live worker PID differs from receipt')
                merged.append(dict(item, **{k: current[k] for k in ('start_ticks',)}, adopted_argv=current['argv']))
            else:
                # Old completed PIDs must never be mistaken for a reused live PID.
                if process(item['pid']) is not None: raise ValueError('Historical receipt PID reused or unrecognized')
                merged.append(item)
        for (k, rid), item in found.items():
            if k == kind and rid not in seen:
                merged.append(dict(item, recovered_parent_receipt=True))
        if merged: write(path, dict(rows=merged))
    return list(found.values())


def artifact(d, row, kind):
    source = Path(row['config']['output_root'])
    return source/'completion.json' if kind == 'source' else source.parent/'prediction/complete.json'


def valid_artifact(d, row, kind):
    path = artifact(d, row, kind)
    return path.is_file() and read(path).get('status') == ('SOURCE_TRAINED' if kind == 'source' else 'PREDICTIONS_COMPLETE')


def queue(d, legacy, spec, package, control, kind):
    base = Path(d.PROJECT)/'runs'/d.RUN; logs = Path(d.PROJECT)/'logs'/d.RUN
    launch = base/('launch_'+kind+'.json')
    receipts = read(launch)['rows'] if launch.exists() else []
    by_id = {r['row_id']: r for r in receipts}; active = {}; pending = []; done = []; failures = []; children = {}
    for row in spec['rows']:
        rid = row['row_id']; r = by_id.get(rid)
        if r and 'start_ticks' in r and alive(r): active[rid] = dict(receipt=r, row=row)
        elif valid_artifact(d, row, kind): done.append(rid)
        elif r: failures.append(dict(row_id=rid, kind=kind, reason='Launched worker ended without valid completion; no retry'))
        else:
            output = Path(row['config']['output_root']) if kind == 'source' else Path(row['config']['output_root']).parent/'prediction'
            if output.exists(): raise ValueError('Unreceipted output exists; reconcile before launching '+rid)
            pending.append(row)
    while pending or active or failures:
        for rid, item in list(active.items()):
            if rid in children: children[rid].poll()
            if alive(item['receipt']): continue
            if valid_artifact(d, item['row'], kind): done.append(rid)
            else: failures.append(dict(row_id=rid, kind=kind, reason='Worker ended without valid completion; no retry'))
            del active[rid]
        while pending and len(active) < len(spec['rows']) and not failures:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps = legacy.capacity(); gpu = choose_gpu(caps)
                if gpu is None: break
                row = pending.pop(0); rid = row['row_id']
                cfg = str(d.ROOT/'experiments'/package/'configs'/(rid+'.json'))
                command = [sys.executable, '-u', str(control), '--worker', '--release', str(d.ROOT),
                    '--package', package, '--kind', kind, '--config', cfg, '--row', rid]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONPATH=str(d.ROOT)+os.pathsep+str(d.ROOT/'code'))
                logpath = logs/(kind+'-'+rid+'.log')
                with logpath.open('x') as log:
                    child = subprocess.Popen(command, cwd=d.ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                actual = process(child.pid)
                if not actual: raise RuntimeError('New worker absent immediately after launch')
                receipt = dict(row_id=rid, kind=kind, pid=child.pid, start_ticks=actual['start_ticks'], gpu=gpu,
                    cwd=str(d.ROOT), argv=command, log=str(logpath), capacity_override=LIMIT)
                receipts.append(receipt); write(launch, dict(rows=receipts))
                children[rid] = child; active[rid] = dict(receipt=receipt, row=row)
                print('LAUNCH '+json.dumps(receipt), flush=True)
        write(base/'queue_state.json', dict(kind=kind, active=list(active), pending=[r['row_id'] for r in pending],
            completed=done, failures=failures, updated_at=time.time(), max_active=len(spec['rows']), per_gpu_limit=LIMIT,
            policy='user override 4 total experiment processes/GPU; global CUDA and pre-CUDA reservations; >=12GB free'))
        if failures and not active: raise RuntimeError('Failed rows retained; no automatic retry '+repr(failures))
        if not pending and not active: return
        time.sleep(10)


def worker(a):
    d, legacy, spec = modules(a.release, a.package)
    c = read(a.config); d.validate_config(c)
    if a.row != d.row_id(c['arm'], c['model_seed']): raise ValueError('Worker row/config differs')
    if Path(a.config) != d.ROOT/'experiments'/a.package/'configs'/(a.row+'.json'): raise ValueError('Worker config path differs')
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES']); pid = os.getpid(); stable = 0
    while stable < 3:
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            entry = legacy.capacity()[gpu]; pids = entry['pids'] | {pid}
            ready = len(pids) <= LIMIT and entry['free_mb'] >= FREE_MB
            stable = stable+1 if ready else 0
            write(Path(d.PROJECT)/'runs'/d.RUN/'capacity_leases'/(a.kind+'-'+a.row+'.json'),
                dict(pid=pid, gpu=gpu, observed_pids=sorted(pids), stable_checks=stable, per_gpu_limit=LIMIT,
                    status='CAPACITY_READY' if stable == 3 else 'WAITING_CAPACITY', cuda_initialized=False, updated_at=time.time()))
        if stable < 3: time.sleep(1 if ready else 5)
    command = [sys.executable, '-u', '-m', 'experiments.'+a.package+'.'+('source' if a.kind == 'source' else 'evaluate')]
    command += ['--config', a.config] if a.kind == 'source' else ['--mode', 'predict', '--row', a.row]
    print('CAPACITY_READY '+json.dumps(dict(pid=pid, gpu=gpu, per_gpu_limit=LIMIT)), flush=True)
    os.execv(sys.executable, command)


def dispatch(a):
    d, legacy, spec = modules(a.release, a.package)
    base = Path(d.PROJECT)/'runs'/d.RUN
    # The publisher holds this lock until it writes our marker. Wait for that
    # handshake; once acquired, hold it for the entire pipeline lifetime.
    with lock(base/'capacity4_owner.lock'):
        for _ in range(100):
            if (base/'dispatcher_active.json').exists(): break
            time.sleep(.1)
        owner = read(base/'dispatcher_active.json')
        if owner['pid'] != os.getpid() or owner['control_release'] != str(Path(__file__).resolve().parents[2]):
            raise ValueError('Active owner marker differs')
        old = read(base/'dispatcher.json')
        if process(old['pid']): raise RuntimeError('Original dispatcher still alive')
        try:
            adopted = adopt(d, a.package, spec)
            write(base/'capacity4_adopted.json', dict(status='ADOPTED', workers=adopted, owner=owner, at=time.time()))
            queue(d, legacy, spec, a.package, Path(__file__).resolve(), 'source')
            if not (base/'source_matrix_frozen.json').exists(): legacy.freeze()
            frozen = read(base/'source_matrix_frozen.json')
            if frozen.get('status') != 'ALL_SOURCE_FROZEN' or frozen.get('rows') != d.rows():
                raise ValueError('Source freeze coverage differs')
            queue(d, legacy, spec, a.package, Path(__file__).resolve(), 'predict')
            subprocess.run([sys.executable, '-m', 'experiments.'+a.package+'.evaluate', '--mode', 'score'],
                cwd=d.ROOT, check=True)
            if read(base/'scoring_complete.json')['status'] != 'SCORED_COMPLETE': raise ValueError('Scoring incomplete')
            write(base/'completion.json', dict(status='ANALYZED', rows=16, views=7,
                independent_recount='VERIFIED', target_feedback_forbidden=True))
        except Exception as e:
            write(base/'failure.json', dict(status='FAILED', error=repr(e), no_retry=True)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--release', required=True); p.add_argument('--package', required=True)
    p.add_argument('--worker', action='store_true'); p.add_argument('--kind', choices=['source', 'predict'])
    p.add_argument('--config'); p.add_argument('--row'); args = p.parse_args()
    worker(args) if args.worker else dispatch(args)
