"""Single-owner R42 CUDA takeover; preserve live predecessor workers and predictions."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import atomic, read, require, write
from . import milkr_r42_suite as suite


def merge_refs(prior, closed):
    def indexed(values):
        result = {}
        for item in values:
            key = (item['seed'], item['capsule'])
            require(key not in result, 'Duplicate prediction reference')
            result[key] = item
        return result
    result = indexed(prior)
    result.update(indexed(closed))
    return list(result.values())


def pending_tasks(tasks, finished, active, predecessor):
    keys = {t['key'] for t in tasks}
    require(len(keys) == len(tasks), 'Duplicate task key')
    require(not (finished & active or finished & predecessor or active & predecessor), 'Task ownership overlaps')
    require((finished | active | predecessor) <= keys, 'Unknown task key')
    return [t for t in tasks if t['key'] not in finished | active | predecessor]


def proc(pid):
    folder = Path('/proc') / str(pid)
    try:
        stat = (folder/'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z': return None
        return dict(pid=pid, state=stat[0], start=stat[19], uid=folder.stat().st_uid,
                    cwd=os.readlink(folder/'cwd'),
                    argv=(folder/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                    env=(folder/'environ').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def predecessor_workers(controller, old, tasks):
    children = (Path('/proc')/str(controller)/'task'/str(controller)/'children').read_text().split()
    found = {}
    for raw in children:
        info = proc(int(raw))
        if info is None: continue
        argv = info['argv']
        if '-m' not in argv or suite.__package__+'.milkr_r42_suite' not in argv:
            require(Path(argv[0]).name == 'nvidia-smi', 'Unexpected live predecessor child')
            continue
        require('worker' in argv and '--key' in argv and '--runtime' in argv, 'Unexpected predecessor phase')
        key = argv[argv.index('--key')+1]
        require(key in tasks and key not in found and argv[argv.index('--runtime')+1] == old['runtime_file'], 'Wrong predecessor worker')
        require(info['uid'] == os.getuid() and info['cwd'] == old['release_root'], 'Wrong predecessor worker owner/CWD')
        gpu = [x.split('=', 1)[1] for x in info['env'] if x.startswith('CUDA_VISIBLE_DEVICES=')]
        require(len(gpu) == 1 and gpu[0].isdigit() and int(gpu[0]) in old['allowed_gpus'], 'Wrong predecessor GPU')
        found[key] = dict(pid=info['pid'], start=info['start'], gpu=int(gpu[0]))
    return found


def alive(record):
    value = proc(record['pid'])
    return value is not None and value['start'] == record['start']


def closed_ref(old, task):
    root = Path(old['run_root'])/'tasks'/task['key']
    folder = root/'predictions'/task['capsule'].split('/')[-1]
    if not (folder/'complete.json').is_file(): return None
    suite.prediction_closed(old, task)
    metrics = root/'scores'/task['capsule'].split('/')[-1]/'metrics.json'
    return dict(seed=task['seed'], capsule=task['capsule'], prediction=str(folder),
                metrics=str(metrics) if metrics.is_file() else None)


def source_smoke(rt, seed, output):
    import numpy as np
    import torch
    from .features import load_reader, source_smoke as verify_source
    from .io import activate
    from .milkr_screen import extract
    from .milkr.d92_branch_local_ridge import fit_branch_local_ridge
    from .milkr.cuda_local_ridge import CudaLocalRidge
    local = dict(rt, model_seeds=[seed], **rt['source_reuse_by_seed'][str(seed)])
    activate(local)
    verify_source(local, Path(output).with_name(f'source-{seed}.json'))
    from experiments.cric_dura_comparison.f_data import SourceDataset
    original = SourceDataset(local['source_cache'], 'L_s', read(local['source_contract'])).x[:2]
    require(len(original) == 2, 'Not enough legal source examples')
    samples = torch.cat([torch.roll(original[i % 2:i % 2+1], 3*i, dims=-1) for i in range(6)])
    raw = np.asarray(samples.tolist(), dtype=np.float32)
    reader = load_reader(local, seed, 'cuda:0')
    features, _ = extract(reader, raw, 'cuda:0', batch=6)
    state = fit_branch_local_ridge(**features, support_labels=np.arange(6),
        support_ids=[f'source_smoke_{i}' for i in range(6)], classes=[f'c{i}' for i in range(6)])
    cpu = state.score(**features); scorer = CudaLocalRidge(state); gpu = scorer.score(**features)
    require(np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), 'CUDA scores differ on legal source')
    require(np.array_equal(cpu.argmax(1), gpu.argmax(1)), 'CUDA predictions differ on legal source')
    write(output, dict(status='PASS', actual_checkpoint=True, seed=seed, source_examples=2,
        target_rows=0, query_rows=0, truth_read=False, max_score_difference=float(np.max(np.abs(cpu-gpu))),
        cpu_tie_rows=scorer.cpu_tie_rows, device=torch.cuda.get_device_name(0)))


def transfer_owner(rt, old, pid, task_by_key, output):
    info = proc(pid)
    require(info is not None and info['uid'] == os.getuid() and info['cwd'] == old['release_root'], 'Predecessor controller identity differs')
    require(suite.__package__+'.milkr_r42_suite' in info['argv'] and 'dispatch' in info['argv']
            and old['runtime_file'] in info['argv'], 'Wrong predecessor controller command')
    os.kill(pid, signal.SIGSTOP)
    terminated = False
    try:
        for _ in range(100):
            paused = proc(pid)
            if paused is not None and paused['start'] == info['start'] and paused['state'] in ('T','t'): break
            time.sleep(.05)
        else: raise RuntimeError('Could not pause predecessor controller')
        workers = predecessor_workers(pid, old, task_by_key)
        original_workers = dict(workers)
        write(output, dict(predecessor_controller=pid, predecessor_start=info['start'],
                           inherited_workers=workers, status='PAUSED_BEFORE_TRANSFER'))
        # Wait for a real predecessor worker to finish so the smoke does not
        # exceed the authorized two R42 GPU processes per device.
        while True:
            for key, record in list(workers.items()):
                if not alive(record):
                    require(closed_ref(old, task_by_key[key]) is not None, 'Predecessor worker failed while paused')
                    del workers[key]
            available = [gpu for gpu in rt['allowed_gpus'] if sum(w['gpu'] == gpu for w in workers.values()) < 2]
            available = [gpu for gpu in available if int(subprocess.check_output(
                ['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) >= rt['minimum_free_MiB']]
            if available: break
            require(alive(info), 'Predecessor controller disappeared before smoke')
            time.sleep(rt['poll_seconds'])
        smoke_dir = output.parent/'smoke'; smoke_dir.mkdir()
        for seed in rt['model_seeds']:
            destination = smoke_dir/f'cuda-{seed}.json'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(available[0]), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            cmd = [sys.executable,'-u','-m',__package__+'.r42_fast_takeover','--runtime',rt['runtime_file'],
                   '--smoke-seed',str(seed),'--smoke-output',str(destination)]
            subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,check=True)
            smoke = read(destination)
            require(smoke['status']=='PASS' and smoke['seed']==seed and smoke['query_rows']==0, 'Source smoke failed')
        workers = predecessor_workers(pid, old, task_by_key)
        require(set(workers) <= set(original_workers), 'Unexpected new worker while predecessor was paused')
        for key, original in original_workers.items():
            if key in workers:
                require(workers[key] == original, 'Predecessor worker identity changed while paused')
            else:
                require(closed_ref(old, task_by_key[key]) is not None, 'Predecessor worker failed during smoke')
        require(proc(pid)['start'] == info['start'], 'Predecessor controller changed')
        os.kill(pid, signal.SIGKILL)
        terminated = True
        for _ in range(100):
            if proc(pid) is None: break
            time.sleep(.05)
        else: raise RuntimeError('Predecessor controller did not exit')
        atomic(output, dict(predecessor_controller=pid, predecessor_start=info['start'],
                            inherited_workers=workers, status='TRANSFERRED'))
        return workers
    finally:
        if not terminated and alive(info): os.kill(pid, signal.SIGCONT)


def dispatch(rt):
    old = read(rt['predecessor_runtime_file'])
    tasks = suite.tasks(rt); by_key = {item['key']: item for item in tasks}
    require(suite.tasks(old) == tasks and rt['local_ridge_cuda'] is True, 'Takeover matrix/backend differs')
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'launches'): (root/name).mkdir()
    write(root/'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv, role='cuda_takeover'))
    for seed in rt['model_seeds']:
        src = suite.source_root(rt, seed); src.mkdir(parents=True)
        parent = suite.source_root(old, seed)
        done = read(parent/'suite_source_complete.json')
        require(done['model_seed'] == seed and done['source_new_optimizer_updates'] == 0, 'Wrong frozen source')
        for name in ('source_orbit_cca.npz','source_selection.json','suite_source_complete.json'):
            os.link(parent/name, src/name)
    predecessor = transfer_owner(rt, old, rt['predecessor_controller_pid'], by_key, root/'takeover.json')
    oldrefs = read(old['prediction_reuse_refs_file']) if old.get('prediction_reuse_refs_file') else []
    inherited = {}; active = {}; finished = set(); failed = []
    for task in tasks:
        if task['key'] not in predecessor:
            ref = closed_ref(old, task)
            if ref: inherited[task['key']] = ref
    refsfile = root/'prediction_reuse_refs.json'
    atomic(refsfile, merge_refs(oldrefs, list(inherited.values())))
    require(rt['prediction_reuse_refs_file'] == str(refsfile), 'Wrong takeover references file')
    while len(finished) < len(tasks):
        for key, record in list(predecessor.items()):
            if alive(record): continue
            ref = closed_ref(old, by_key[key])
            if ref is None: failed.append(dict(key=key, reason='predecessor worker exited without complete prediction'))
            else: inherited[key] = ref
            del predecessor[key]
        atomic(refsfile, merge_refs(oldrefs, list(inherited.values())))
        for key, item in list(active.items()):
            status = item['process'].poll()
            if status is None: continue
            item['log'].close(); del active[key]
            if status: failed.append(dict(key=key, exit_code=status))
            else: suite.prediction_closed(rt, by_key[key]); finished.add(key)
        atomic(root/'worker_state.json', dict(status='FAILED' if failed else 'RUNNING', completed=len(finished),
            total=len(tasks), failed=failed, pending=len(tasks)-len(finished)-len(active),
            active=[dict(key=k, pid=v['process'].pid, gpu=v['gpu']) for k,v in active.items()],
            predecessor_active=[dict(key=k, **v) for k,v in predecessor.items()], inherited_closed=len(inherited)))
        if failed:
            if not active: raise RuntimeError('Takeover worker failure; predecessor outputs preserved: '+str(failed))
        else:
            pending = pending_tasks(tasks, finished, set(active), set(predecessor))
            for gpu in rt['allowed_gpus']:
                if not pending: break
                used = sum(v['gpu'] == gpu for v in active.values()) + sum(v['gpu'] == gpu for v in predecessor.values())
                if used >= 2: continue
                free = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free < rt['minimum_free_MiB']: continue
                item = pending.pop(0); key = item['key']
                cmd = [sys.executable,'-u','-m',suite.__package__+'.milkr_r42_suite','worker','--runtime',rt['runtime_file'],'--key',key]
                log = (root/'logs'/('worker-'+key+'.log')).open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[key] = dict(process=process, gpu=gpu, log=log)
                write(root/'launches'/('worker-'+key+'.json'),dict(pid=process.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
        if len(finished) < len(tasks): time.sleep(rt['poll_seconds'])
    require(not predecessor and not active and not failed, 'Takeover predictions not closed')
    for task in tasks: suite.prediction_closed(rt,task)
    rows = sum(len(suite.desired_rows(t['points'])) for t in tasks)
    require(rows == 167496, 'Wrong total prediction rows')
    write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED',tasks=len(tasks),rows=rows,truth_read=False,query_fit=False))
    for i, task in enumerate(tasks):
        cmd=[sys.executable,'-u','-m',suite.__package__+'.milkr_r42_suite','score','--runtime',rt['runtime_file'],'--key',task['key']]
        with (root/'logs'/('score-'+task['key']+'.log')).open('x') as log:
            subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        done=read(Path(suite.task_runtime(rt,task)['run_root'])/'scores'/task['capsule'].split('/')[-1]/'complete.json')
        require(done['status']=='SCORED_COMPLETE','Score did not close')
        atomic(root/'score_state.json',dict(status='RUNNING',completed=i+1,total=len(tasks)))
    write(root/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(tasks),rows=rows,truth_last=True,target_feedback=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True)
    parser.add_argument('--smoke-seed',type=int);parser.add_argument('--smoke-output')
    args=parser.parse_args();runtime=read(args.runtime)
    if args.smoke_seed is not None:
        require(args.smoke_output is not None, 'Smoke output required')
        source_smoke(runtime,args.smoke_seed,args.smoke_output)
    else:
        dispatch(runtime)
