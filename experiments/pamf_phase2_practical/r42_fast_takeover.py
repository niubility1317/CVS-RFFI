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


def gpu_worker_limit(rt, gpu):
    limits = rt.get('max_workers_by_gpu')
    if limits is None:
        return 2
    require(set(limits) == {str(g) for g in rt['allowed_gpus']}, 'GPU concurrency map differs from allowed GPUs')
    limit = limits[str(gpu)]
    require(type(limit) is int and 1 <= limit <= 3, 'Invalid GPU concurrency limit')
    return limit


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
    except PermissionError:
        # A same-user process can lose /proc/environ access as it exits.
        # Only treat it as gone when its original stat identity has ended.
        if 'stat' in locals() and not process_running(pid,stat[19]):return None
        raise


def running_stat_fields(stat,start):
    # Linux PF_EXITING is set before /proc/<pid>/environ becomes unreadable.
    return stat[0] not in ('Z','X','x') and not (int(stat[6]) & 4) and (start is None or stat[19]==start)


def process_running(pid,start):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
        return running_stat_fields(stat,start)
    except (FileNotFoundError,ProcessLookupError):
        return False


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
    return process_running(record['pid'],record['start'])


def controller_command_matches(argv, runtime_file):
    if argv.count('-m') != 1 or argv.count('--runtime') != 1:
        return False
    if argv[argv.index('--runtime')+1] != runtime_file or '--smoke-seed' in argv:
        return False
    module = argv[argv.index('-m')+1]
    return (module == suite.__package__+'.milkr_r42_suite' and 'dispatch' in argv
            or module == suite.__package__+'.r42_fast_takeover' and 'dispatch' not in argv)


def validate_predecessor_state(state):
    require(state.get('status') == 'RUNNING', 'Predecessor is not in prediction phase')
    require(not state.get('failed'), 'Predecessor has failed tasks')
    require(not state.get('predecessor_active'), 'Predecessor still has inherited workers')


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
    from .r42_suite_worker import orbit_features, _restore
    from .features import load_G
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
    if rt.get('fast_dlpack_transfer',False):
        alternate, _ = extract(reader, raw, 'cuda:0', batch=6, fast_transfer=True)
        require(all(np.array_equal(features[k],alternate[k]) for k in features), 'Fast baseline features differ')
        original_orbit, original_logits = orbit_features(reader,raw,'cuda:0',6,False)
        fast_orbit, fast_logits = orbit_features(reader,raw,'cuda:0',6,True)
        require(all(np.array_equal(original_orbit[k],fast_orbit[k]) for k in original_orbit), 'Fast orbit features differ')
        require(np.array_equal(original_logits,fast_logits), 'Fast orbit logits differ')
        engine=load_G(local,'cuda:0')
        require(np.array_equal(_restore(engine,raw,'cuda:0',6),_restore(engine,raw,'cuda:0',6,fast_transfer=True)), 'Fast DURA output differs')
    state = fit_branch_local_ridge(**features, support_labels=np.arange(6),
        support_ids=[f'source_smoke_{i}' for i in range(6)], classes=[f'c{i}' for i in range(6)])
    cpu = state.score(**features)
    scorer = CudaLocalRidge(state,pair_budget=rt.get('local_ridge_pair_budget',2048),
                            fast_transfer=rt.get('fast_dlpack_transfer',False))
    gpu = scorer.score(**features)
    require(np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), 'CUDA scores differ on legal source')
    require(np.array_equal(cpu.argmax(1), gpu.argmax(1)), 'CUDA predictions differ on legal source')
    write(output, dict(status='PASS', actual_checkpoint=True, seed=seed, source_examples=2,
        target_rows=0, query_rows=0, truth_read=False, max_score_difference=float(np.max(np.abs(cpu-gpu))),
        cpu_tie_rows=scorer.cpu_tie_rows, device=torch.cuda.get_device_name(0)))


def transfer_owner(rt, old, pid, task_by_key, output):
    info = proc(pid)
    require(info is not None and info['uid'] == os.getuid() and info['cwd'] == old['release_root'], 'Predecessor controller identity differs')
    require(controller_command_matches(info['argv'],old['runtime_file']), 'Wrong predecessor controller command')
    os.kill(pid, signal.SIGSTOP)
    terminated = False
    try:
        for _ in range(100):
            paused = proc(pid)
            if paused is not None and paused['start'] == info['start'] and paused['state'] in ('T','t'): break
            time.sleep(.05)
        else: raise RuntimeError('Could not pause predecessor controller')
        if suite.__package__+'.r42_fast_takeover' in info['argv']:
            validate_predecessor_state(read(Path(old['run_root'])/'worker_state.json'))
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
            if not process_running(pid,info['start']): break
            time.sleep(.05)
        else: raise RuntimeError('Predecessor controller did not exit')
        atomic(output, dict(predecessor_controller=pid, predecessor_start=info['start'],
                            inherited_workers=workers, status='TRANSFERRED'))
        return workers
    finally:
        if not terminated and alive(info) and not rt.get('keep_predecessor_paused_on_failure', False): os.kill(pid, signal.SIGCONT)


def adopt_orphans(rt, old, task_by_key):
    transfer=read(rt['failed_takeover_file'])
    require(transfer['status']=='PAUSED_BEFORE_TRANSFER' and transfer['predecessor_controller']==rt['predecessor_controller_pid'],
            'Wrong failed takeover record')
    require(not process_running(rt['predecessor_controller_pid'],transfer['predecessor_start']), 'Predecessor controller still active')
    require(not process_running(rt['failed_controller_pid'],None), 'Failed takeover controller still active')
    state=read(Path(old['run_root'])/'worker_state.json')
    validate_predecessor_state(state)
    recorded=transfer['inherited_workers'];require(set(recorded)<=set(task_by_key), 'Unknown orphan worker')
    candidates=dict(recorded)
    for item in state.get('active',[]):
        key=item['key'];require(key in task_by_key,'Unknown worker in predecessor state')
        if key in candidates:
            require(candidates[key]['pid']==item['pid'] and candidates[key]['gpu']==item['gpu'], 'Worker snapshots disagree')
        else:
            info=proc(item['pid'])
            if info is None:
                require(closed_ref(old,task_by_key[key]) is not None,'Predecessor worker exited without complete prediction')
                continue
            candidates[key]=dict(pid=item['pid'],start=info['start'],gpu=item['gpu'])
    inherited={}
    for key,record in candidates.items():
        info=proc(record['pid'])
        if info is None or info['start']!=record['start']:
            require(closed_ref(old,task_by_key[key]) is not None,'Predecessor worker exited without complete prediction')
            continue
        argv=info['argv']
        require(info['uid']==os.getuid() and info['cwd']==old['release_root'], 'Wrong orphan owner/CWD')
        require(suite.__package__+'.milkr_r42_suite' in argv and 'worker' in argv and '--key' in argv
                and argv[argv.index('--key')+1]==key and old['runtime_file'] in argv, 'Wrong orphan worker command')
        gpu=[x.split('=',1)[1] for x in info['env'] if x.startswith('CUDA_VISIBLE_DEVICES=')]
        require(len(gpu)==1 and gpu[0].isdigit() and int(gpu[0])==record['gpu'] and record['gpu'] in old['allowed_gpus'], 'Wrong orphan GPU')
        inherited[key]=record
    return inherited


def smoke_adopted(rt,old,workers,task_by_key,output):
    while True:
        for key,record in list(workers.items()):
            if not alive(record):
                require(closed_ref(old,task_by_key[key]) is not None,'Predecessor worker exited without complete prediction')
                del workers[key]
        available=[gpu for gpu in rt['allowed_gpus'] if sum(w['gpu']==gpu for w in workers.values())<2]
        available=[gpu for gpu in available if int(subprocess.check_output(
            ['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=rt['minimum_free_MiB']]
        if available:break
        time.sleep(rt['poll_seconds'])
    smoke_dir=output.parent/'smoke';smoke_dir.mkdir()
    for seed in rt['model_seeds']:
        destination=smoke_dir/f'cuda-{seed}.json'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(available[0]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,'-u','-m',__package__+'.r42_fast_takeover','--runtime',rt['runtime_file'],
             '--smoke-seed',str(seed),'--smoke-output',str(destination)]
        subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,check=True)
        smoke=read(destination)
        require(smoke['status']=='PASS' and smoke['seed']==seed and smoke['query_rows']==0,'Source smoke failed')
    for key,record in list(workers.items()):
        if not alive(record):
            require(closed_ref(old,task_by_key[key]) is not None,'Predecessor worker exited without complete prediction')
            del workers[key]
    write(output,dict(status='TRANSFERRED',mode='RECOVERED_ORPHANS',inherited_workers=workers,
                      predecessor_controller=rt['predecessor_controller_pid'],failed_controller=rt['failed_controller_pid']))
    return workers


def score_queue(rt, tasks, limit):
    root=Path(rt['run_root'])
    require(read(root/'predictions_complete.json')['status']=='ALL_PREDICTIONS_FIXED', 'Global prediction closure required')
    require(type(limit) is int and 1<=limit<=4, 'CPU scoring concurrency must be between 1 and 4')
    pending=list(tasks);active={};completed=0;failed=[]
    while pending or active:
        for key,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();del active[key]
            if code:
                failed.append(dict(key=key,exit_code=code))
                continue
            task=item['task'];rx=task['capsule'].split('/')[-1]
            try:
                done=read(Path(suite.task_runtime(rt,task)['run_root'])/'scores'/rx/'complete.json')
                require(done['status']=='SCORED_COMPLETE','Score did not close')
            except (OSError,ValueError,KeyError) as exc:
                failed.append(dict(key=key,error=str(exc)))
                continue
            completed+=1
        atomic(root/'score_state.json',dict(status='FAILED' if failed else 'RUNNING',completed=completed,
            total=len(tasks),pending=len(pending),active=[dict(key=k,pid=v['process'].pid) for k,v in active.items()],failed=failed))
        if failed:
            pending.clear()
        while pending and len(active)<limit and not failed:
            task=pending.pop(0);key=task['key']
            cmd=[sys.executable,'-u','-m',suite.__package__+'.milkr_r42_suite','score','--runtime',rt['runtime_file'],'--key',key]
            log=(root/'logs'/('score-'+key+'.log')).open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            try:process=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            except OSError as exc:
                log.close();failed.append(dict(key=key,error=str(exc)));break
            active[key]=dict(task=task,process=process,log=log)
        if active:time.sleep(rt['poll_seconds'])
    require(not failed and completed==len(tasks), 'Independent scoring failed; outputs retained: '+str(failed))
    atomic(root/'score_state.json',dict(status='COMPLETE',completed=completed,total=len(tasks),pending=0,active=[],failed=[]))


def selected_tasks(rt, all_tasks):
    require(not (rt.get('mix_only') is True and rt.get('scene_only') is True), 'Conflicting requested scopes')
    if rt.get('scene_only') is True:
        selected = [item for item in all_tasks if '/MIX' not in item['capsule']]
        require(len(selected) == 1260 and all(item['points'] == [[5,20]] for item in selected), 'Wrong scene-only matrix')
        return selected
    if rt.get('mix_only') is not True:
        return all_tasks
    selected = [item for item in all_tasks if '/MIX' in item['capsule']]
    require(len(selected) == 672, 'Wrong MIX-only matrix')
    return selected


def conflicting_prior_command(argv, runtime_files):
    return (any(runtime in argv for runtime in runtime_files) and '-m' in argv and
            any(module in argv for module in (suite.__package__+'.r42_fast_takeover',
                                              suite.__package__+'.milkr_r42_suite')))


def resume_stopped_scenes(rt, output):
    """Use only closed prior predictions; never compete with a prior owner."""
    require(rt.get('scene_only') is True, 'Stopped-owner resume requires scene-only scope')
    runtimes=rt['prior_runtime_files']
    require(isinstance(runtimes,list) and rt['predecessor_runtime_file'] in runtimes and len(set(runtimes))==len(runtimes), 'Bad predecessor runtime list')
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        try:
            if folder.stat().st_uid != os.getuid(): continue
            argv=(folder/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except (FileNotFoundError,ProcessLookupError):
            continue
        require(not conflicting_prior_command(argv,runtimes), 'Prior R42 owner or worker still running')
    available=[gpu for gpu in rt['allowed_gpus'] if int(subprocess.check_output(
        ['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) >= rt['minimum_free_MiB']]
    require(available, 'No authorized GPU has enough free memory for source smoke')
    smoke_dir=output.parent/'smoke';smoke_dir.mkdir()
    for seed in rt['model_seeds']:
        destination=smoke_dir/f'cuda-{seed}.json'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(available[0]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable,'-u','-m',__package__+'.r42_fast_takeover','--runtime',rt['runtime_file'],
                        '--smoke-seed',str(seed),'--smoke-output',str(destination)],env=env,stdin=subprocess.DEVNULL,check=True)
        smoke=read(destination)
        require(smoke['status']=='PASS' and smoke['seed']==seed and smoke['query_rows']==0, 'Source smoke failed')
    write(output,dict(status='SCENE_RESUME_READY', prior_runtime_files=runtimes, inherited_workers={}))
    return {}


def dispatch(rt):
    old = read(rt['predecessor_runtime_file'])
    all_tasks = suite.tasks(rt); by_key = {item['key']: item for item in all_tasks}
    require(suite.tasks(old) == all_tasks and rt['local_ridge_cuda'] is True, 'Takeover matrix/backend differs')
    tasks = selected_tasks(rt, all_tasks)
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
    if rt.get('scene_only') is True:
        predecessor=resume_stopped_scenes(rt,root/'takeover.json')
    elif rt.get('failed_takeover_file'):
        predecessor=adopt_orphans(rt,old,by_key)
        predecessor=smoke_adopted(rt,old,predecessor,by_key,root/'takeover.json')
    else:
        predecessor = transfer_owner(rt, old, rt['predecessor_controller_pid'], by_key, root/'takeover.json')
    # Do not relaunch or terminate a predecessor's already running single-scene
    # task. Let it close in its original immutable root before dispatching MIX.
    while any('/MIX' not in by_key[key]['capsule'] for key in predecessor):
        for key, record in list(predecessor.items()):
            if '/MIX' in by_key[key]['capsule'] or alive(record): continue
            require(closed_ref(old, by_key[key]) is not None, 'Single-scene predecessor did not close')
            del predecessor[key]
        if any('/MIX' not in by_key[key]['capsule'] for key in predecessor): time.sleep(rt['poll_seconds'])
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
                if used >= gpu_worker_limit(rt, gpu): continue
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
    scope='SCENE_ONLY' if rt.get('scene_only') is True else 'MIX_ONLY' if rt.get('mix_only') is True else 'GROUP1_FULL'
    require(rows == {'SCENE_ONLY':27720,'MIX_ONLY':139776,'GROUP1_FULL':167496}[scope], 'Wrong total prediction rows')
    write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED',tasks=len(tasks),rows=rows,scope=scope,truth_read=False,query_fit=False))
    score_queue(rt,tasks,rt.get('max_cpu_scorers',1))
    write(root/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(tasks),rows=rows,scope=scope,truth_last=True,target_feedback=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True)
    parser.add_argument('--smoke-seed',type=int);parser.add_argument('--smoke-output')
    args=parser.parse_args();runtime=read(args.runtime)
    if args.smoke_seed is not None:
        require(args.smoke_output is not None, 'Smoke output required')
        source_smoke(runtime,args.smoke_seed,args.smoke_output)
    else:
        dispatch(runtime)
