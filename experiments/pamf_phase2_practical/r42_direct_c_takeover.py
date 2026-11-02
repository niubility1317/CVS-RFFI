"""Move the unfinished direct-C queue to FP64 CUDA without stopping live workers."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import atomic, read, require, write
from .r42_direct_c_dispatch import gpu_free, jobs
from .r42_fast_takeover import proc, process_running


def remaining(keys, finished, inherited, own):
    require(not (finished & inherited or finished & own or inherited & own), 'Duplicate task ownership')
    require((finished | inherited | own) <= keys, 'Unknown task')
    return keys - finished - inherited - own


def free_slots(gpus, limits, inherited, own, free):
    slots = []
    for gpu in gpus:
        cap = limits[str(gpu)]
        require(type(cap) is int and 1 <= cap <= 3 and (gpu != 7 or cap == 1), 'Invalid concurrency cap')
        used = sum(v['gpu'] == gpu for v in inherited.values()) + sum(v['gpu'] == gpu for v in own.values())
        require(used <= cap, 'Exceeded task-specific GPU cap')
        if used < cap and free.get(gpu, 0) >= 6000:
            slots.append(gpu)
    return slots


def reconcile_old(keys, state_active, inherited, started, complete, child_pids, ignored_pids):
    recorded = {}
    for entry in state_active:
        name = entry['key']
        require(name.startswith('prediction-'), 'Unexpected predecessor active entry')
        key = name[len('prediction-'):]
        require(key in keys and key not in recorded, 'Wrong predecessor active key')
        recorded[key] = entry
        if key in inherited:
            require(entry['pid'] == inherited[key]['pid'] and entry['gpu'] == inherited[key]['gpu'], 'Predecessor PID/GPU changed')
        else:
            require(key in complete, 'Recorded predecessor exited without closure')
    known = {entry['pid'] for entry in recorded.values()} | {entry['pid'] for entry in inherited.values()} | ignored_pids
    require(not (child_pids - known), 'Unidentified predecessor child')
    require(not (started - set(inherited) - complete), 'Started predecessor task without closure')


def closed(old_root, task):
    folder = Path(old_root) / 'tasks' / task['key'] / 'predictions'
    if not (folder / 'complete.json').is_file():
        return None
    done = read(folder / 'complete.json')
    index = read(folder / 'predictions.json')
    require(done['status'] == 'PREDICTIONS_COMPLETE' and done['rows'] == 128
            and done['truth_read'] is False and done['query_fit'] is False
            and done['target_old_support_read'] is False, 'Wrong predecessor closure')
    require(index['capsule'] == task['capsule'] and index['model_seed'] == task['seed']
            and index['capsule_id'] == done['capsule_id'], 'Wrong predecessor prediction identity')
    expected = {(mode, 'C', k, n, f'S{sp}Q{qp}') for mode in ('R42_direct_head', 'R42_direct_source_prototype')
                for k, n in task['points'] for sp in (0, 1) for qp in (0, 1)}
    actual = {(r['arm'], r['stage'], r['k'], r['new_count'], r['input_path']) for r in index['rows']}
    require(actual == expected and len(index['rows']) == 128, 'Wrong predecessor row coverage')
    require(all((folder / r['prediction']).is_file() for r in index['rows']), 'Missing old prediction file')
    return folder


def link_old(root, key, folder):
    parent = Path(root) / 'tasks' / key
    parent.mkdir(parents=True, exist_ok=False)
    os.symlink(str(folder), str(parent / 'predictions'), target_is_directory=True)


def predecessor_workers(pid, old, tasks):
    children = (Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().split()
    found, ignored = {}, set()
    for child in children:
        info = proc(int(child))
        if info is None:
            continue
        argv = info['argv']
        if Path(argv[0]).name == 'nvidia-smi':
            ignored.add(int(child))
            continue
        require('experiments.pamf_phase2_practical.r42_direct_c_worker' in argv and 'predict' in argv
                and '--runtime' in argv and '--job' in argv, 'Unexpected predecessor child')
        require(info['uid'] == os.getuid() and info['cwd'] == old['release_root']
                and argv[argv.index('--runtime')+1] == old['runtime_file'], 'Wrong predecessor child identity')
        job_file = Path(argv[argv.index('--job')+1])
        require(job_file.parent == Path(old['run_root']) / 'jobs', 'Wrong predecessor job path')
        key = job_file.stem
        require(key in tasks and key not in found, 'Unknown predecessor worker')
        job = read(job_file)
        require(job['seed'] == tasks[key]['seed'] and job['capsule'] == tasks[key]['capsule']
                and job['output'] == str(Path(old['run_root'])/'tasks'/key/'predictions'), 'Wrong predecessor job identity')
        gpu = [value.split('=', 1)[1] for value in info['env'] if value.startswith('CUDA_VISIBLE_DEVICES=')]
        require(len(gpu) == 1 and gpu[0].isdigit(), 'Unknown predecessor GPU')
        found[key] = dict(pid=info['pid'], start=info['start'], gpu=int(gpu[0]))
    return found, {int(child) for child in children}, ignored


def takeover(runtime):
    rt = read(runtime)
    require(rt.get('direct_c_cuda') is True and rt.get('direct_c_pair_budget') == 262144
            and rt.get('direct_c_cuda_min_support') == 256, 'Wrong CUDA execution configuration')
    old = dict(run_root=rt['predecessor_root'], runtime_file=rt['predecessor_runtime'], release_root=rt['predecessor_release'])
    previous = read(old['runtime_file'])
    for key in ('data_root', 'source_parent_root', 'source_cache', 'source_contract', 'PAMF_sources',
                'old_classes', 'new_classes', 'DURA', 'model_seeds', 'capsules', 'target_receivers',
                'phase1_release', 'batch_size', 'decision_batch'):
        require(rt.get(key) == previous.get(key), 'Predecessor scientific setting differs: ' + key)
    require(previous['run_root'] == old['run_root'] and rt['run_root'] != old['run_root'], 'Wrong output ownership')
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    for name in ('tasks', 'jobs', 'logs', 'scores', 'smoke'):
        (root / name).mkdir()
    os.symlink(str(Path(old['run_root']) / 'sources'), str(root / 'sources'), target_is_directory=True)
    task_list = jobs(rt)
    tasks = {task['key']: task for task in task_list}
    require(len(tasks) == 84 and read(Path(old['run_root'])/'worker_state.json')['phase'] == 'predictions', 'Predecessor is not predicting')
    require(not (Path(old['run_root'])/'failed.json').exists(), 'Predecessor already failed')
    # One real-checkpoint, source-only numerical smoke per seed, prior to transfer.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    require(gpu_free().get(0, 0) >= 6000, 'Not enough free GPU0 memory for source-only smoke')
    for seed in rt['model_seeds']:
        file = root / 'smoke' / f'source-{seed}.json'
        cmd = [sys.executable, '-u', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_fast_smoke',
               '--runtime', runtime, '--seed', str(seed), '--output', str(file)]
        subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL)
        result = read(file)
        require(result['status'] == 'PASS' and result['target_reads'] == 0 and result['query_reads'] == 0, 'Source-only CUDA smoke failed')
    pid = rt['predecessor_controller_pid']
    info = proc(pid)
    require(info is not None and info['uid'] == os.getuid() and info['cwd'] == old['release_root'], 'Wrong predecessor controller')
    argv = info['argv']
    require('experiments.pamf_phase2_practical.r42_direct_c_dispatch' in argv and 'dispatch' in argv
            and '--runtime' in argv and argv[argv.index('--runtime')+1] == old['runtime_file'], 'Wrong predecessor command')
    stopped = False
    killed = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            current = proc(pid)
            if current is not None and current['start'] == info['start'] and current['state'] in ('T', 't'):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Could not pause predecessor controller')
        state = read(Path(old['run_root'])/'worker_state.json')
        require(state['phase'] == 'predictions', 'Predecessor entered scoring')
        require(not (Path(old['run_root'])/'failed.json').exists(), 'Predecessor failed during smoke')
        require(not (Path(old['run_root'])/'predictions_complete.json').exists(), 'Predecessor predictions already closed')
        inherited, children, ignored = predecessor_workers(pid, old, tasks)
        initial = {key: closed(old['run_root'], task) for key, task in tasks.items()}
        started = {key for key in tasks if (Path(old['run_root'])/'tasks'/key).exists()}
        complete_keys = {key for key, folder in initial.items() if folder is not None}
        reconcile_old(set(tasks), state['active'], inherited, started, complete_keys, children, ignored)
        require(state['completed']['predictions'] <= len(complete_keys), 'Missing predecessor completed predictions')
        # Even a child with its completion marker can still occupy its GPU;
        # count it until the exact recorded process exits.
        for key in inherited:
            initial[key] = None
        write(root/'takeover.json', dict(status='PAUSED_BEFORE_TRANSFER', predecessor=pid,
              predecessor_start=info['start'], inherited=inherited, prior_closed=sum(v is not None for v in initial.values())))
        current = proc(pid)
        require(current is not None and current['start'] == info['start'] and current['state'] in ('T', 't'), 'Controller identity changed')
        os.kill(pid, signal.SIGKILL)
        killed = True
        for _ in range(100):
            if not process_running(pid, info['start']):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Predecessor controller did not exit')
        atomic(root/'takeover.json', dict(status='TRANSFERRED', predecessor=pid,
              predecessor_start=info['start'], inherited=inherited, prior_closed=sum(v is not None for v in initial.values())))
    finally:
        if stopped and not killed and process_running(pid, info['start']):
            os.kill(pid, signal.SIGCONT)
    finished = set()
    for key, folder in initial.items():
        if folder is not None:
            link_old(root, key, folder)
            finished.add(key)
    own = {}
    scoring = {}
    score_done = set()
    phase = 'predictions'
    while True:
        for key, record in list(inherited.items()):
            if not process_running(record['pid'], record['start']):
                folder = closed(old['run_root'], tasks[key])
                require(folder is not None, 'Inherited worker exited without complete prediction')
                link_old(root, key, folder)
                finished.add(key)
                del inherited[key]
        for key, record in list(own.items()):
            code = record['process'].poll()
            if code is None:
                continue
            record['log'].close()
            require(code == 0 and closed(root, tasks[key]) is not None, 'CUDA worker failed')
            finished.add(key)
            del own[key]
        for key, record in list(scoring.items()):
            code = record['process'].poll()
            if code is None:
                continue
            record['log'].close()
            require(code == 0 and read(root/'scores'/key/'complete.json')['status'] == 'SCORED_COMPLETE', 'Independent score failed')
            score_done.add(key)
            del scoring[key]
        if phase == 'predictions' and len(finished) == len(tasks):
            require(not inherited and not own, 'Workers remain after prediction closure')
            write(root/'predictions_complete.json', dict(status='ALL_PREDICTIONS_FIXED', units=84, rows=10752))
            phase = 'scores'
        if phase == 'scores' and len(score_done) == len(tasks):
            require(not scoring, 'Scorers remain active')
            write(root/'scoring_complete.json', dict(status='SCORED_COMPLETE', units=84, rows=10752))
            atomic(root/'worker_state.json', dict(phase='complete', predictions=84, scores=84))
            return
        if phase == 'predictions':
            pending = sorted(remaining(set(tasks), finished, set(inherited), set(own)))
            while pending:
                slots = free_slots(rt['gpu_ids'], rt['max_workers_by_gpu'], inherited, own, gpu_free())
                if not slots:
                    break
                gpu = slots[0]
                key = pending.pop(0)
                job = dict(tasks[key], output=str(root/'tasks'/key/'predictions'))
                jobfile = root/'jobs'/(key+'.json')
                write(jobfile, job)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                command = [sys.executable, '-u', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_worker',
                           'predict', '--runtime', runtime, '--job', str(jobfile)]
                log = (root/'logs'/('prediction-'+key+'.log')).open('x', encoding='utf-8')
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                own[key] = dict(process=process, log=log, gpu=gpu)
        else:
            pending = sorted(set(tasks)-score_done-set(scoring))
            while pending and len(scoring) < 4:
                key = pending.pop(0)
                command = [sys.executable, '-u', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_dispatch',
                           'score', '--runtime', runtime, '--key', key]
                log = (root/'logs'/('score-'+key+'.log')).open('x', encoding='utf-8')
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                scoring[key] = dict(process=process, log=log)
        atomic(root/'worker_state.json', dict(phase=phase, predictions=len(finished), scores=len(score_done),
              pending=len(pending), inherited=inherited,
              active=[dict(key=k, pid=v['process'].pid, gpu=v['gpu']) for k,v in own.items()],
              active_scorers=[dict(key=k, pid=v['process'].pid) for k,v in scoring.items()]))
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    try:
        takeover(args.runtime)
    except Exception as exc:
        root = Path(read(args.runtime)['run_root'])
        if root.is_dir():
            atomic(root/'failed.json', dict(error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
