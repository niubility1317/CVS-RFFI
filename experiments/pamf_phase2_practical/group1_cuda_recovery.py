"""Fresh-root CUDA recovery of the unchanged MoPC Group1 matrix, without A training."""
import argparse
from collections import Counter
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from .io import read, write, atomic, require
from .group1 import source_scope
from .group1_dispatch import jobs, completed


def matrix(rt, parent):
    require(rt['method_profiles'] == parent['method_profiles'] == ['mopc_hr'], 'Recovery is limited to MoPC')
    require(rt['model_seeds'] == parent['model_seeds'] == [2026092701,2026092702,2026092703,2026092704], 'Recovery seed matrix differs')
    require(rt['capsules'] == parent['capsules'] and len(rt['capsules']) == 483, 'Recovery capsule matrix differs')
    require(rt['data_root'] == parent['data_root'], 'Recovery target data differs')
    require(read(rt['source_contract']) == read(parent['source_contract']), 'Recovery source contract differs')
    current, old = jobs(rt), {j['id']: j for j in jobs(parent)}
    require(len(current) == len(old) == 3870 and {j['id'] for j in current} == set(old), 'Recovery job matrix differs')
    sources = [j['id'] for j in current if j['kind'] == 'source']
    for job in current:
        origin = old[job['id']]
        require(all(job.get(k) == origin.get(k) for k in ('kind','profile','seed','capsule','scope','gpu','status')),
                'Recovery scientific job identity differs')
        job['recovery_parent_output'] = origin['output']
        if job['kind'] != 'source':
            job['dependencies'] = list(dict.fromkeys(sources + job['dependencies']))
    return current, old


def link_tree(origin, destination, source=False):
    """Only regular files; every link remains immutable in the new worker path."""
    origin, destination = Path(origin), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    for path in sorted(origin.rglob('*')):
        if not stat.S_ISREG(path.lstat().st_mode):
            continue
        relative = path.relative_to(origin)
        if source and relative == Path('group1_complete.json'):
            relative = Path('recovery_parent_group1_complete.json')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        count += 1
    return count


def prediction_complete(job):
    if not completed(job):
        return False, 0
    root = Path(job['output'])
    index_path = root / 'predictions.json'
    if not index_path.is_file():
        return False, 0
    done, index = read(job['completion']), read(index_path)
    rows = index.get('rows', [])
    valid = (done.get('truth_read') is False and done.get('query_fit') is False
        and index.get('model_seed') == job['seed'] and index.get('capsule') == job['capsule']
        and done.get('capsule_id') == index.get('capsule_id') and bool(rows)
        and done.get('rows') == len(rows)
        and all((root / row['prediction']).is_file() for row in rows))
    return bool(valid), len(rows) if valid else 0


def score_complete(job, prediction_rows):
    if not completed(job):
        return False
    done = read(job['completion'])
    return (done.get('truth_last') is True and done.get('target_feedback') is False
        and done.get('rows') == prediction_rows and (Path(job['output']) / 'metrics.json').is_file())


def prepare(rt, parent):
    root, old_root = Path(rt['run_root']), Path(parent['run_root'])
    require(root.resolve() != old_root.resolve() and old_root.resolve() not in root.resolve().parents,
            'Recovery must use an independent immutable root')
    current, old = matrix(rt, parent)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'job_results').mkdir()
    write(root / 'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv,
        runtime=rt['runtime_file'], parent_runtime=rt['recovery_parent_runtime'],
        source_training=False, augmentation_service_started=False))
    write(root / 'jobs.json', current)
    atomic(root / 'pipeline_state.json', dict(status='PREPARING_REUSE', pid=os.getpid(),
        cwd=str(Path.cwd()), argv=sys.argv, prepared_jobs=0, total_jobs=len(current), active={}))
    reused, row_counts = {}, {}
    for position, job in enumerate(current):
        origin = old[job['id']]
        if job['kind'] == 'source':
            require(completed(origin), 'Frozen parent source is incomplete')
            marker = read(origin['completion'])
            require(marker.get('seed') == job['seed'] and marker.get('profile') == 'mopc_hr'
                    and (Path(origin['output']) / 'source.pt').is_file(), 'Parent source seed/checkpoint mismatch')
            count = link_tree(origin['output'], job['output'], source=True)
            write(Path(job['output']) / 'recovery_source_lineage.json', dict(
                parent_run_id=parent['run_id'], parent_source=origin['output'],
                parent_runtime=rt['recovery_parent_runtime'], linked_regular_files=count,
                parent_checkpoint_identity_preserved=True, source_validation_pending=True))
        elif job['kind'] == 'predict':
            good, rows = prediction_complete(origin)
            if not good:
                continue
            link_tree(origin['output'], job['output'])
            reused[job['id']] = dict(kind='predict', rows=rows, reused_from=origin['output'])
            row_counts[job['id']] = rows
        elif job['kind'] == 'score':
            prediction_id = 'predict-' + job['id'].split('score-predict-', 1)[1]
            rows = row_counts.get(prediction_id)
            if rows is None or not score_complete(origin, rows):
                continue
            link_tree(origin['output'], job['output'])
            reused[job['id']] = dict(kind='score', rows=rows, reused_from=origin['output'])
        if job['id'] in reused:
            write(Path(job['output']) / 'recovery_reuse.json', dict(reused[job['id']],
                parent_runtime=rt['recovery_parent_runtime'], source_training='parent frozen A; no target training',
                source_revalidation_required_before_new_admission=True))
        if position % 100 == 0 or job['kind']=='source':
            atomic(root / 'pipeline_state.json', dict(status='PREPARING_REUSE', pid=os.getpid(),
                cwd=str(Path.cwd()), argv=sys.argv, prepared_jobs=position+1, total_jobs=len(current),
                reused_jobs=len(reused), active={}))
    write(root / 'reuse.json', reused)
    return current, reused


def reuse_source(rt, job, device='cuda:0'):
    """First and only source action: existing provenance checks and real-L smoke."""
    import torch
    from .io import activate
    from .group1_source import load_source, source_data, recipe
    config = source_scope(rt)
    activate(config)
    torch.set_num_threads(config.get('source_cpu_threads', 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle = load_source(config, job['profile'], job['seed'], device)
    data = source_data(config, 'L_s')
    with torch.inference_mode():
        logits = bundle['score_fn'](bundle['model'], data.x[:2].to(device))
    require(tuple(logits.shape) == (2,6) and bool(torch.isfinite(logits).all()), 'Reused real-checkpoint no-query smoke failed')
    out = Path(job['output'])
    write(out / 'recovery_source_smoke.json', dict(status='PASS', seed=job['seed'],
        real_checkpoint=True, source_role='L_s', source_rows=2, target_rows=0, query_rows=0,
        new_optimizer_updates=0, source_new_optimizer_updates=0))
    write(out / 'group1_complete.json', dict(status='GROUP1_SOURCE_READY', profile=job['profile'],
        seed=job['seed'], recipe=recipe(job['profile']), real_checkpoint_smoke='PASS',
        reused_from=job['recovery_parent_output'], checkpoint=str(out / 'source.pt'),
        source_checkpoint_origin_identity_preserved=True, source_contract=config['source_contract'],
        target_reads=0, query_reads=0, new_optimizer_updates=0, source_new_optimizer_updates=0))


def worker(rt, job):
    if job['kind'] == 'source':
        return reuse_source(rt, job)
    if job['kind'] == 'predict':
        from .group1_dispatch import worker as original_worker
        return original_worker(rt, job)
    if job['kind'] == 'score':
        from .episodic_score import run
        require(read(Path(job['prediction']) / 'complete.json')['status'] == 'PREDICTIONS_COMPLETE',
                'Scoring requires completed prediction')
        return run(rt, job)
    raise ValueError('Unknown recovery job kind')


def eligible_gpu(device, own, minimum_free=6000):
    return own < 1 and device['free'] >= max(6000, minimum_free)


def observe_children(active, by, done, failed, root):
    rows = {}
    for key, (process, item) in list(active.items()):
        code = process.poll()
        if code is None:
            continue
        marker_error = None
        try:
            good = code == 0 and completed(by[key])
            if good and by[key]['kind'] != 'source':
                rows[key] = int(read(by[key]['completion'])['rows'])
        except (OSError, ValueError, KeyError) as error:
            good = False
            marker_error = repr(error)
        result = dict(item, status='COMPLETE' if good else 'FAILED', exit_code=code, ended_at=time.time())
        if marker_error is not None:
            result.update(failed_layer='completion_marker_readback', error=marker_error)
        if good and by[key]['kind'] != 'source':
            result['rows'] = rows[key]
        write(root / 'job_results' / (key + '.json'), result)
        (done if good else failed).add(key)
        del active[key]
    return rows


def dispatch(rt):
    require(rt['allowed_gpus'] == [1,5], 'Recovery GPU allocation must be physical 1 and 5')
    parent = read(rt['recovery_parent_runtime'])
    current, reused = prepare(rt, parent)
    root = Path(rt['run_root'])
    by = {j['id']: j for j in current}
    done, failed, active = set(reused), set(), {}
    remaining = set(by) - done
    errors = []
    completed_rows = {key: value['rows'] for key,value in reused.items()}
    while True:
        completed_rows.update(observe_children(active, by, done, failed, root))
        stop = (root / 'USER_STOP.json').is_file()
        available = sorted((by[k] for k in remaining if all(d in done for d in by[k]['dependencies'])),
            key=lambda j: (j['kind'] != 'source', j['priority'], j['id'])) if not failed and not errors and not stop else []
        devices = {}
        if any(j['gpu'] for j in available):
            try:
                from .runner import occupancy
                devices = occupancy()
            except Exception as error:
                errors.append(dict(layer='GPU_occupancy', error=repr(error)))
                available = []
        def launch(job, gpu):
            command = [rt['python'], '-u', '-m', __package__ + '.group1_cuda_recovery',
                       '--runtime', rt['runtime_file'], '--job', job['id']]
            env = dict(os.environ, PYTHONPATH=rt['release_root'],
                CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            log = root / 'logs' / (job['id'] + '.log')
            with log.open('x') as stream:
                process = subprocess.Popen(command, cwd=rt['release_root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            item = dict(pid=process.pid, gpu=gpu, kind=job['kind'], profile=job['profile'],
                cwd=rt['release_root'], argv=command, log=str(log), started_at=time.time())
            active[job['id']] = (process, item)
            remaining.remove(job['id'])
            available.remove(job)
        if available:
            try:
                if not any(item['kind'] == 'score' for _, item in active.values()):
                    scorer = next((j for j in available if j['kind'] == 'score'), None)
                    if scorer:
                        launch(scorer, None)
                for gpu in rt['allowed_gpus']:
                    own = sum(item['gpu'] == gpu for _, item in active.values())
                    if gpu not in devices or not eligible_gpu(devices[gpu], own, rt.get('minimum_free_MiB',6000)):
                        continue
                    job = next((j for j in available if j['gpu']), None)
                    if job:
                        launch(job, gpu)
            except Exception as error:
                errors.append(dict(layer='worker_launch', error=repr(error)))
        terminal_failure = bool(failed or errors)
        status = ('STOP_NEW_ADMISSIONS_TECHNICAL_FAILURE' if terminal_failure else
                  'STOP_REQUESTED' if stop else 'RUNNING')
        if not active:
            if terminal_failure:
                status = 'PARTIAL_TECHNICAL_FAILURE'
            elif stop:
                status = 'STOPPED'
            elif not remaining:
                status = 'COMPLETE'
        counts = dict(COMPLETE=len(done), FAILED=len(failed), RUNNING=len(active), QUEUED=len(remaining))
        reuse_counts = Counter(v['kind'] for v in reused.values())
        state = dict(status=status, pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv,
            updated_at=time.time(), run_id=rt['run_id'], counts=counts,
            active={k: item for k, (_, item) in active.items()}, failed_jobs=sorted(failed),
            technical_errors=errors, reused_prediction_jobs=reuse_counts['predict'],
            reused_score_jobs=reuse_counts['score'],
            reused_prediction_rows=sum(v['rows'] for v in reused.values() if v['kind']=='predict'),
            reused_score_rows=sum(v['rows'] for v in reused.values() if v['kind']=='score'),
            completed_prediction_rows=sum(v for k,v in completed_rows.items() if by[k]['kind']=='predict'),
            completed_score_rows=sum(v for k,v in completed_rows.items() if by[k]['kind']=='score'),
            verified_source_jobs=sum(by[k]['kind']=='source' for k in done),
            source_new_optimizer_updates=0, augmentation_service_started=False,
            admission_rule='one own GPU worker per physical GPU1/5, free>=6000MiB, one CPU scorer',
            stop_rule='any technical failure stops all new admissions; drain healthy owned workers; no retry')
        atomic(root / 'pipeline_state.json', state)
        if not active and (terminal_failure or stop or not remaining):
            break
        time.sleep(rt.get('poll_seconds',10))
    if status == 'COMPLETE':
        write(root / 'complete.json', dict(status='SCORED_COMPLETE', run_id=rt['run_id'], jobs=len(current),
            source_new_optimizer_updates=0, reused_prediction_jobs=reuse_counts['predict'],
            reused_score_jobs=reuse_counts['score'], truth_last=True, target_feedback=False))
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--job')
    args = parser.parse_args()
    runtime = read(args.runtime)
    if args.job:
        found = [j for j in read(Path(runtime['run_root']) / 'jobs.json') if j['id'] == args.job]
        require(len(found) == 1, 'Unknown recovery job')
        worker(runtime, found[0])
    else:
        dispatch(runtime)
