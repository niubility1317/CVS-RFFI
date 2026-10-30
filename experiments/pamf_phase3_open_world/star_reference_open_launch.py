"""Two matched reference-open rows: source training, full development prediction, score."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .star_representation_launch import _read, _write, _env, _gpu_free, _job

RUN = '20261030-phase3-star-reference-open-s2026092701-r01'
ARMS = ('C0', 'C1')
PRIMARY = 'C1::relation_max'
TRAIN_STATUS = 'STAR_REFERENCE_OPEN_SOURCE_FROZEN'
EVAL_STATUS = 'STAR_REFERENCE_OPEN_PREDICTIONS_COMPLETE'
SCORE_STATUS = 'STAR_REFERENCE_OPEN_SCORES_COMPLETE'
CACHE_RUN = '20261029-phase3-star-representation-s2026092701-r01'
FIELDS = {'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
    'minimum_free_mb','max_added_workers_per_gpu','max_concurrent_rows','rows','novel_truth','target_truth'}


def validate(r):
    from .star_reference_open_train import validate_config as train_validate
    from .star_reference_open_eval import validate_config as eval_validate
    if (not isinstance(r, dict) or set(r) != FIELDS or r.get('run_id') != RUN
            or any(not isinstance(r.get(k), str) or not r[k].strip() for k in
                ('code_commit','run_root','release_root','python','launch_owner','novel_truth','target_truth'))
            or re.fullmatch('[0-9a-f]{40}', r['code_commit']) is None
            or r.get('allowed_gpus') != list(range(8)) or r.get('minimum_free_mb') != 8192
            or r.get('max_added_workers_per_gpu') != 2 or type(r.get('max_concurrent_rows')) is not int
            or r['max_concurrent_rows'] != 2 or not isinstance(r.get('rows'), dict)
            or set(r['rows']) != set(ARMS)):
        raise ValueError('Only the fixed two-row development runtime and scorer-only truth fields are allowed')
    root = Path(r['run_root']); cache = root.parent / CACHE_RUN / 'cache'
    for arm, row in r['rows'].items():
        if not isinstance(row, dict) or set(row) != {'train_config','eval_config'}:
            raise ValueError('One independent train/eval pair per arm required')
        t, e = row['train_config'], row['eval_config']; train_validate(t); eval_validate(e)
        if (t['arm'] != arm or e['arm'] != arm
                or any(c['run_id'] != RUN or c['code_commit'] != r['code_commit'] for c in (t, e))
                or t['steps'] != 2220 or t['lr'] != .0001 or t['sampling_seed'] != 2026103017 or t['slot_seed'] != 2026103018
                or Path(t['output_root']) != root/'rows'/arm/'train'
                or Path(t['L_root']) != cache/'L' or Path(t['OE_root']) != cache/'OE'
                or Path(e['cache_root']) != cache or Path(e['output_root']) != root/'rows'/arm/'predict'
                or e['train_config'] != t or Path(e['checkpoint']) != root/'rows'/arm/'train/final.pt'):
            raise ValueError('Owned arm/run/code/budget or original-F0 source-cache paths differ')
    return r


def start(runtime):
    r = validate(_read(runtime)); root = Path(r['run_root'])
    if root.exists():
        raise FileExistsError('Observe existing output; never relaunch or overwrite')
    root.mkdir(); (root/'configs').mkdir(); (root/'logs').mkdir(); (root/'rows').mkdir()
    for arm, row in r['rows'].items():
        (root/'rows'/arm).mkdir()
        for role, config in row.items():
            _write(root/'configs'/(arm+'_'+role+'.json'), config)
    command = [r['python'],'-B','-u','-m',__package__+'.star_reference_open_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=r['release_root'], env=_env(r), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(run_id=RUN, code_commit=r['code_commit'], pid=child.pid, argv=command, cwd=r['release_root'],
        time=time.time(), launch_owner=r['launch_owner'])
    _write(root/'launch.json', value); print(json.dumps(value), flush=True)


def _terminal(r, arm, stage):
    folder = dict(train='train', eval='predict', score='score')[stage]
    result = _read(Path(r['run_root'])/'rows'/arm/folder/'complete.json')
    status = dict(train=TRAIN_STATUS, eval=EVAL_STATUS, score=SCORE_STATUS)[stage]
    if any(result.get(k) != v for k, v in dict(status=status, run_id=RUN, code_commit=r['code_commit'], arm=arm).items()):
        raise ValueError(stage+' complete identity differs')
    if stage == 'train':
        if (result.get('updates') != 2220 or result.get('cumulative_updates') != 13320
                or any(result.get(k) != 0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads'))
                or result.get('identity_head_replaced') is not False):
            raise ValueError('Source training budget/default-head permissions differ')
    elif stage == 'eval':
        if (result.get('known_count') != 27000 or result.get('novel_count') != 3150 or result.get('target_count') != 864
                or any(result.get(k) != 0 for k in ('source_truth_reads','target_truth_reads'))
                or result.get('target_Q_fit') is not False or result.get('formal_target_accessed') is not False
                or result.get('thresholds_frozen_before_novel_and_target') is not True
                or result.get('source_unknown_success_target_entry_gate') is not False):
            raise ValueError('Full frozen development prediction must precede independent truth scoring')
    elif (type(result.get('eligible_primary')) is not bool or not isinstance(result.get('next_step'), str) or not result['next_step']
            or result.get('truth_last') is not True or result.get('formal_target_accessed') is not False
            or result.get('final_phase3_complete') is not False):
        raise ValueError('Independent development score terminal fields differ')
    return result


def row_driver(runtime, arm, gpu):
    r = validate(_read(runtime))
    if arm not in ARMS or gpu not in r['allowed_gpus']:
        raise ValueError('Only an owned arm and declared GPU may run')
    root = Path(r['run_root']); path = root/'rows'/arm/'state.json'
    if path.exists():
        raise FileExistsError('Existing owned row state must be observed, never restarted')
    state = dict(run_id=RUN, arm=arm, code_commit=r['code_commit'], pid=os.getpid(), status='RUNNING',
        started_at=time.time(), jobs={}); _write(path, state)
    commands = [
        ('train',[r['python'],'-B','-u','-m',__package__+'.star_reference_open_train','--config',str(root/'configs'/(arm+'_train_config.json')),'--device','cuda:0'],gpu),
        ('eval',[r['python'],'-B','-u','-m',__package__+'.star_reference_open_eval','--config',str(root/'configs'/(arm+'_eval_config.json')),'--device','cuda:0'],gpu),
        ('score',[r['python'],'-B','-u','-m',__package__+'.star_reference_open_score','--prediction-root',str(root/'rows'/arm/'predict'),
            '--novel-truth',r['novel_truth'],'--target-truth',r['target_truth'],'--output',str(root/'rows'/arm/'score')],None)]
    try:
        for stage, command, device in commands:
            if device is not None:
                while device not in _gpu_free(r):
                    state['status'] = 'WAITING_MEMORY'; _write(path, state); time.sleep(20)
                state['status'] = 'RUNNING'; _write(path, state)
            if not _job(r, state, path, arm+'_'+stage, command, device):
                raise RuntimeError(stage+' failed; preserve partial output, no automatic restart')
            result = _terminal(r, arm, stage)
        state.update(status='ANALYZED_DEVELOPMENT', eligible_primary=result['eligible_primary'],
            next_step=result['next_step'], formal_target_accessed=False, final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED', error=str(exc))
    state['finished_at'] = time.time(); _write(path, state); print(state['status'], arm, flush=True)


def _available_gpu(r, active):
    # A foreign process never excludes a card; actual free memory decides.
    counts = {g: sum(v['gpu'] == g for v in active.values()) for g in r['allowed_gpus']}
    available = [g for g in _gpu_free(r) if counts[g] < r['max_added_workers_per_gpu']]
    return min(available, key=lambda g: counts[g]) if available else None


def driver(runtime):
    r = validate(_read(runtime)); root = Path(r['run_root']); path = root/'pipeline_state.json'
    if path.exists():
        raise FileExistsError('Existing pipeline state must be observed, never restarted')
    state = dict(run_id=RUN, code_commit=r['code_commit'], pid=os.getpid(), status='RUNNING', started_at=time.time(), jobs={})
    _write(path, state); pending = list(ARMS); active = {}
    try:
        while pending or active:
            while pending and len(active) < r['max_concurrent_rows']:
                gpu = _available_gpu(r, active)
                if gpu is None:
                    break
                arm = pending.pop(0)
                command = [r['python'],'-B','-u','-m',__package__+'.star_reference_open_launch','row',
                    '--runtime',str(runtime),'--arm',arm,'--gpu',str(gpu)]
                log = (root/'logs'/(arm+'_driver.log')).open('xb')
                try:
                    child = subprocess.Popen(command, cwd=r['release_root'], env=_env(r), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                except Exception:
                    log.close(); raise
                active[arm] = dict(process=child, gpu=gpu, log=log)
                state['jobs'][arm] = dict(status='RUNNING', pid=child.pid, gpu=gpu, argv=command,
                    cwd=r['release_root'], started_at=time.time(), minimum_free_mb=r['minimum_free_mb'])
                _write(path, state)
            for arm, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                item['log'].close(); sp = root/'rows'/arm/'state.json'; row = _read(sp) if sp.is_file() else {}
                state['jobs'][arm].update(status='COMPLETE' if code == 0 and row.get('status') == 'ANALYZED_DEVELOPMENT' else 'FAILED',
                    returncode=code, finished_at=time.time())
                del active[arm]; _write(path, state)
            if pending or active:
                time.sleep(20 if pending and not active else 2)
        complete = all(state['jobs'][arm]['status'] == 'COMPLETE' for arm in ARMS)
        primary = _terminal(r, 'C1', 'score') if complete else None
        state.update(status='ANALYZED_DEVELOPMENT' if complete else 'PARTIAL_FAILED_PRESERVED', primary=PRIMARY,
            all_rows_complete=complete, eligible_primary=complete and primary['eligible_primary'],
            next_step=primary['next_step'] if complete else 'NO_PROMOTION', formal_target_accessed=False, final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED', error=str(exc), eligible_primary=False, final_phase3_complete=False)
    state['finished_at'] = time.time(); _write(path, state); print(state['status'], flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['start','driver','row']); p.add_argument('--runtime', required=True)
    p.add_argument('--arm', choices=ARMS); p.add_argument('--gpu', type=int); a = p.parse_args()
    if a.mode == 'row':
        if a.arm is None or a.gpu is None:
            p.error('row requires --arm and --gpu')
        return row_driver(a.runtime, a.arm, a.gpu)
    return dict(start=start, driver=driver)[a.mode](a.runtime)


if __name__ == '__main__':
    main()
