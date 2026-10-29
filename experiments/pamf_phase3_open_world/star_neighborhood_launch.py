"""One owned GPU worker: source anchors, unlabeled U/future Q, independent score."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .star_representation_launch import _read, _write, _env, _gpu_free, _job

RUN = '20261029-phase3-star-neighborhood-s2026092701-r01'
FIELDS = {'run_id','code_commit','run_root','release_root','python','launch_owner',
          'allowed_gpus','minimum_free_mb','max_added_workers_per_gpu','source_config',
          'target_config','U_truth','Q_truth','targets_path'}


def validate(r):
    from .star_neighborhood_run import validate_source_config, validate_target_config
    if (not isinstance(r, dict) or set(r) != FIELDS or r.get('run_id') != RUN
            or r.get('allowed_gpus') != list(range(8)) or r.get('minimum_free_mb') != 4096
            or r.get('max_added_workers_per_gpu') != 2
            or any(not isinstance(r[k], str) or not r[k].strip() for k in
                   ('run_root','release_root','python','launch_owner','U_truth','Q_truth','targets_path'))):
        raise ValueError('Only the fixed single-worker development chain is allowed')
    s, t, root = r['source_config'], r['target_config'], Path(r['run_root'])
    validate_source_config(s); validate_target_config(t)
    if (any(c['run_id'] != RUN or c['code_commit'] != r['code_commit'] for c in (s,t))
            or Path(s['output_root']) != root/'source'
            or Path(t['output_root']) != root/'prediction'
            or Path(t['source_root']) != root/'source'
            or root.resolve().is_relative_to(Path(s['L_cache_root']).resolve())):
        raise ValueError('Owned identities and immutable source/prediction roots differ')
    return r


def start(runtime):
    r = validate(_read(runtime)); root = Path(r['run_root'])
    if root.exists(): raise FileExistsError('Observe the existing run; never overwrite or relaunch')
    root.mkdir(); (root/'logs').mkdir(); (root/'configs').mkdir()
    for name in ('source','target'): _write(root/'configs'/(name+'.json'), r[name+'_config'])
    command = [r['python'],'-B','-u','-m',__package__+'.star_neighborhood_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=r['release_root'], env=_env(r), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(run_id=RUN, code_commit=r['code_commit'], pid=child.pid, argv=command,
                 cwd=r['release_root'], time=time.time(), launch_owner=r['launch_owner'])
    _write(root/'launch.json', value); print(json.dumps(value), flush=True)


def driver(runtime):
    r = validate(_read(runtime)); root = Path(r['run_root']); path = root/'pipeline_state.json'
    state = dict(run_id=RUN, code_commit=r['code_commit'], pid=os.getpid(), status='RUNNING',
                 started_at=time.time(), jobs={})
    _write(path, state)
    try:
        for name, mode in (('source','source'),('target','target')):
            available = _gpu_free(r)
            if not available: raise RuntimeError('No card has the declared actual free memory; preserve root')
            command = [r['python'],'-B','-u','-m',__package__+'.star_neighborhood_run',mode,
                       '--config',str(root/'configs'/(name+'.json')),'--device','cuda:0']
            if not _job(r,state,path,name,command,available[0]):
                raise RuntimeError(name+' failed; preserve partial output without automatic restart')
        command = [r['python'],'-B','-u','-m',__package__+'.star_neighborhood_score',
                   '--predictions-root',str(root/'prediction'),'--U-truth',r['U_truth'],
                   '--Q-truth',r['Q_truth'],'--targets',r['targets_path'],'--output',str(root/'score')]
        if not _job(r,state,path,'score',command,None): raise RuntimeError('Independent score failed; preserve output')
        from .star_neighborhood_run import STATUS
        from .star_neighborhood_score import STATUS as SCORED
        predicted, scored = _read(root/'prediction/complete.json'), _read(root/'score/complete.json')
        if (predicted.get('status') != STATUS or scored.get('status') != SCORED
                or any(x.get('run_id') != RUN for x in (predicted,scored))
                or predicted.get('code_commit') != r['code_commit']
                or type(scored.get('primary_pass')) is not bool):
            raise ValueError('Complete predictions and independent score identities differ')
        state.update(status='ANALYZED_DEVELOPMENT', primary_pass=scored['primary_pass'],
                     next_step=scored['next_step'], formal_target_reads=0, final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED', error=str(exc))
    state['finished_at'] = time.time(); _write(path,state); print(state['status'],flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode',choices=['start','driver']); p.add_argument('--runtime',required=True)
    a = p.parse_args(); return dict(start=start,driver=driver)[a.mode](a.runtime)


if __name__ == '__main__': main()
