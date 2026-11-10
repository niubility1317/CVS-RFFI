"""Resume the registered paper matrix in a new root after two diagnosed failures."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.pipeline import done, alive
from experiments.practical_full_2x2.cache import atomic
from . import gpu01_drain


def remap(value, mapping):
    if isinstance(value, dict):
        return {k: remap(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, mapping) for v in value]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if value == old or value.startswith(old + os.sep):
                return new + value[len(old):]
    return value


def source_contract_matches(saved, original, cfg):
    # build_contract_split preserves the original physical roles and adds the
    # actual input representation to its saved training contract.
    return (all(k in saved and saved[k] == value for k, value in original.items()) and
            saved.get('physical_roles') == 'EXACT_MATCH' and
            saved.get('dataset_path') == cfg['dataset'] and
            saved.get('equalized') == 1 and saved.get('out_len') == 256 and
            saved.get('normalize') is True and len(saved.get('classes', [])) == 6)


def plan_recovery(old_root, new_root, jobs, state, sources, recovery_commit):
    old_root, new_root = Path(old_root), Path(new_root)
    mapping = {}
    for job in jobs:
        status = state['jobs'][job['id']]['status']
        if status == 'COMPLETE':
            if not done(job): raise ValueError('Completed job lacks its completion artifact')
            continue
        if status not in ('FAILED', 'QUEUED', 'BLOCKED_TECHNICAL_DEPENDENCY'):
            raise ValueError('Unreconciled previous job: ' + job['id'])
        if status == 'FAILED' and (job['family'] != 'architecture' or job['kind'] != 'train' or
             state['jobs'][job['id']].get('failure') not in ("AttributeError: 'CVSincNet' object has no attribute '_factorial_aux'",
             'ValueError: Saved training evidence is incomplete or incompatible')):
            raise ValueError('Unexpected failure: ' + job['id'])
        old = Path(read(job['config'])['output_root'])
        try: relative = old.relative_to(old_root)
        except ValueError: raise ValueError('Output outside original root')
        mapping[str(old)] = str(new_root / relative)
    rebuilt = []
    for job in jobs:
        job = deepcopy(job); status = state['jobs'][job['id']]['status']
        if status != 'COMPLETE':
            original_cfg = read(job['config']); cfg = remap(original_cfg, mapping)
            if status == 'FAILED':
                cfg['recover_ce_from'] = original_cfg.get('recover_ce_from',original_cfg['output_root'])
                cfg['recovery_commit'] = recovery_commit
            elif job['kind']=='train' and 'recover_ce_from' in cfg:
                cfg['recovery_commit'] = recovery_commit
            elif job['kind']=='train':
                cfg['code_commit'] = recovery_commit
            cfg_path = new_root/'configs'/(job['id']+'.json')
            job['config'] = str(cfg_path)
            job['completion'] = remap(job['completion'], mapping)
            job['dependencies'] = remap(job['dependencies'], mapping)
            job.pop('worker_release', None);job.pop('recovery_log', None)
            job['_new_config'] = cfg
        rebuilt.append(job)
    return rebuilt, remap(sources, mapping)


def recover_ce(c):
    import torch
    from experiments.pamf_repl_completion.runner import activate
    old = Path(c['recover_ce_from']); out = Path(c['output_root'])
    base = dict(c);base.pop('recover_ce_from')
    activate(base)
    from experiments.standard_ce_baselines.source import validate_config
    from experiments.cvs_clean_design.profile import resource_profile
    from .model import ce_build
    # validate_config needs the registered CE builder scope.
    from .model import ce_scope
    with ce_scope(): validate_config(base)
    init = read(old/'initialization.json'); resolved = read(old/'resolved_config.json')
    contract = read(old/'source_contract.json')
    epochs = [__import__('json').loads(x) for x in (old/'epoch_metrics.jsonl').read_text().splitlines()]
    steps = [__import__('json').loads(x) for x in (old/'step_metrics.jsonl').read_text().splitlines()]
    if (len(epochs)!=200 or len(steps)!=10000 or epochs[-1]['epoch']!=200 or
        epochs[-1]['optimizer_steps_total']!=10000 or steps[-1]['step']!=10000 or
        init.get('scratch_only') is not True or init.get('target_access') is not False or
        init.get('model_seed')!=base['model_seed'] or resolved.get('model_seed')!=base['model_seed'] or
        resolved.get('source_counts')!={'L_s':6300,'U_s':56700,'V':27000} or
        not source_contract_matches(contract,read(base['source_contract']),base)):
        raise ValueError('Saved training evidence is incomplete or incompatible')
    device=torch.device('cuda:0'); payload=torch.load(old/'last.pt', map_location=device,weights_only=False)
    if (payload.get('epoch')!=200 or payload.get('variant')!=base['variant'] or
        payload.get('source_contract')!=contract or payload.get('initialization')!=init or
        payload.get('selection')!='fixed_last_epoch'):
        raise ValueError('Saved checkpoint provenance differs')
    net=ce_build(base['variant'],base['model_seed']).to(device)
    net.load_state_dict(payload['model'],strict=True); net.eval()
    with torch.inference_mode():
        scores=net(torch.zeros(2,2,256,device=device))
        if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Checkpoint smoke failed')
    profile=resource_profile(net,device)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('last.pt','source_contract.json','initial_smoke.pt','resolved_config.json',
                 'initialization.json','epoch_metrics.jsonl','epoch_metrics.csv','step_metrics.jsonl'):
        shutil.copyfile(old/name,out/name)
    write(out/'resource_profile.json',profile)
    write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,checkpoint=str(out/'last.pt'),
          target_access=False,target_evaluated=False,final_source_metrics=epochs[-1],recovered_from=str(old),
          recovery_commit=base['recovery_commit']))
    write(out/'freeze.json',dict(status='FROZEN',epoch=200,variant=base['variant'],model_seed=base['model_seed'],
          checkpoint=str(out/'last.pt'),target_access=False,target_score_used=False))
    write(out/'complete.json',dict(status='SOURCE_FROZEN',epoch=200,updates=10000,target_access=False,
          U_truth_read=False,head_fit_updates=0,config=base,recovered_from=str(old),recovery_commit=base['recovery_commit']))


def worker(kind, cfg):
    if kind == 'train' and cfg.get('recover_ce_from'):
        return recover_ce(cfg)
    from .runner import worker as native
    return native(cfg, kind)


def dispatch(rt):
    old_root=Path(rt['old_run_root']); new_root=Path(rt['run_root']); ctl=Path(rt['old_control'])
    previous=read(ctl/'pipeline_state.json')
    if previous['status']!='PARTIAL_TECHNICAL_FAILURE' or alive(previous['pid']):
        raise ValueError('Previous owner not stopped with expected failure')
    jobs=read(ctl/'jobs.json');sources=read(old_root/'sources.json')
    jobs,sources=plan_recovery(old_root,new_root,jobs,previous,sources,rt['code_commit'])
    if sum('recover_ce_from' in j.get('_new_config',{}) for j in jobs)!=6:
        raise ValueError('Expected exactly six E200 post-training failures')
    new_root.mkdir(parents=True,exist_ok=False);(new_root/'configs').mkdir();(new_root/'logs').mkdir()
    for job in jobs:
        cfg=job.pop('_new_config',None)
        if cfg is not None:write(job['config'],cfg)
    write(new_root/'sources.json',sources);write(new_root/'external_sources.json',read(old_root/'external_sources.json'))
    write(new_root/'jobs.json',jobs)
    state=dict(status='SMOKE_PENDING',pid=os.getpid(),owner=rt['launch_owner'],code_commit=rt['code_commit'],
       jobs={j['id']:(dict(status='COMPLETE') if previous['jobs'][j['id']]['status']=='COMPLETE' else dict(status='QUEUED')) for j in jobs},
       failures={},dispatch_gpus=[5,6,7])
    atomic(new_root/'pipeline_state.json',state)
    # Run the source-only smoke on an eligible allowed GPU before dispatch.
    from experiments.pamf_repl_completion.runner import capacity
    from experiments.pamf_branch_factorial.runner import resource_sample, resource_guard
    prev=resource_sample()
    while True:
        time.sleep(1);sample=resource_sample();guard=resource_guard(prev,sample);prev=sample
        gpu=next((g for g in (5,6,7) if guard['allowed'] and capacity(g)[1]>=12000),None)
        if gpu is not None:break
        time.sleep(14)
    env=dict(os.environ,PYTHONPATH=rt['source_release'],CUDA_VISIBLE_DEVICES=str(gpu),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[rt['python'],'-u','-m',rt['worker_module'],'worker','--kind','smoke','--config',rt['runtime_file']]
    with (new_root/'logs'/'smoke.log').open('x') as log:
        child=subprocess.Popen(command,cwd=rt['source_release'],env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['smoke']=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command);atomic(new_root/'pipeline_state.json',state)
        code=child.wait()
    state['smoke'].update(status='PASS' if code==0 else 'FAILED',exit_code=code)
    if code or read(new_root/'smoke.json')['status']!='PASS':
        state['status']='FAILED_NO_QUERY_SMOKE';atomic(new_root/'pipeline_state.json',state)
        raise ValueError('No-query smoke failed; queue remains stopped')
    state['status']='RUNNING';atomic(new_root/'pipeline_state.json',state)
    gpu01_drain.dispatch_gpus=lambda: (5,6,7)
    gpu01_drain.limits=lambda gpu,gpu7_limit=1: gpu7_limit if gpu==7 else 3
    gpu01_drain.loop(rt,new_root,state,jobs,Counter())


def analyze_old(rt):
    old_ctl=Path(rt['old_control']);previous=read(old_ctl/'pipeline_state.json')
    if previous.get('status')!='FAILED_HANDOFF_RECOVERABLE' or previous.get('error')!="KeyError('count')" or alive(previous['pid']):
        raise ValueError('Previous analysis failure differs')
    if any(v.get('status')!='COMPLETE' for v in previous['jobs'].values()):
        raise ValueError('Old queue is not fully complete')
    target=Path(rt['analysis_run_root']);target.mkdir(parents=True,exist_ok=False)
    state=dict(status='ANALYZING',pid=os.getpid(),owner=rt['launch_owner'],code_commit=rt['code_commit'])
    atomic(target/'pipeline_state.json',state)
    try:
        from experiments.pamf_no_reference_completion.analyze import run
        run(dict(rt,run_root=rt['old_run_root'],analysis_root=str(target/'analysis')))
    except Exception as error:
        state.update(status='FAILED_ANALYSIS',error=repr(error),ended_at=time.time())
        atomic(target/'pipeline_state.json',state)
        raise
    state.update(status='ANALYZED',ended_at=time.time());atomic(target/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker','analyze-old'));p.add_argument('--config',required=True);p.add_argument('--kind');a=p.parse_args()
    c=read(a.config)
    if a.action=='dispatch':dispatch(c)
    elif a.action=='analyze-old':analyze_old(c)
    else:worker(a.kind,c)
