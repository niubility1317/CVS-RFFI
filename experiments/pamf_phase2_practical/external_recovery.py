"""Repair only the two MoPC implementations; adopt all healthy Phase2 workers."""
import os
from pathlib import Path
import signal
import time
from .io import read,write,require
from .external_takeover import process


def jobs(rt):
    old=read(rt['continuation_runtime']);oldroot=Path(old['run_root']);root=Path(rt['run_root'])
    profiles=set(rt['repair_profiles']);result=[]
    require(profiles=={'mopc_paper','mopc_public_code'},'Recovery is limited to the two failed MoPC versions')
    for row in read(oldroot/'jobs.json'):
        j=dict(row);j['dependencies']=list(row['dependencies'])
        if j['profile'] in profiles and j['kind'] in ('predict','score'):
            j['output']=str(root/Path(j['output']).relative_to(oldroot))
            if j['kind']=='score':j['prediction']=str(root/Path(j['prediction']).relative_to(oldroot))
            j['dependencies'].append('repair-source-smoke')
            if j.get('seed')==rt['model_seeds'][0] and j.get('capsule')==rt['capsules'][0]:j['priority']=0
            j['implementation_revision']='MoPC_C_support_train_mode_r02'
        else:
            j['worker_runtime']=old['runtime_file'];j['worker_release']=old['release_root']
        result.append(j)
    result.append(dict(id='repair-source-smoke',kind='repair_smoke',profile='repair_source_only',
        output=str(root/'repair_source_smoke'),dependencies=[],gpu=True,priority=-1))
    return result


def source_smoke(rt,job):
    import torch
    from .io import activate
    from .external_execution import source_runtime,smoke_bundle
    from .external_native import load_source
    cfg=source_runtime(rt);activate(cfg);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks={}
    for profile in rt['repair_profiles']:
        bundle=load_source(cfg,profile,rt['model_seeds'][0],'cuda:0')
        checks[profile]=smoke_bundle(cfg,bundle,'cuda:0')
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'complete.json',dict(status='PASS',checks=checks,target_rows=0,query_rows=0,
        code_commit=rt['code_commit'],source_contract_reused=True))


def adopt(rt,output):
    old=read(rt['continuation_runtime']);root=Path(old['run_root']);state=read(root/'pipeline_state.json')
    require(old['run_id']=='20261029-phase2-external-threepool-m4-r01','Unexpected continuation run')
    failed=[read(root/'job_results'/(key+'.json')) for key in state.get('failed_jobs',[])]
    require({x['profile'] for x in failed}==set(rt['repair_profiles']) and len(failed)==2,
            'Unexpected failure set; reconcile before taking over')
    parent=process(state['controller_pid']);require(parent is not None,'Existing controller disappeared; reconcile first')
    require(parent['uid']==os.getuid() and parent['cwd']==old['release_root'] and
        old['runtime_file'] in parent['argv'] and 'experiments.pamf_phase2_practical.external_dispatch' in parent['argv'] and
        '--job' not in parent['argv'],'Existing dispatcher identity mismatch')
    os.kill(parent['pid'],signal.SIGSTOP);workers={}
    try:
        time.sleep(.2)
        core=read(old['shared_core_runtime']);paths={old['runtime_file']:old['release_root'],core['runtime_file']:core['release_root']}
        oldjobs={j['id']:j for j in read(root/'jobs.json')}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            item=process(int(p.name))
            if not item or item['uid']!=os.getuid() or '--job' not in item['argv']:continue
            bound=next((path for path,cwd in paths.items() if path in item['argv'] and item['cwd']==cwd),None)
            if bound is None:continue
            module='experiments.pamf_phase2_practical.'+('episodic_dispatch' if bound==core['runtime_file'] else 'external_dispatch')
            require(module in item['argv'],'Unexpected worker module on owned runtime')
            key=item['argv'][item['argv'].index('--job')+1]
            if bound==core['runtime_file']:key='core__'+key
            require(key in oldjobs and oldjobs[key]['profile'] not in rt['repair_profiles'],'Unexpected/active MoPC worker')
            try:env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
            except (FileNotFoundError,ProcessLookupError):continue
            gpu=env.get('CUDA_VISIBLE_DEVICES','');j=oldjobs[key]
            prior=state.get('active',{}).get(key,{})
            log=prior.get('log',str((Path(core['run_root']) if j['kind']=='core' else root)/'logs'/((j.get('original_id',key))+'.log')))
            workers[key]=dict(item,gpu=int(gpu) if gpu.isdigit() else None,kind=j['kind'],profile=j['profile'],
                original_kind=j.get('original_kind'),log=log,adopted=True)
        os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT)
        deadline=time.monotonic()+5
        while process(parent['pid']) and time.monotonic()<deadline:time.sleep(.1)
        require(process(parent['pid']) is None,'Previous controller did not exit')
    finally:
        if process(parent['pid']):os.kill(parent['pid'],signal.SIGCONT)
    write(output,dict(status='VERIFIED',previous_run=old['run_id'],old_dispatcher_pid=parent['pid'],
        old_dispatcher_exited=True,model_workers_restarted=0,adopted_workers=list(workers.values()),
        unchanged_source_training=True,unchanged_other_profiles=True,failed_profiles=rt['repair_profiles']))
    return old,workers
