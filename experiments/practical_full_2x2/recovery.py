"""Bounded legacy-Torch startup recovery; preserve healthy scientific PIDs."""
from copy import deepcopy
import os,shutil,signal,subprocess,time
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .cache import atomic

FINGERPRINT="AttributeError: module 'torch' has no attribute 'set_float32_matmul_precision'"

def check_identity(actual,cwd,expected_argv,expected_cwd):
    if actual!=expected_argv or cwd!=expected_cwd:raise ValueError('Owned PID/CWD/argv mismatch; no stop/adoption')

def process_identity(pid):
    proc=Path('/proc')/str(pid)
    return [x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x],os.readlink(proc/'cwd')

def replacements(rt,previous,jobs,state):
    """Pure config resolution: no PID, IQ, model, prediction or truth access."""
    root=Path(rt['run_root']);folder=root/'recovery_configs';folder.mkdir()
    failed=[j for j in jobs if j['kind']=='F' and state['jobs'][j['id']].get('failure')==FINGERPRINT]
    if len(failed)!=2:raise ValueError('Expected exactly the two legacy API startup failures')
    row_ids={read(j['config'])['row_id'] for j in failed};selected=[];paths={}
    for j in jobs:
        c=read(j['config'])
        if c.get('row_id') not in row_ids or j['kind'] not in ('F','predict','score'):continue
        if state['jobs'][j['id']]['status'] not in ('FAILED','BLOCKED_TECHNICAL_DEPENDENCY'):raise ValueError('Refuse to replace a healthy scientific row')
        selected.append(j);row=c['row_id'];out=root/'technical_replacements'/row
        newout=out/('source' if j['kind']=='F' else 'predictions' if j['kind']=='predict' else 'scores')
        paths[c['output_root']]=str(newout)
    if len(selected)!=6:raise ValueError('Only two F/predict/score chains may be replaced')
    def remap(value):
        if isinstance(value,str):
            for old,new in paths.items():
                if value==old or value.startswith(old+'/'):return new+value[len(old):]
            return value
        if isinstance(value,list):return [remap(x) for x in value]
        if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
        return value
    selected_ids={j['id'] for j in selected};resolved=[];mapping=[]
    for job in jobs:
        j=dict(job,worker_release=job.get('worker_release',previous['release_root']))
        resolved.append(j)
        if j['id'] in selected_ids:
            c=remap(read(j['config']));c['code_commit']=rt['code_commit'];new_id=j['id']+'--legacy-api-recovery-r04'
            path=folder/(new_id+'.json');write(path,c)
            replacement=remap(j);replacement.update(id=new_id,config=str(path),worker_release=rt['release_root'],log_name=new_id)
            resolved.append(replacement);mapping.append(dict(old_job=j['id'],new_job=new_id,logical_row=c['row_id'],new_output=c['output_root']))
        elif j['kind']=='F' and state['jobs'][j['id']]['status']=='QUEUED':
            c=read(j['config']);c['code_commit']=rt['code_commit'];path=folder/(j['id']+'.json');write(path,c)
            j.update(config=str(path),worker_release=rt['release_root'])
    return resolved,mapping

def recover(rt):
    from .runner import command,schedule
    from experiments.cric_dura_comparison.pipeline import alive,done
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    previous=read(rt['recover_previous_runtime']);oldroot=Path(previous['run_root'])
    state=read(oldroot/'pipeline_state.json');jobs=read(oldroot/'jobs.json')
    if previous['allowed_gpus']!=rt['allowed_gpus'] or set(rt['allowed_gpus'])&{5,6,7}:raise ValueError('Reserved GPU policy changed')
    if state.get('failures',{}).get(FINGERPRINT)!=2:raise ValueError('Only the locally reproduced startup fingerprint is resolved')
    resolved,mapping=replacements(rt,previous,jobs,state)
    # Stop exactly the owned admission controller. Its detached healthy
    # science workers and cache producer keep their original PID/CWD/config.
    pid=state['pid']
    if alive(pid):
        argv,cwd=process_identity(pid);check_identity(argv,cwd,command(previous,'dispatch',previous['runtime_file']),previous['release_root'])
        os.kill(pid,signal.SIGTERM)
        for _ in range(50):
            if not alive(pid):break
            time.sleep(.1)
        if alive(pid):raise RuntimeError('Previous owner did not exit; no second dispatcher')
    latest=read(oldroot/'pipeline_state.json');by={j['id']:j for j in jobs};active={}
    for key,item in latest['jobs'].items():
        if item['status']!='RUNNING':continue
        job=by[key]
        if done(job):latest['jobs'][key]=dict(item,status='COMPLETE');continue
        if not alive(item['pid']):raise RuntimeError('Reconcile finished worker before recovery: '+key)
        argv,cwd=process_identity(item['pid'])
        check_identity(argv,cwd,command(previous,'worker',job['config'],job['python']),job.get('worker_release',previous['release_root']))
        active[key]=dict(item,adopted=True)
    excluded=set(latest.get('excluded_job_ids',[]));excluded.update(m['old_job'] for m in mapping)
    for m in mapping:latest['jobs'][m['new_job']]=dict(status='QUEUED')
    failures=dict(latest.get('failures',{}));failures.pop(FINGERPRINT)
    latest.update(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=command(rt,'dispatch',rt['runtime_file']),
        code_commit=rt['code_commit'],previous_owner_pid=pid,owner=rt['launch_owner'],failures=failures,
        resolved_technical_failures={FINGERPRINT:2},excluded_job_ids=sorted(excluded),adopted=len(active),recovery_started_at=time.time())
    producer=read(oldroot/'augmentation_producer.json');producer_pid=producer['pid']
    if not alive(producer_pid):raise RuntimeError('Augmentation producer must be reconciled separately; no duplicate producer')
    argv,cwd=process_identity(producer_pid);check_identity(argv,cwd,producer['argv'],producer['cwd'])
    write(root/'augmentation_producer.json',producer)
    (root/'data').mkdir();write(root/'data/handles.json',read(oldroot/'data/handles.json'))
    for name in ('source_smoke.json','preparation_complete.json'):shutil.copyfile(oldroot/name,root/name)
    write(root/'jobs.json',resolved);write(root/'technical_recovery.json',dict(status='RESOLVED_EXECUTION_STARTUP',fingerprint=FINGERPRINT,
        replacements=mapping,healthy_workers=active,producer=producer,new_scientific_models=0,source_IQ_regenerated=0))
    atomic(oldroot/'handoff.json',dict(new_owner_pid=os.getpid(),new_runtime=rt['runtime_file'],new_root=str(root)))
    atomic(root/'pipeline_state.json',latest)
    return schedule(rt,resolved,latest,active,producer_pid)
