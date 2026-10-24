"""Exact blocked-owner recovery; preserve all scientific jobs and healthy PIDs."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import shutil
from experiments.cric_dura_comparison.io import read,write

FINGERPRINT="KeyError: 'frozen_sources'"
FAILED_ROWS=('F0-supervised-s2026092701','F0-supervised-s2026092702')

def validate_owner(info,rt):
    expected=[rt['python'],'-u','-m','experiments.pamf_repl_completion.runner','dispatch','--config',rt['runtime_file']]
    if info['cwd']!=rt['release_root'] or info['argv']!=expected:
        raise ValueError('Exact blocked dispatch owner identity mismatch')

def process_info(pid):
    p=Path('/proc')/str(pid)
    try:
        if p.joinpath('stat').read_text().split()[2]=='Z':return None
        env=dict(x.split(b'=',1) for x in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
        return dict(cwd=str(p.joinpath('cwd').resolve()),argv=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x],gpu=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
    except (FileNotFoundError,ProcessLookupError):return None

class AdoptedProcess:
    def __init__(self,pid):self.pid=pid
    def poll(self):return None if process_info(self.pid) is not None else 0

def attach(jobs,state):
    active={};children={};failures=Counter()
    for job in jobs:
        item=state['jobs'][job['id']]
        if item['status']=='FAILED':failures[item['failure']]+=1
        if item['status']!='RUNNING':continue
        info=process_info(item['pid'])
        if info is not None:
            if info['cwd']!=item['cwd'] or info['argv']!=item['argv'] or info['gpu']!=('' if item['gpu'] is None else str(item['gpu'])):
                raise ValueError('Healthy worker identity changed; refuse duplicate adoption')
        # A worker that just completed is observed by the normal completion path.
        active[job['id']]=dict(item);children[job['id']]=AdoptedProcess(item['pid'])
    return active,children,failures

def prepare(rt):
    from experiments.cric_dura_comparison.pipeline import done
    oldrt=read(rt['recover_previous_runtime']);oldroot=Path(oldrt['run_root']);root=Path(rt['run_root'])
    state=read(oldroot/'pipeline_state.json');oldjobs=read(oldroot/'jobs.json')
    if process_info(state['pid']) is not None:raise ValueError('Old owner still alive; reconcile handoff before recovery')
    if state['code_commit']!=rt['previous_execution_commit']:raise ValueError('Unexpected previous owner version')
    if not state.get('systemic_stop_new_launches') or state.get('failures')!={FINGERPRINT:2}:raise ValueError('Recovery scope differs from reproduced failure')
    result=[];newstate=deepcopy(state);mapping=[];cfgroot=root/'configs';cfgroot.mkdir()
    for oldjob in oldjobs:
        job=deepcopy(oldjob);item=state['jobs'][job['id']]
        if item['status']=='FAILED':
            c=read(job['config'])
            if job['kind']!='score' or c['row_id'] not in FAILED_ROWS or item['failure']!=FINGERPRINT:raise ValueError('Unrelated failure is not authorized for replacement')
            if Path(c['output_root']).exists() or Path(job['completion']).exists():raise ValueError('Score may already exist; reconcile instead of overwriting')
            output=root/'score_recovery'/c['row_id'];c['output_root']=str(output)
            path=cfgroot/('score_recovery__'+c['row_id']+'.json');write(path,c)
            oldid=job['id'];job.update(id='score_recovery__'+c['row_id'],config=str(path),completion=str(output/'complete.json'),execution_release=rt['release_root'])
            del newstate['jobs'][oldid];newstate['jobs'][job['id']]=dict(status='QUEUED')
            mapping.append(dict(old_job=oldid,new_job=job['id'],old_completion=oldjob['completion'],new_completion=job['completion'],historical_status='FAILED_PRESERVED_EXCLUDED'))
        else:
            # All unstarted scientific workers retain the original immutable CWD,
            # source configuration, outputs and execution code. CPU only is fixed.
            job['execution_release']=rt['release_root'] if job['kind']=='score' else oldrt['release_root']
            if item['status']=='COMPLETE' and not done(job):raise ValueError('Previously complete artifact is missing')
        result.append(job)
    if len(mapping)!=2 or len(result)!=733 or sum(j['kind']=='score' for j in result)!=228:raise ValueError('Scientific matrix changed')
    smoke=read(oldroot/'smoke.json')
    if smoke['status']!='PASS' or smoke['target_reads'] or smoke['U_truth_reads']:raise ValueError('Original unchanged source smoke invalid')
    shutil.copyfile(oldroot/'smoke.json',root/'smoke.json');shutil.copyfile(oldroot/'sources.json',root/'sources.json')
    newstate['resolved_technical_failures']={FINGERPRINT:dict(count=2,replacement=mapping,scope='CPU entry only; no scientific model changes')}
    newstate['failures']={};newstate['systemic_stop_new_launches']=False
    write(root/'jobs.json',result);write(root/'score_entry_recovery.json',dict(scientific_run_id=rt['run_id'],old_root=str(oldroot),new_root=str(root),
        historical_stages=735,excluded_failed_scores=2,effective_stages=733,new_F=0,new_G=0,new_IQ_cache=0,new_predictions=0,replacements=mapping,
        unchanged_GPU_execution_commit=oldrt['code_commit'],new_CPU_and_owner_execution_commit=rt['code_commit'],source_smoke_reused_unchanged=True))
    return result,newstate
