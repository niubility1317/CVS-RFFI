"""Replace only the queue owner; adopt healthy rows and share GPUs2/3."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from experiments.phase1_external_dg.data import read,write
from experiments.phase1_external_dg.driver import atomic,environment,process_alive,failure_fingerprint


def may_launch(gpu,foreign_count,free_mib):
    if foreign_count==0:return free_mib>=12000
    return gpu in (2,3) and foreign_count==1 and free_mib>=12000


def active_rows(rows,excluded):
    if set(excluded)-{5}:raise ValueError('Only the user-excluded five-source scope may be removed')
    return [r for r in rows if r['source_count'] not in excluded]


def worker_identity(argv,cwd,rt):
    prefix=[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','worker','--config']
    if argv[:6]!=prefix or len(argv)!=7 or str(cwd)!=rt['release_root']:return None
    config=Path(argv[6])
    if config.parent!=Path(rt['run_root']) or config.suffix!='.json':return None
    return config.stem


def discover(rt,lookup):
    found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
            rid=worker_identity(argv,os.readlink(str(p/'cwd')),rt)
            if rid is None:continue
            if rid not in lookup:raise ValueError('Unknown live worker in this run')
            env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
            gpu=int(env[b'CUDA_VISIBLE_DEVICES'])
            if gpu in found:raise ValueError('Duplicate own workers on one GPU')
            found[gpu]=dict(pid=int(p.name),row=lookup[rid],process=None,log=None,argv=argv)
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return found


def adopt(c,control):
    rt=c['runtime'];root=Path(rt['run_root']);old=c['old_dispatcher_pid']
    expected=c.get('old_owner_argv') or [rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','dispatch','--config',rt['runtime_file']]
    expected_cwd=c.get('old_owner_cwd',rt['release_root'])
    proc=Path('/proc')/str(old)
    actual=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if actual!=expected or os.readlink(str(proc/'cwd'))!=expected_cwd:
        raise ValueError('Queue-owner identity mismatch; no process touched')
    plan=read(rt['matrix']);lookup={r['row_id']:r for r in plan['rows']}
    required=active_rows(plan['rows'],c.get('excluded_source_counts',[]))
    required_ids={r['row_id'] for r in required}
    stopped=False;terminated=False
    try:
        os.kill(old,signal.SIGSTOP);stopped=True
        for _ in range(40):
            status=(proc/'status').read_text()
            if any(line.startswith('State:') and line.split()[1] in ('T','t') for line in status.splitlines()):break
            time.sleep(.05)
        else:raise RuntimeError('Queue owner did not stop; takeover cancelled')
        state=read(root/'pipeline_state.json')
        if state['pid']!=old:raise ValueError('Another owner already took over')
        active=discover(rt,lookup)
        write(control/'takeover_snapshot.json',dict(old_state=state,adopted={str(g):dict(pid=x['pid'],row_id=x['row']['row_id']) for g,x in active.items()}))
        os.kill(old,signal.SIGTERM);terminated=True;os.kill(old,signal.SIGCONT)
        for _ in range(100):
            if not process_alive(old,expected_cwd):break
            time.sleep(.05)
        else:raise RuntimeError('Old queue still alive; no new rows launched')
    finally:
        if stopped and not terminated:
            try:os.kill(old,signal.SIGCONT)
            except ProcessLookupError:pass
    # Reconcile the possible Popen->state-write gap while the old owner was
    # frozen. Live workers are found from their exact command and original CWD.
    for gpu,item in active.items():
        rid=item['row']['row_id']
        state['rows'][rid]=dict(status='RUNNING',pid=item['pid'],gpu=gpu,argv=item['argv'],
            cwd=rt['release_root'],log=str(Path(rt['log_root'])/(rid+'.log')),adopted=True)
    live={x['row']['row_id'] for x in active.values()}
    failures=Counter()
    for row in plan['rows']:
        rid=row['row_id'];done=root/rid/'complete.json'
        if rid in live:continue
        if done.is_file():
            d=read(done)
            if d['row_id']!=rid or d['status']!='ANALYZED':raise ValueError('Invalid existing row closure')
            state['rows'].setdefault(rid,{})['status']='ANALYZED'
        elif rid not in required_ids:
            state['rows'].setdefault(rid,{}).update(status='SKIPPED_USER',
                reason='five-source results already exist; no new five-source work requested')
        elif rid in state['rows'] or (root/rid).exists():
            entry=state['rows'].setdefault(rid,{})
            logfile=entry.get('log',str(Path(rt['log_root'])/(rid+'.log')))
            fp=failure_fingerprint(logfile,rid);entry.update(status='FAILED',failure=fp,log=logfile);failures[fp]+=1
    pending=[r for r in plan['rows'] if r['row_id'] not in state['rows']]
    state.update(pid=os.getpid(),cwd=str(control),previous_dispatcher_pid=old,controller_commit=c['control_commit'],
        worker_code_commit=rt['code_commit'],shared_gpus=[2,3],status='RUNNING',pending_rows=len(pending),
        excluded_source_counts=c.get('excluded_source_counts',[]),effective_required_rows=len(required),
        effective_required_row_ids=sorted(required_ids))
    atomic(root/'pipeline_state.json',state)
    return state,active,pending,failures


def run(c):
    rt=c['runtime'];control=Path(c['control_root']);root=Path(rt['run_root'])
    state,active,pending,failures=adopt(c,control)
    while pending or active:
        for gpu,item in list(active.items()):
            if item['process'] is not None:
                code=item['process'].poll();alive=code is None
            else:code=None;alive=process_alive(item['pid'],rt['run_root'])
            if alive:continue
            rid=item['row']['row_id'];done=root/rid/'complete.json';entry=state['rows'][rid]
            success=done.is_file() and read(done).get('row_id')==rid and read(done).get('status')=='ANALYZED'
            entry.update(status='ANALYZED' if success else 'FAILED',returncode=code)
            if item['log']:item['log'].close()
            if not success:
                fp=failure_fingerprint(entry['log'],rid);failures[fp]+=1;entry['failure']=fp
            del active[gpu]
        if pending and failures and max(failures.values())>=2:
            state['unstarted_rows']=[r['row_id'] for r in pending];pending.clear()
            state['stop_reason']='two_matching_technical_failures'
        for gpu in rt['allowed_gpus']:
            if gpu in active or not pending:continue
            apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if not may_launch(gpu,len(apps),free):continue
            row=pending.pop(0);rid=row['row_id'];config=root/(rid+'.json')
            handle=read(root/'data_handles.json')[row['protocol_id']]
            value=dict(runtime=rt,row=row,handle=handle)
            if config.exists():
                if read(config)!=value:raise ValueError('Unstarted row config differs')
            else:write(config,value)
            path=Path(rt['log_root'])/(rid+'.log')
            if path.exists() and path.stat().st_size:raise ValueError('Unstarted row log already contains output')
            log=path.open('a' if path.exists() else 'x')
            command=[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','worker','--config',str(config)]
            child=subprocess.Popen(command,cwd=rt['release_root'],env=environment(rt,gpu),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=dict(pid=child.pid,row=row,process=child,log=log)
            state['rows'][rid]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,cwd=rt['release_root'],log=str(path),adopted=False)
        state.update(status='RUNNING' if active else 'WAITING_GPU',pending_rows=len(pending))
        atomic(root/'pipeline_state.json',state)
        if pending or active:time.sleep(30)
    state['status']='PARTIAL' if failures or state.get('unstarted_rows') else 'TRAINING_PREDICTIONS_SCORING_DONE'
    atomic(root/'pipeline_state.json',state)
    analysis_runtime=dict(rt,excluded_source_counts=c.get('excluded_source_counts',[]))
    analysis_config=control/'analysis_runtime.json';write(analysis_config,analysis_runtime)
    analysis_env=environment(rt)
    analysis_env['PYTHONPATH']=str(control)+os.pathsep+analysis_env['PYTHONPATH']
    with (control/'analysis.log').open('x') as log:
        subprocess.run([rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','analysis','--config',str(analysis_config)],
            cwd=control,env=analysis_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(read(a.config))
