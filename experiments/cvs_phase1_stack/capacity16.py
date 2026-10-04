"""User-authorized controller handoff; original worker code/weights stay intact."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CONTROL_RELEASE='cvs_reference_stack_dispatch16_20261004_r01'
WORKER_COMMIT='0c73c904c8331f26254761042516f1ae43ed9c98'
MAX_ACTIVE=16


def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return dict(pid=int(pid),start_ticks=int(stat[19]),cwd=str((p/'cwd').resolve()),
                    argv=[v for v in (p/'cmdline').read_bytes().decode().split('\0') if v])
    except (FileNotFoundError,ProcessLookupError):return None


class Adopted:
    def __init__(self,identity,complete):self.identity=identity;self.pid=identity['pid'];self.complete=complete
    def poll(self):
        now=proc(self.pid)
        if now and now['start_ticks']==self.identity['start_ticks']:
            if now!=self.identity:raise RuntimeError('Adopted process identity changed')
            return None
        return 0 if self.complete() else 1


def write_new_or_same(d,path,value):
    if path.exists():
        if d.read(path)!=value:raise ValueError('Existing config differs: '+str(path))
    else:d.write(path,value)


def complete(d,row,kind):
    folder=Path(d.BASE)/row['row_id']/('source' if kind=='source' else 'prediction')
    path=folder/('completion.json' if kind=='source' else 'phase1_complete.json')
    if not path.exists():return False
    c=d.read(path)
    if kind=='source':
        d.require_budget(c['logged_steps'],c['optimizer_steps'])
        return c['status']=='SOURCE_TRAINED' and c['config']==d.read(row['config']) and c['epoch']==200 and c['target_access'] is False and c['target_evaluated'] is False
    return c['status']=='PREDICTIONS_COMPLETE' and c['count']==168000 and c['truth_read'] is False


def queue(d,jobs,kind,phase,adoptions):
    sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'))
    from scripts.dispatch_xuc_full import available_gpu,occupancy
    launch=Path(d.BASE)/('launch_'+phase+'.json')
    receipts=d.read(launch)['rows'] if launch.exists() else []
    if len({r['row_id'] for r in receipts})!=len(receipts):raise ValueError('Duplicate launch records')
    byid={r['row_id']:r for r in receipts};pending=[];active={};failures=[]
    for row in jobs:
        rid=row['row_id']
        if rid in byid:
            prior=byid[rid];identity=adoptions.get(rid)
            if identity is not None:
                p=Adopted(identity,lambda row=row:complete(d,row,kind))
                code=p.poll()
                if code is None:active[rid]=dict(process=p,gpu=prior['gpu'],row=row)
                elif code:failures.append(dict(row_id=rid,error='Adopted worker ended without valid completion'))
            elif not complete(d,row,kind):raise RuntimeError('Previously launched incomplete worker lacks adoption '+rid)
        elif complete(d,row,kind):raise RuntimeError('Completion without launch receipt '+rid)
        else:pending.append(row)
    while pending or active or failures:
        for rid,job in list(active.items()):
            rc=job['process'].poll()
            if rc is not None:
                if rc or not complete(d,job['row'],kind):failures.append(dict(row_id=rid,exit_code=rc,error='Worker failed or missing valid completion'))
                del active[rid]
        while pending and len(active)<MAX_ACTIVE and not failures:
            gpu=available_gpu(active)
            if gpu is None:break
            caps=occupancy(active)
            if len(caps[gpu]['pids'])>=2 or caps[gpu]['free_mb']<12000:break
            row=pending.pop(0);rid=row['row_id']
            cmd=[sys.executable,'-u','-m','experiments.cvs_phase1_stack.'+kind,'--config',row['config']]
            log=Path(d.PROJECT)/'logs'/d.RUN/(kind+'-'+rid+'.log')
            env=dict(os.environ,PYTHONPATH=str(d.ROOT)+os.pathsep+str(d.ROOT/'code'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            with log.open('x') as h:p=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            active[rid]=dict(process=p,gpu=gpu,row=row)
            receipts.append(dict(row_id=rid,pid=p.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log),controller_release=CONTROL_RELEASE))
            d.write(launch,dict(rows=receipts));print('LAUNCH '+json.dumps(receipts[-1]),flush=True)
        d.write(Path(d.BASE)/'queue_state.json',dict(phase=phase,kind=kind,active=list(active),pending=[r['row_id'] for r in pending],failures=failures,updated_at=time.time(),max_active=MAX_ACTIVE,per_gpu_limit=2,controller_pid=os.getpid(),controller_release=CONTROL_RELEASE))
        if failures and not active:raise RuntimeError('Failed rows preserved; no retry '+repr(failures))
        if active or pending:time.sleep(10)


def run(worker_root,handoff):
    sys.path[:0]=[str(worker_root),str(worker_root/'code')]
    from experiments.cvs_phase1_stack import dispatch as d
    if d.ROOT!=worker_root or (worker_root/'release_commit.txt').read_text().strip()!=WORKER_COMMIT:raise ValueError('Original worker release differs')
    root=Path(d.BASE);prior=d.read(handoff)
    if prior['run_id']!=d.RUN or prior['max_active']!=16 or prior['worker_commit']!=WORKER_COMMIT:raise ValueError('Invalid handoff scope')
    if proc(prior['old_dispatcher']['pid']):raise RuntimeError('Old dispatcher still active')
    if (root/'failure.json').exists() or (root/'completion.json').exists():raise RuntimeError('Run is failed or already completed')
    marker=root/'capacity16_owner.json'
    with marker.open('x') as f:json.dump(dict(pid=os.getpid(),control_release=CONTROL_RELEASE,started=time.time()),f)
    active=dict(pid=os.getpid(),cwd=os.getcwd(),argv=[sys.executable,*sys.argv],owner=d.OWNER,max_active=16,per_gpu_limit=2,
                worker_release=str(worker_root),worker_commit=WORKER_COMMIT,control_release=CONTROL_RELEASE,
                control_commit=(Path(__file__).parent/'release_commit.txt').read_text().strip(),handoff=str(handoff))
    d.write(root/'dispatcher_active.json',active)
    try:
        parent=d.read(root/'parent_source_freeze.json')
        from experiments.cvs_phase1_overlay.dispatch import validate_freeze
        validate_freeze(parent)
        if parent['continuation_arm']!='leo':raise ValueError('Parent source choice changed')
        selected=['leo'];prediction_rows=[]
        for stage in d.STAGES:
            configs=[d.config(r,selected) for r in d.rows() if r['stage']==stage];jobs=[]
            for c in configs:
                path=root/'configs'/('source-'+c['row_id']+'.json');write_new_or_same(d,path,c);jobs.append(dict(row_id=c['row_id'],config=str(path)))
            write_new_or_same(d,root/(stage+'_resolved_matrix.json'),dict(stage=stage,parent_features=selected,rows=configs,target_access=False))
            frozen=root/(stage+'_source_frozen.json')
            if frozen.exists():
                f=d.read(frozen);d.validate_freeze(f,stage)
                if [r['config'] for r in f['rows']]!=configs:raise ValueError('Frozen source configs differ')
            else:
                queue(d,jobs,'source',stage,prior['workers']);f=d.freeze(stage,configs)
            selected=f['features']
            for c in configs:
                path=root/'configs'/('predict-'+c['row_id']+'.json')
                pred=dict(row_id=c['row_id'],source_config=str(root/'configs'/('source-'+c['row_id']+'.json')),source_output=c['output_root'],output_root=d.BASE+'/'+c['row_id']+'/prediction',p1_capsule=d.CAPSULE)
                write_new_or_same(d,path,pred)
                prediction_rows.append(dict(row_id=c['row_id'],config=str(path),output_root=pred['output_root'],model_seed=c['model_seed'],method=c['stage']+'-'+c['arm'],stage='phase12'))
        write_new_or_same(d,root/'all_sources_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=[r['row_id'] for r in d.rows()],stages=list(d.STAGES),target_access=False))
        write_new_or_same(d,root/'evaluation_spec.json',dict(run_id=d.RUN,runtime_root=d.BASE,p1_truth=d.TRUTH,rows=prediction_rows))
        queue(d,prediction_rows,'predict','predict',prior['workers'])
        for module,args in [('comparison_suite.score',['--stage','p1']),('experiments.cvs_phase1_stack.analyze',[])]:
            subprocess.run([sys.executable,'-m',module,'--spec',str(root/'evaluation_spec.json'),*args],cwd=d.ROOT,check=True)
        d.write(root/'completion.json',dict(status='ANALYZED',rows=136,independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as e:
        d.write(root/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True,controller_release=CONTROL_RELEASE));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path,required=True);p.add_argument('--handoff',type=Path,required=True)
    a=p.parse_args();run(a.worker_root,a.handoff)
