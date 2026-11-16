"""Native satellite registration, with fixed A0 reuse and per-task scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import atomic, read, require, write
from .satellite_native_da import RECEIVERS, SEEDS

PROFILES=('orthogonal','lora_triplet_knn','mopc_hr')
EPISODES=tuple([f'MIX{i:02}' for i in range(1,9)]+[f'{p}{i:02}' for p in ('LS','MU','LU') for i in range(1,6)])


def validate_scope(job):
    parts=job.get('capsule','').split('/')
    require(len(parts)==4 and parts[0]=='capsules' and parts[1] in ('P01','P02','P03') and
            parts[2] in EPISODES and parts[3] in RECEIVERS and job.get('seed') in SEEDS and
            job.get('profile') in PROFILES,'Invalid satellite registration identity')
    episode=parts[2];profile=job['profile']
    points=[(k,n) for k in (1,5,10,20) for n in (2,5,10,20)] if episode.startswith('MIX') else [(5,20)]
    a0=([1,5,10,20] if episode.startswith('MIX') else [5]) if profile=='orthogonal' else \
       ([1,10,20] if episode.startswith('MIX') and episode!='MIX01' else [])
    require(job.get('satellite_c_points')==[list(p) for p in points] and job.get('satellite_a0_ks')==a0,
            'Wrong registered rows or A0 reuse scope')
    require(job.get('mopc_stabilization','none')==('clip1000_v1' if profile=='mopc_hr' else 'none'),
            'Wrong native C version')
    return points,a0


def verify_prediction(job):
    folder=Path(job['output']);done=read(folder/'complete.json');index=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('profile')==job['profile'] and done.get('model_seed')==job['seed'] and
            index.get('capsule')==job['capsule'] and index.get('capsule_id')==done.get('capsule_id') and
            index.get('model_seed')==job['seed'],'Wrong native C identity')
    points,a0=validate_scope(job)
    carm={'mopc_hr':'mopc_hr_clip1000_v1','lora_triplet_knn':'lora_triplet_knn_balanced_v1',
          'orthogonal':'orthogonal_tempered_pl_v2'}[job['profile']]
    a0arm='orthogonal_tempered_pl_v2' if job['profile']=='orthogonal' else job['profile']
    expected={('A0',k,0,'P0',a0arm) for k in a0}
    expected.update(('C',k,n,'P0',carm) for k,n in points)
    rows=index['rows'];actual=[(v['stage'],v['k'],v['new_count'],v['input_path'],v['arm']) for v in rows]
    require(len(actual)==len(set(actual)) and set(actual)==expected and len(rows)==done['rows'] and
            all(v.get('input_condition')=='satellite_received' and (folder/v['prediction']).is_file() for v in rows),
            'Wrong satellite native C rows')


def make_plan(a0_inventory,runtime_files,run_root,release_root,prior_configs,wait_run):
    require(a0_inventory.get('status')=='VERIFIED' and not a0_inventory.get('issues'),'Original A0 inventory not verified')
    root=Path(run_root).resolve();release=Path(release_root).resolve()
    require(not root.exists() and release.is_dir() and len(prior_configs)==2,'Wrong native C output or prior queues')
    prior=[read(p) for p in prior_configs];gpus=[0,1,5,6,7]
    require(set(prior[0]['allowed_gpus'])==set(gpus) and set(prior[1]['allowed_gpus']).isdisjoint(gpus) and
            Path(prior[0]['run_root']).resolve()==Path(wait_run).resolve(),'Wrong predecessor GPU binding')
    for profile in ('lora_triplet_knn','mopc_hr'):
        info=a0_inventory['counts'][profile]
        require(info['expected']==3948 and info['existing']==2184 and info['missing']==1764 and
                info['score_jobs']==1933,'Original A0 reuse inventory incomplete')
    available={p:{(v['capsule'],v['seed'],v['k']) for v in a0_inventory['existing'][p]}
               for p in ('lora_triplet_knn','mopc_hr')}
    planned=[]
    source_root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
    expected_source={'orthogonal':source_root/'20261114-phase2-orthogonal-fixed-source-m4-r02',
                     'lora_triplet_knn':source_root/'20261030-phase2-group1-concat-midPL-threepool-m4-r01',
                     'mopc_hr':source_root/'20261101-phase2-mopc-author-group1-m4-r04'}
    for profile in PROFILES:
        rt=read(runtime_files[profile]);require(not rt.get('clean_root') and Path(rt['data_root']).is_dir() and
             Path(rt['source_contract']).is_file() and Path(rt['source_cache']).is_dir(),'Wrong satellite source runtime')
        require(Path(rt['run_root']).resolve()==expected_source[profile].resolve(), 'A0 source provenance mismatch')
        require(all((Path(rt['run_root'])/'source'/f'{profile}-s{seed}'/'source.pt').is_file() for seed in SEEDS),
                'Missing source checkpoint')
        for pool in ('P01','P02','P03'):
            for episode in EPISODES:
                for rx in RECEIVERS:
                    for seed in SEEDS:
                        cap=f'capsules/{pool}/{episode}/{rx}'
                        points=[[k,n] for k in (1,5,10,20) for n in (2,5,10,20)] if episode.startswith('MIX') else [[5,20]]
                        a0=([1,5,10,20] if episode.startswith('MIX') else [5]) if profile=='orthogonal' else \
                           ([1,10,20] if episode.startswith('MIX') and episode!='MIX01' else [])
                        for k in {p[0] for p in points}-set(a0):
                            require((cap,seed,k) in available[profile],'Existing source A0 missing')
                        ident=f'{profile}-s{seed}-capsules-{pool}-{episode}-{rx}'
                        job=dict(id=ident,profile=profile,capsule=cap,seed=seed,satellite_c_points=points,
                                 satellite_a0_ks=a0,output=str(root/'predictions'/ident),score=str(root/'scores'/ident))
                        if profile=='mopc_hr':job['mopc_stabilization']='clip1000_v1'
                        validate_scope(job);planned.append(job)
    require(len(planned)==5796 and len({j['id'] for j in planned})==5796,'Wrong native C matrix')
    return dict(input_condition='satellite_received',run_root=str(root),release_root=str(release),
        runtimes={p:str(Path(runtime_files[p]).resolve()) for p in PROFILES},prior_configs=list(map(str,prior_configs)),
        wait_run=str(Path(wait_run).resolve()),allowed_gpus=gpus,max_per_gpu=7,minimum_free_MiB=6000,
        worker_reserve_MiB=3000,jobs=planned)


def run_worker(config,job):
    from .group1_target import predict
    from .vendor_native.comparison_suite.registration import MoPCNumericalError
    rt=read(config['runtimes'][job['profile']]);require(not rt.get('clean_root'),'Clean input forbidden')
    validate_scope(job)
    try:predict(rt,job)
    except MoPCNumericalError as exc:
        if job['profile']=='mopc_hr' and Path(job['output']).is_dir():
            write(Path(job['output'])/'numerical_failure.json',dict(status='MOPC_NUMERICAL_FAILURE',
                 profile=job['profile'],capsule=job['capsule'],seed=job['seed'],error=str(exc),truth_read=False,query_fit=False))
        raise


def run_score(config,job):
    from .episodic_score import run
    verify_prediction(job);run(read(config['runtimes'][job['profile']]),dict(prediction=job['output'],output=job['score']))


def predecessor_done(config):
    root=Path(config['wait_run'])/'scheduler'
    if (root/'complete.json').is_file():return True
    if not (root/'state.json').is_file():return False
    state=read(root/'state.json')
    return state.get('status') in ('FAILED','STOPPED') and not state.get('active')


def dispatch(config,config_file):
    from .clean_scheduler import choose
    from .runner import occupancy
    root=Path(config['run_root']);require(not root.exists(),'Existing run root')
    root.mkdir(parents=True);control=root/'scheduler';control.mkdir();(control/'logs').mkdir()
    write(control/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    pending=list(config['jobs']);active={};queue=[];scorer=None;done=[];scored=[];failed=[];quarantined=[];stopped=False
    while pending or active or queue or scorer:
        if (control/'USER_STOP.json').is_file():stopped=True;pending.clear()
        for key,item in list(active.items()):
            code=item['proc'].poll()
            if code is None:continue
            item['log'].close();job=item['job'];del active[key]
            try:
                if code==0:verify_prediction(job);done.append(key);queue.append(job);continue
                marker=read(Path(job['output'])/'numerical_failure.json')
                valid=(job['profile']=='mopc_hr' and marker.get('status')=='MOPC_NUMERICAL_FAILURE' and
                       all(marker.get(k)==job.get(k) for k in ('profile','capsule','seed')) and
                       marker.get('truth_read') is False and marker.get('query_fit') is False)
                if valid:quarantined.append(key);continue
            except (OSError,ValueError,KeyError,json.JSONDecodeError):pass
            failed.append(dict(id=key,stage='prediction',exit_code=code));pending.clear()
        if scorer and scorer['proc'].poll() is not None:
            code=scorer['proc'].returncode;scorer['log'].close();job=scorer['job']
            try:ok=code==0 and read(Path(job['score'])/'complete.json').get('status')=='SCORED_COMPLETE'
            except (OSError,ValueError,KeyError,json.JSONDecodeError):ok=False
            if ok:scored.append(job['id'])
            else:failed.append(dict(id=job['id'],stage='score',exit_code=code));pending.clear()
            scorer=None
        if not scorer and queue:
            job=queue.pop(0);log=None
            try:
                log=(control/'logs'/(job['id']+'-score.log')).open('x')
                cmd=[sys.executable,'-u','-m',__package__+'.satellite_native_c','score','--config',config_file,'--job-id',job['id']]
                proc=subprocess.Popen(cmd,cwd=config['release_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                scorer=dict(job=job,log=log,proc=proc)
            except OSError as exc:
                if log is not None:log.close()
                failed.append(dict(id=job['id'],stage='score_launch',error=str(exc)));pending.clear()
        ready=predecessor_done(config)
        if ready and not failed and not stopped and pending:
            try:
                admissions=choose([dict(id=j['id']) for j in pending],
                    {k:dict(pid=v['proc'].pid,gpu=v['gpu']) for k,v in active.items()},occupancy(),config['allowed_gpus'],
                    config['minimum_free_MiB'],config['max_per_gpu'],worker_reserve=config['worker_reserve_MiB'],one_per_snapshot=True)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                failed.append(dict(stage='occupancy',error=str(exc)));pending.clear();admissions=[]
            for entry,gpu in admissions:
                job=next(j for j in pending if j['id']==entry['id']);log=None
                try:
                    log=(control/'logs'/(job['id']+'.log')).open('x')
                    cmd=[sys.executable,'-u','-m',__package__+'.satellite_native_c','worker','--config',config_file,'--job-id',job['id']]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    proc=subprocess.Popen(cmd,cwd=config['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                except OSError as exc:
                    if log is not None:log.close()
                    failed.append(dict(id=job['id'],stage='worker_launch',error=str(exc)));pending.clear();break
                active[job['id']]=dict(job=job,proc=proc,log=log,gpu=gpu);pending.remove(job)
        state='FAILED' if failed else 'STOPPED' if stopped else 'RUNNING' if ready else 'WAITING_PREDECESSOR'
        atomic(control/'state.json',dict(status=state,completed=len(done),scored=len(scored),pending=len(pending),
               active=[dict(id=k,pid=v['proc'].pid,gpu=v['gpu']) for k,v in active.items()],score_pending=len(queue),
               score_active=scorer['job']['id'] if scorer else None,quarantined=quarantined,failed=failed))
        if pending or active or queue or scorer:time.sleep(5)
    if failed:raise RuntimeError('Technical failure retained')
    if not stopped:write(control/'complete.json',dict(status='COMPLETED_WITH_QUARANTINE' if quarantined else 'SCORED_COMPLETE',
                         scored=len(scored),quarantined=quarantined,truth_last=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','dispatch','worker','score'))
    p.add_argument('--config',required=True);p.add_argument('--job-id');p.add_argument('--inventory')
    p.add_argument('--orthogonal-runtime');p.add_argument('--lora-runtime');p.add_argument('--mopc-runtime')
    p.add_argument('--run-root');p.add_argument('--release-root');p.add_argument('--wait-run')
    p.add_argument('--prior-config',action='append',default=[]);a=p.parse_args()
    if a.action=='prepare':
        require(all((a.inventory,a.orthogonal_runtime,a.lora_runtime,a.mopc_runtime,a.run_root,a.release_root,a.wait_run))
                and len(a.prior_config)==2,'Missing native C preparation inputs')
        path=Path(a.config).resolve();require(not path.exists() and not path.parent.exists(),'Plan exists')
        path.parent.mkdir(parents=True,exist_ok=False);runtimes={}
        for profile,source in (('orthogonal',a.orthogonal_runtime),('lora_triplet_knn',a.lora_runtime),('mopc_hr',a.mopc_runtime)):
            runtime=read(source);runtime.pop('clean_root',None);runtime.pop('restored_root',None)
            runtime['release_root']=str(Path(a.release_root).resolve());runtime['runtime_file']=str(path.parent/(profile+'-runtime.json'))
            write(runtime['runtime_file'],runtime);runtimes[profile]=runtime['runtime_file']
        config=make_plan(read(a.inventory),runtimes,a.run_root,a.release_root,a.prior_config,a.wait_run)
        write(path,config);print(json.dumps(dict(status='PREPARED',jobs=len(config['jobs']),truth_read=False)));return
    config=read(a.config)
    if a.action=='dispatch':dispatch(config,a.config);return
    selected=[j for j in config['jobs'] if j['id']==a.job_id];require(len(selected)==1,'Unknown job')
    (run_worker if a.action=='worker' else run_score)(config,selected[0])


if __name__=='__main__':main()
