"""Run explicitly missing satellite native-DA K rows with per-task scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import atomic, read, require, write

PROFILES = ('protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage', 'poster_ft')
RECEIVERS = ('1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8')
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)


def validate_scope(job):
    parts = job.get('capsule', '').split('/')
    require(len(parts) == 4 and parts[0] == 'capsules' and parts[1] in ('P01', 'P02', 'P03')
            and parts[2] in tuple(f'MIX{i:02}' for i in range(2, 9)) and parts[3] in RECEIVERS,
            'Outside missing satellite MIX02-08 scope')
    require(job.get('profile') in PROFILES and job.get('seed') in SEEDS, 'Wrong profile or seed')
    ks = job.get('satellite_ks')
    require(isinstance(ks, list) and ks and all(type(k) is int and k in (1, 10, 20) for k in ks)
            and ks == sorted(set(ks)), 'Invalid or already-covered K')
    return ks


def validate_target(job, missing):
    ks = validate_scope(job)
    require(all((job['capsule'], job['seed'], stage, k, 0) in missing
                for k in ks for stage in ('A0', 'B')), 'Some target rows already exist or lack evidence')
    return ks


def make_plan(inventory, runtime_file, run_root, release_root, profiles, allowed_gpus, prior_configs=()):
    require(inventory.get('status') in ('VERIFIED','INCOMPLETE') and isinstance(inventory.get('issues'),list),
            'Invalid reuse inventory')
    runtime_file=Path(runtime_file).resolve();require(runtime_file.is_file(), 'Runtime file missing')
    runtime=read(runtime_file)
    root=Path(run_root).resolve();release=Path(release_root).resolve()
    require(not root.exists() and release.is_dir(), 'Run exists or release absent')
    require(not runtime.get('clean_root') and runtime.get('data_root') and runtime.get('source_run_root'),
            'Runtime must bind satellite and source data')
    require(Path(runtime['run_root']).resolve()==Path(runtime['source_run_root']).resolve(), 'Wrong source run')
    require(Path(runtime['data_root']).is_dir() and Path(runtime['source_cache']).is_dir() and
            Path(runtime['source_contract']).is_file() and Path(runtime['phase1_release']).is_dir(),
            'Missing fixed data or source handles')
    require(prior_configs, 'Prior satellite queue must be supplied for overlap checks')
    require(profiles and len(set(profiles))==len(profiles) and all(p in PROFILES for p in profiles),
            'Invalid native DA selection')
    require(allowed_gpus and len(set(allowed_gpus))==len(allowed_gpus) and
            len(allowed_gpus) in (3,5) and all(type(g) is int and 0<=g<=7 for g in allowed_gpus),
            'Invalid GPU set')
    planned=[]
    for profile in profiles:
        info=inventory['counts'][profile]
        require(info['expected_rows']==7896 and info['verified_existing']==4368 and
                info['missing']==3528 and info['score_jobs']==1933 and
                not any(profile in str(issue[0]) for issue in inventory['issues']),
                'Selected profile reuse inventory is incomplete')
        missing={(v['capsule'],v['seed'],v['stage'],v['k'],v['n']) for v in inventory['missing'][profile]}
        grouped={}
        for cap,seed,stage,k,n in missing:
            if '/MIX' in cap and cap.split('/')[2]!='MIX01' and k in (1,10,20) and n==0:
                grouped.setdefault((cap,seed),set()).add(k)
        for (cap,seed), ks in sorted(grouped.items()):
            job=dict(profile=profile,capsule=cap,seed=seed,satellite_ks=sorted(ks))
            validate_target(job,missing)
            ident=profile+'-s'+str(seed)+'-'+cap.replace('/','-')
            job.update(id=ident,output=str(root/'predictions'/ident),score=str(root/'scores'/ident))
            planned.append(job)
    require(len(planned)==len({j['id'] for j in planned}) and planned, 'No jobs or duplicate identities')
    require(len(planned)==588*len(profiles) and all(j['satellite_ks']==[1,10,20] for j in planned),
            'Selected missing matrix is incomplete')
    wanted={(j['profile'],j['capsule'],j['seed'],k) for j in planned for k in j['satellite_ks']}
    for name in prior_configs:
        prior=read(name)
        require(set(allowed_gpus).isdisjoint(prior['allowed_gpus']), 'Prior queue shares GPUs')
        owned={(j['profile'],j['capsule'],j['seed'],k) for j in prior['jobs'] for k in j['satellite_ks']}
        require(wanted.isdisjoint(owned), 'Prior queue already owns requested tasks')
    require(all((Path(runtime['source_run_root'])/'source'/f'{profile}-s{seed}'/'source.pt').is_file()
                for profile in profiles for seed in SEEDS), 'Missing source checkpoint')
    for job in planned:
        require(not Path(job['output']).exists() and not Path(job['score']).exists(), 'Output already exists')
    return dict(input_condition='satellite_received',run_root=str(root),release_root=str(release),runtime_file=str(runtime_file),
                allowed_gpus=list(allowed_gpus),prior_configs=[str(Path(p).resolve()) for p in prior_configs],
                max_per_gpu=7,minimum_free_MiB=6000,
                worker_reserve_MiB=3000,one_per_snapshot=True,jobs=planned)


def run_worker(config, job):
    from .group1_target import predict
    validate_scope(job)
    rt=read(config['runtime_file'])
    require(not rt.get('clean_root'), 'Clean data forbidden')
    predict(rt,job)


def verify_prediction(job):
    folder=Path(job['output']);done=read(folder/'complete.json');index=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('profile')==job['profile'] and
            done.get('model_seed')==job['seed'] and index.get('capsule')==job['capsule'] and
            index.get('model_seed')==job['seed'] and index.get('capsule_id')==done.get('capsule_id'),
            'Prediction completion/identity mismatch')
    rows=index['rows'];expected={(stage,k,0,'P0',job['profile']) for stage in ('A0','B') for k in job['satellite_ks']}
    actual=[(v['stage'],v['k'],v['new_count'],v['input_path'],v['arm']) for v in rows]
    require(len(actual)==len(set(actual)) and set(actual)==expected and done['rows']==len(rows) and
            all(v.get('input_condition')=='satellite_received' and (folder/v['prediction']).is_file() for v in rows),
            'Wrong satellite prediction rows')


def run_score(config, job):
    from .episodic_score import run
    rt=read(config['runtime_file'])
    verify_prediction(job)
    run(rt,dict(prediction=job['output'],output=job['score']))


def dispatch(config, config_file):
    from .clean_scheduler import choose
    from .runner import occupancy
    root=Path(config['run_root']);require(not root.exists(), 'Existing run root')
    root.mkdir(parents=True);control=root/'scheduler';control.mkdir();(control/'logs').mkdir()
    write(control/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    pending=list(config['jobs']);active={};score_queue=[];scorer=None;done=[];scored=[];failed=[];stopped=False
    while pending or active or score_queue or scorer:
        if (control/'USER_STOP.json').is_file():stopped=True;pending.clear()
        for key,record in list(active.items()):
            code=record['proc'].poll()
            if code is None:continue
            record['log'].close();job=record['job'];del active[key]
            try:
                marker=read(Path(job['output'])/'complete.json')
                valid=(code==0 and marker.get('status')=='PREDICTIONS_COMPLETE' and
                       marker.get('truth_read') is False and marker.get('query_fit') is False and
                       marker.get('profile')==job['profile'] and marker.get('model_seed')==job['seed'] and
                       marker.get('rows')==2*len(job['satellite_ks']))
            except (OSError,ValueError,KeyError,json.JSONDecodeError):valid=False
            if valid:done.append(key);score_queue.append(job)
            else:failed.append(dict(id=key,stage='prediction',exit_code=code));pending.clear()
        if scorer and scorer['proc'].poll() is not None:
            code=scorer['proc'].returncode;scorer['log'].close();job=scorer['job']
            try:ok=code==0 and read(Path(job['score'])/'complete.json').get('status')=='SCORED_COMPLETE'
            except (OSError,ValueError,KeyError,json.JSONDecodeError):ok=False
            if ok:scored.append(job['id'])
            else:failed.append(dict(id=job['id'],stage='score',exit_code=code));pending.clear()
            scorer=None
        if not scorer and score_queue:
            job=score_queue.pop(0);log=None
            try:
                log=(control/'logs'/(job['id']+'-score.log')).open('x')
                cmd=[sys.executable,'-u','-m',__package__+'.satellite_native_da','score','--config',config_file,'--job-id',job['id']]
                proc=subprocess.Popen(cmd,cwd=config['release_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                scorer=dict(job=job,log=log,proc=proc)
            except OSError as exc:
                if log is not None:log.close()
                failed.append(dict(id=job['id'],stage='score_launch',error=str(exc)));pending.clear()
        if not failed and not stopped and pending:
            candidates=[dict(id=j['id']) for j in pending]
            try:
                admissions=choose(candidates,{k:dict(pid=v['proc'].pid,gpu=v['gpu']) for k,v in active.items()},
                                  occupancy(),config['allowed_gpus'],config['minimum_free_MiB'],config['max_per_gpu'],
                                  worker_reserve=config['worker_reserve_MiB'],one_per_snapshot=True)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                failed.append(dict(stage='occupancy',error=str(exc)));pending.clear();admissions=[]
            for entry,gpu in admissions:
                job=next(j for j in pending if j['id']==entry['id']);log=None
                try:
                    log=(control/'logs'/(job['id']+'.log')).open('x')
                    cmd=[sys.executable,'-u','-m',__package__+'.satellite_native_da','worker','--config',config_file,'--job-id',job['id']]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    proc=subprocess.Popen(cmd,cwd=config['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                except OSError as exc:
                    if log is not None:log.close()
                    failed.append(dict(id=job['id'],stage='worker_launch',error=str(exc)));pending.clear();break
                active[job['id']]=dict(job=job,proc=proc,log=log,gpu=gpu);pending.remove(job)
        status='FAILED' if failed else 'STOPPED' if stopped else 'RUNNING'
        atomic(control/'state.json',dict(status=status,completed=len(done),scored=len(scored),pending=len(pending),
               active=[dict(id=k,pid=v['proc'].pid,gpu=v['gpu']) for k,v in active.items()],
               score_pending=len(score_queue),score_active=scorer['job']['id'] if scorer else None,failed=failed))
        if pending or active or score_queue or scorer:time.sleep(5)
    if failed:raise RuntimeError('Technical failure retained')
    if not stopped:write(control/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(scored),truth_last=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','dispatch','worker','score'))
    parser.add_argument('--config',required=True);parser.add_argument('--job-id')
    parser.add_argument('--inventory');parser.add_argument('--source-runtime')
    parser.add_argument('--run-root');parser.add_argument('--release-root');parser.add_argument('--profiles',nargs='+')
    parser.add_argument('--allowed-gpus',nargs='+',type=int);parser.add_argument('--prior-config',action='append',default=[]);args=parser.parse_args()
    if args.action=='prepare':
        require(all((args.inventory,args.source_runtime,args.run_root,args.release_root,args.profiles,args.allowed_gpus)),
                'Missing preparation argument')
        path=Path(args.config).resolve();require(not path.exists() and not path.parent.exists(), 'Plan already exists')
        path.parent.mkdir(parents=True,exist_ok=False)
        runtime=read(args.source_runtime);runtime.pop('clean_root',None);runtime.pop('restored_root',None)
        runtime['release_root']=str(Path(args.release_root).resolve());runtime['runtime_file']=str(path.parent/'runtime.json')
        write(path.parent/'runtime.json',runtime)
        config=make_plan(read(args.inventory),path.parent/'runtime.json',args.run_root,args.release_root,
                         tuple(args.profiles),args.allowed_gpus,args.prior_config)
        write(path,config);print(json.dumps(dict(status='PREPARED',jobs=len(config['jobs']),truth_read=False)))
        return
    config=read(args.config)
    if args.action=='dispatch':dispatch(config,args.config);return
    matches=[j for j in config['jobs'] if j['id']==args.job_id];require(len(matches)==1,'Unknown job')
    (run_worker if args.action=='worker' else run_score)(config,matches[0])


if __name__=='__main__':main()
