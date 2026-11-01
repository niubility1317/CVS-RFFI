"""Independent CPU score completion for already frozen native predictions.

The original owner and its output tree remain untouched. No model or IQ input
is loaded by this controller; truth is connected only in isolated score workers.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from .io import read,write,require

PROFILES=('protonet_cda','mrior_sda','dadda_sda','twostage','feature_separation_hu2024','poster_ft','radionet_ada_knn','orthogonal','lora_triplet_knn')
EXPECTED_JOBS=17397


def manifest(rt):
    profiles=rt.get('native_profiles',PROFILES)
    require(len(profiles)==len(PROFILES) and set(profiles)==set(PROFILES),'Only the nine fixed native profiles may be scored')
    parent=Path(rt['native_parent_jobs']);root=Path(rt['run_root']);raw=read(parent)
    require(isinstance(raw,list),'Parent jobs must be the original jobs list')
    selected=[j for j in raw if j['kind']=='score' and j['profile'] in PROFILES]
    require(len(selected)==EXPECTED_JOBS,'Incomplete native score job manifest')
    counts=Counter(j['profile'] for j in selected)
    require(all(counts[p]==EXPECTED_JOBS//len(PROFILES) for p in PROFILES),'Incomplete score coverage for one native profile')
    require(not root.resolve().is_relative_to(parent.parent.resolve()),'Independent output root cannot be inside the old owner root')
    result=[];keys=set();outputs=set();predictions=set()
    for old in selected:
        key=old['id'];require(re.fullmatch(r'[A-Za-z0-9_.-]+',key) and key not in keys,'Invalid or duplicate native score job ID')
        source=Path(old['output']);output=root/'scores'/old['profile']/f's{old["seed"]}'/source.name
        require(str(output) not in outputs and old['prediction'] not in predictions,'Duplicate native prediction/output identity')
        require(not root.resolve().is_relative_to(source.parents[3].resolve()),'Independent score root cannot modify an old score owner tree')
        keys.add(key);outputs.add(str(output));predictions.add(old['prediction'])
        result.append(dict(key=key,profile=old['profile'],seed=old['seed'],prediction=old['prediction'],
            output=str(output),completion=str(output/'complete.json'),parent_output=str(source),parent_completion=old['completion'],
            parent_job_id=key,original_id=old.get('original_id',key),parent_runtime_file=old.get('runtime_file'),
            parent_jobs=str(parent),execution_role='independent_score_only_completion_not_takeover'))
    return result


def bind_external_jobs(rt,jobs=None):
    return [dict(parent_prediction=j['prediction'],new_score_output=j['output'],new_complete=j['completion'],
                 profile=j['profile'],seed=j['seed'],parent_score_output=j['parent_output'],parent_job_id=j['parent_job_id'])
            for j in (manifest(rt) if jobs is None else jobs)]


def validate_prediction(rt,job):
    directory=Path(job['prediction']);done=read(directory/'complete.json');index=read(directory/'predictions.json')
    require(done['status']=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and done.get('query_fit') is False,'Prediction must be frozen before score recovery')
    require(index['model_seed']==job['seed'] and index['capsule_id']==done['capsule_id'],'Prediction seed/capsule binding differs')
    require(index.get('profile',job['profile'])==job['profile'],'Wrong native prediction profile')
    cap=read(Path(rt['data_root'])/index['capsule']/'manifest.json')
    require(cap['protocol_schema']=='p2_min_v1' and cap['phase2_data_status']=='VALIDATED_ONCE' and cap['capsule_id']==index['capsule_id'],'Wrong validated data handle')
    require(index['rows'] and done['rows']==len(index['rows']),'Incomplete prediction index')
    keys=set();files=set()
    for row in index['rows']:
        key=(row['split_id'],row['stage'],row['arm'],row['input_path'])
        require(key not in keys and row['prediction'] not in files,'Duplicate native prediction row')
        require(row['arm']==job['profile'] and row['input_path']=='P0' and row['stage'] in ('A0','B','C'),'Only frozen native raw prediction rows are allowed')
        require((directory/row['prediction']).is_file(),'Missing frozen prediction file')
        keys.add(key);files.add(row['prediction'])
    return index


def validate_metrics(directory,index):
    directory=Path(directory);done=read(directory/'complete.json');payload=read(directory/'metrics.json')
    require(done['status']=='SCORED_COMPLETE' and payload.get('truth_last') is True and payload.get('target_feedback') is False,'Score result is not independently complete')
    require(payload['model_seed']==index['model_seed'] and payload['capsule_id']==index['capsule_id'],'Old metrics seed/capsule binding differs')
    require(done['rows']==len(index['rows'])==len(payload['rows']),'Score row count differs')
    expected={r['prediction']:r for r in index['rows']};seen=set()
    for row in payload['rows']:
        name=row['prediction'];require(name in expected and name not in seen,'Missing/duplicate/unbound score row')
        reference=expected[name]
        require(all(row[k]==reference[k] for k in ('split_id','registered_classes','k','new_count','stage','arm','input_path')),'Score/prediction row binding differs')
        require(row.get('model_seed')==index['model_seed'],'Wrong scored row model seed')
        require(isinstance(row.get('metrics'),dict),'Score row metrics missing')
        seen.add(name)
    require(seen==set(expected),'Incomplete native score rows')
    return payload


def _process(pid):
    folder=Path('/proc')/str(pid)
    try:
        if (folder/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return None
        return dict(pid=int(pid),uid=folder.stat().st_uid,cwd=os.readlink(folder/'cwd'),
                    argv=(folder/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def matches_process(job,process):
    argv=process['argv'];ids={job['parent_job_id'],job['original_id']}
    if '--job' not in argv:return False
    at=argv.index('--job')+1
    if at>=len(argv) or argv[at] not in ids:return False
    if job.get('parent_runtime_file'):return job['parent_runtime_file'] in argv
    # Older manifests may omit runtime_file; read only the actual command's
    # runtime metadata to bind its score owner root to the original output.
    if '--runtime' not in argv:return False
    at=argv.index('--runtime')+1
    if at>=len(argv) or not Path(argv[at]).is_file():return False
    return read(argv[at]).get('run_root')==str(Path(job['parent_output']).parents[3])


def active_scorers(job):
    require(os.name!='nt' and Path('/proc').is_dir(),'Actual old-score process matching requires the N607 /proc environment')
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        process=_process(int(path.name))
        if process and process['uid']==os.getuid() and matches_process(job,process):result.append(process)
    return result


def _old_complete(job):
    path=Path(job['parent_completion'])
    return path.is_file() and read(path).get('status')=='SCORED_COMPLETE'


def worker(rt,job):
    output=Path(job['output']);require(not output.exists(),'Independent score output collision; never overwrite')
    index=validate_prediction(rt,job)
    while True:
        if _old_complete(job):
            validate_metrics(job['parent_output'],index)
            output.mkdir(parents=True,exist_ok=False)
            for name in ('metrics.json','complete.json'):os.link(Path(job['parent_output'])/name,output/name)
            write(output/'reuse.json',dict(status='REUSED_COMPLETE_SCORE',reused=True,parent_job_id=job['parent_job_id'],
                parent_prediction=job['prediction'],parent_score_output=job['parent_output'],parent_owner_unchanged=True))
            return 'reused'
        processes=active_scorers(job)
        if not processes:
            if _old_complete(job):continue
            break
        wait=dict(status='WAITING_FOR_ORIGINAL_SCORER',job=job['key'],prediction=job['prediction'],processes=processes,observed_at=time.time())
        write(Path(rt['run_root'])/'waits'/(job['key']+'.json'),wait,exclusive=False)
        print('NATIVE_SCORE_WAIT',job['key'],processes,flush=True);time.sleep(rt.get('poll_seconds',5))
    # Only this isolated subprocess imports the truth-side scorer.
    from .episodic_score import run
    validate_prediction(rt,job)
    run(rt,dict(prediction=job['prediction'],output=job['output']))
    validate_metrics(output,index)
    write(output/'reuse.json',dict(status='NEW_INDEPENDENT_SCORE',reused=False,parent_job_id=job['parent_job_id'],
        parent_prediction=job['prediction'],parent_score_output=job['parent_output'],parent_owner_unchanged=True))
    return 'new'


def dispatch(rt):
    jobs=manifest(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    owner=dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv,role='independent_CPU_score_completion',parent_jobs=rt['native_parent_jobs'],parent_owner_unchanged=True)
    write(root/'owner.json',owner);write(root/'jobs.json',jobs);write(root/'external_score_overrides.json',bind_external_jobs(rt,jobs))
    print('NATIVE_SCORE_OWNER',owner,flush=True);completed=reused=new=0;current=None
    try:
        # Validate existing fixed prediction handles, without opening prediction
        # arrays, IQ, source weights, or any truth-side file in this controller.
        write(root/'score_state.json',dict(status='VALIDATING_FIXED_PREDICTIONS',completed=0,total=len(jobs),validated_predictions=0,
              reused_jobs=0,new_score_jobs=0,active=None),exclusive=False)
        for number,job in enumerate(jobs,1):
            validate_prediction(rt,job)
            if number%1000==0 or number==len(jobs):
                state=dict(status='VALIDATING_FIXED_PREDICTIONS',completed=0,total=len(jobs),validated_predictions=number,
                           reused_jobs=0,new_score_jobs=0,active=None)
                write(root/'score_state.json',state,exclusive=False);print('NATIVE_SCORE_PREFLIGHT',state,flush=True)
        for job in jobs:
            current=job['key'];cmd=[sys.executable,'-u','-m',__package__+'.native_score_recovery','worker','--runtime',rt['runtime_file'],'--key',current]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (root/'logs'/(current+'.log')).open('x',encoding='utf-8') as log:
                process=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state=dict(status='RUNNING',completed=completed,total=len(jobs),reused_jobs=reused,new_score_jobs=new,
                    active=dict(key=current,pid=process.pid,cwd=str(Path.cwd()),argv=cmd),parent_owner_unchanged=True)
                write(root/'score_state.json',state,exclusive=False);print('NATIVE_SCORE_PROGRESS',state,flush=True)
                code=process.wait()
            require(code==0,'Independent score worker failed; stop without retry: '+current)
            result=read(Path(job['output'])/'reuse.json');require(read(job['completion'])['status']=='SCORED_COMPLETE','Score worker did not close')
            reused+=result['reused'];new+=not result['reused'];completed+=1
            write(root/'score_state.json',dict(status='RUNNING',completed=completed,total=len(jobs),reused_jobs=reused,new_score_jobs=new,active=None),exclusive=False)
        done=dict(status='SCORED_COMPLETE',completed=completed,total=len(jobs),reused_jobs=reused,new_score_jobs=new,
                  independent_CPU_scorers=1,training_jobs=0,prediction_jobs=0,parent_owner_unchanged=True)
        write(root/'complete.json',done);write(root/'score_state.json',done,exclusive=False);print('NATIVE_SCORE_COMPLETE',done,flush=True)
    except Exception as exc:
        failure=dict(status='FAILED',job=current,error=str(exc),completed=completed,total=len(jobs),reused_jobs=reused,new_score_jobs=new,
                     retries=0,parent_owner_unchanged=True)
        write(root/'score_state.json',failure,exclusive=False);write(root/'complete.json',failure);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dispatch','worker']);parser.add_argument('--runtime',required=True);parser.add_argument('--key');args=parser.parse_args();rt=read(args.runtime)
    if args.action=='dispatch':dispatch(rt)
    else:
        jobs=read(Path(rt['run_root'])/'jobs.json');found=[j for j in jobs if j['key']==args.key]
        require(len(found)==1,'Unknown independent score job');worker(rt,found[0])
