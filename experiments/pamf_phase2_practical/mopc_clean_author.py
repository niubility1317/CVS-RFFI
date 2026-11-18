"""Clean MoPC author-version prediction and independent per-task scoring."""
import argparse
from collections import Counter
from pathlib import Path, PurePosixPath

from .io import read, require, write
from .satellite_native_da import RECEIVERS

POINTS = [[5,5],[5,10],[10,5],[10,10]]
PROFILES = ('mopc_hr','protonet_cda','mrior_sda','twostage','poster_ft','tweak_m1')


def new_job(old,run_root):
    profile=old['profile'];cap=old['capsule'];seed=old['seed']
    ident=f"{'native' if profile=='mopc_hr' else 'hybrid'}-{profile}-s{seed}-{'-'.join(cap.split('/'))}"
    job=dict(old,id=ident,key=ident,points=POINTS,output=str(PurePosixPath(run_root)/'predictions'/ident),
             score=str(PurePosixPath(run_root)/'scores'/ident),mopc_sampling_version='author_drop_last_v1',
             mopc_stabilization='none',source_smoke_in_dispatch=True)
    for key in ('reuse','reuse_a0','reuse_saved_b','reuse_original_b','reuse_author_b'):
        job.pop(key,None)
    if profile=='mopc_hr':job['clean_author_c_only']=True
    else:
        require(job.get('methods')==['mopc_hr'] and job.get('save_B_for_mopc') is True,'Wrong clean hybrid')
        if profile=='protonet_cda':job['mopc_bridge']='proto_affine_v1'
    validate_scope(job)
    return job


def partition(jobs,part):
    require(part in ('pilot','remaining'),'Wrong clean partition')
    def is_pilot(j):
        return j['capsule']=='capsules/P01/MIX01/1-1' and j['seed']==2026092701
    return [j for j in jobs if is_pilot(j)==(part=='pilot')]


def make_plan(previous,run_root,release_root,runtimes,wait_run,part):
    require(PurePosixPath(run_root).is_absolute() and not Path(run_root).exists() and
            PurePosixPath(release_root).is_absolute() and PurePosixPath(wait_run).is_absolute(), 'Wrong clean run paths')
    selected=[]
    for entry in previous['jobs']:
        old=read(entry['job']);parts=old.get('capsule','').split('/')
        if old.get('profile') not in PROFILES or old.get('seed') not in (2026092701,2026092702,2026092703) or \
           len(parts)!=4 or parts[2] not in ('MIX01','MIX02','MIX03'):
            continue
        selected.append(new_job(old,run_root))
    require(Counter(j['profile'] for j in selected)==Counter({p:189 for p in PROFILES}) and
            len({j['id'] for j in selected})==1134, 'Incomplete clean matrix')
    selected=partition(selected,part)
    require(len(selected)==(6 if part=='pilot' else 1128),'Wrong clean partition size')
    config=dict(entry_module='mopc_clean_author',input_condition='clean_ground_pre_satellite_normalized',
                run_root=run_root,release_root=release_root,runtimes=runtimes,wait_run=wait_run,
                allowed_gpus=list(range(8)),max_per_gpu=4,minimum_free_MiB=6000,worker_reserve_MiB=3000,
                allow_shared_gpu=True,author_source_smoke=True,jobs=selected)
    if part=='remaining':config['require_author_pilot_scored']=True
    return config


def validate_scope(job):
    parts=job.get('capsule','').split('/')
    require(len(parts)==4 and parts[0]=='capsules' and parts[1] in ('P01','P02','P03') and
            parts[2] in ('MIX01','MIX02','MIX03') and parts[3] in RECEIVERS and
            job.get('profile') in PROFILES and job.get('seed') in (2026092701,2026092702,2026092703) and
            job.get('points')==POINTS and job.get('mopc_sampling_version')=='author_drop_last_v1' and
            job.get('mopc_stabilization')=='none', 'Invalid clean MoPC author scope')


def verify_prediction(job):
    from .mopc_identity import require_author_mopc_row, require_author_sampling
    from .group2_worker import method_arm
    validate_scope(job)
    folder=Path(job['output']); done=read(folder/'complete.json'); index=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('profile')==job['profile'] and
            done.get('model_seed')==job['seed'] and index.get('capsule')==job['capsule'] and
            index.get('capsule_id')==done.get('capsule_id') and index.get('model_seed')==job['seed'],
            'Wrong clean task identity')
    arm='mopc_hr' if job['profile']=='mopc_hr' else method_arm(job['profile'],'mopc_hr',job.get('mopc_bridge','original'),'none')
    rows=index['rows']; actual=[(r['k'],r['new_count'],r['arm']) for r in rows]
    require(len(rows)==done.get('rows')==4 and set(actual)=={(k,n,arm) for k,n in POINTS} and
            all(r.get('stage')=='C' and r.get('input_condition')=='clean_ground_pre_satellite_normalized' and
                (folder/r['prediction']).is_file() for r in rows), 'Wrong clean prediction matrix')
    for r in rows:require_author_mopc_row(r);require_author_sampling(r)
    if job['profile']!='mopc_hr':
        require(all((folder/f'B-K{k}.pt').is_file() for k in (5,10)), 'Missing clean B state')
    if job['profile']=='tweak_m1':
        require(all(r.get('query_records_per_decision')==1 for r in rows), 'Tweak must decide one query at a time')


def run_worker(config,job):
    from .vendor_native.comparison_suite.registration import MoPCNumericalError, MoPCSamplingError
    validate_scope(job);rt=read(config['runtimes']['mopc_hr' if job['profile']=='mopc_hr' else 'hybrid'])
    require(rt.get('clean_root') and rt.get('input_condition')!='satellite_received', 'Clean input unavailable')
    try:
        if job['profile']=='mopc_hr':
            from .group1_target import predict
            predict(rt,job)
        elif job['profile']=='tweak_m1':
            from .clean_tweak_lora_worker import run
            run(rt,job)
        else:
            from .group2_worker import predict
            predict(rt,job)
    except (MoPCNumericalError,MoPCSamplingError) as exc:
        if Path(job['output']).is_dir():
            write(Path(job['output'])/'numerical_failure.json',dict(
                status='MOPC_SAMPLING_FAILURE' if isinstance(exc,MoPCSamplingError) else 'MOPC_NUMERICAL_FAILURE',
                profile=job['profile'],capsule=job['capsule'],seed=job['seed'],error=str(exc),truth_read=False,query_fit=False))
        raise


def run_score(config,job):
    from .episodic_score import run
    verify_prediction(job)
    run(read(config['runtimes']['mopc_hr' if job['profile']=='mopc_hr' else 'hybrid']),
        dict(prediction=job['output'],output=job['score']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','dispatch','worker','score'))
    parser.add_argument('--config',required=True);parser.add_argument('--job-id')
    parser.add_argument('--previous');parser.add_argument('--native-runtime');parser.add_argument('--hybrid-runtime')
    parser.add_argument('--native-source-root');parser.add_argument('--code-commit');parser.add_argument('--run-root')
    parser.add_argument('--release-root');parser.add_argument('--wait-run');parser.add_argument('--part',choices=('pilot','remaining'))
    args=parser.parse_args()
    if args.action=='prepare':
        from .mopc_author_satellite_restart import native_runtime_for_derived_source
        require(all((args.previous,args.native_runtime,args.hybrid_runtime,args.native_source_root,args.code_commit,
                     args.run_root,args.release_root,args.wait_run,args.part)),'Missing clean preparation inputs')
        path=Path(args.config);require(not path.exists() and not path.parent.exists(),'Clean preparation exists')
        native=native_runtime_for_derived_source(read(args.native_runtime),args.native_source_root,args.release_root,args.code_commit)
        hybrid=read(args.hybrid_runtime);hybrid.update(release_root=args.release_root,code_commit=args.code_commit)
        require(native.get('clean_root') and hybrid.get('clean_root') and
                native['clean_root']==hybrid['clean_root'] and native['data_root']==hybrid['data_root'], 'Wrong clean data binding')
        runtimes=dict(mopc_hr=str(path.parent/'native-runtime.json'),hybrid=str(path.parent/'hybrid-runtime.json'))
        config=make_plan(read(args.previous),args.run_root,args.release_root,runtimes,args.wait_run,args.part)
        path.parent.mkdir(parents=True,exist_ok=False);write(runtimes['mopc_hr'],native);write(runtimes['hybrid'],hybrid);write(path,config)
        print('PREPARED_CLEAN_AUTHOR',args.part,len(config['jobs']),flush=True);return
    config=read(args.config)
    if args.action=='dispatch':
        from .satellite_native_c import dispatch
        dispatch(config,args.config);return
    jobs=[j for j in config['jobs'] if j['id']==args.job_id];require(len(jobs)==1,'Unknown clean job')
    (run_worker if args.action=='worker' else run_score)(config,jobs[0])


if __name__=='__main__':main()
