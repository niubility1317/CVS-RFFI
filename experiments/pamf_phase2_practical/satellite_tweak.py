"""Satellite Tweak M1, one physical query decision per inference call."""
import argparse
import json
from pathlib import Path
from .io import read,require,write
from .satellite_native_da import RECEIVERS,SEEDS
from .satellite_native_c import EPISODES,dispatch


def validate_scope(job):
    parts=job.get('capsule','').split('/')
    require(len(parts)==4 and parts[0]=='capsules' and parts[1] in ('P01','P02','P03') and
            parts[2] in EPISODES and parts[3] in RECEIVERS and job.get('seed') in SEEDS and
            job.get('profile')=='tweak_m1','Invalid Tweak M1 task identity')
    if job.get('design')=='mix3_v1':
        require(parts[2] in ('MIX01','MIX02','MIX03') and job['seed'] in SEEDS[:3], 'Outside revised comparison')
        points=[[k,5] for k in (1,5,10,20)]
    else:
        points=[[k,n] for k in (1,5,10,20) for n in (2,5,10,20)] if parts[2].startswith('MIX') else [[5,20]]
    require(job.get('points')==points,'Wrong Tweak workpoints')
    return sorted({k for k,n in points})


def verify_prediction(job):
    ks=validate_scope(job);folder=Path(job['output']);done=read(folder/'complete.json');idx=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('profile')=='tweak_m1' and done.get('model_seed')==job['seed'] and
            idx.get('capsule')==job['capsule'] and idx.get('model_seed')==job['seed'] and
            idx.get('capsule_id')==done.get('capsule_id'),'Tweak completion identity mismatch')
    expected={(stage,k,0,'S0Q0','tweak_m1_single_v1') for stage in ('A0','B') for k in ks}
    rows=idx['rows'];actual=[(v['stage'],v['k'],v['new_count'],v['input_path'],v['arm']) for v in rows]
    require(len(actual)==len(set(actual)) and set(actual)==expected and done['rows']==len(rows) and
            all(v.get('input_condition')=='satellite_received' and v.get('query_records_per_decision')==1 and
                (folder/v['prediction']).is_file() for v in rows),'Wrong Tweak satellite prediction rows')


def make_plan(runtime_file,run_root,release_root,prior_configs,wait_run):
    root=Path(run_root).resolve();release=Path(release_root).resolve();rt=read(runtime_file)
    require(not root.exists() and release.is_dir() and len(prior_configs)==3,'Wrong run or prior queues')
    require(rt.get('input_condition')=='satellite_received' and not rt.get('clean_root') and
            Path(rt['data_root']).is_dir() and Path(rt['source_contract']).is_file() and
            Path(rt['source_cache']).is_dir(),'Wrong Tweak source or satellite input')
    sources=rt['additional_sources']['tweak_m1']
    expected=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase2-external-threepool-m4-r01/source')
    require(all(Path(sources[str(seed)]).resolve()==(expected/f'tweak_m1-s{seed}').resolve() and
                (Path(sources[str(seed)])/'source.pt').is_file() for seed in SEEDS),'Wrong Tweak source provenance')
    prior=[read(p) for p in prior_configs];gpus=[2,3,4]
    require(set(prior[1]['allowed_gpus'])==set(gpus) and Path(prior[1]['run_root']).resolve()==Path(wait_run).resolve() and
            set(prior[0]['allowed_gpus']).isdisjoint(gpus) and set(prior[2]['allowed_gpus']).isdisjoint(gpus),
            'Wrong predecessor or GPU allocation')
    planned=[]
    for pool in ('P01','P02','P03'):
        for episode in EPISODES:
            for rx in RECEIVERS:
                for seed in SEEDS:
                    cap=f'capsules/{pool}/{episode}/{rx}';ident=f'tweak_m1-s{seed}-capsules-{pool}-{episode}-{rx}'
                    points=[[k,n] for k in (1,5,10,20) for n in (2,5,10,20)] if episode.startswith('MIX') else [[5,20]]
                    job=dict(id=ident,profile='tweak_m1',capsule=cap,seed=seed,points=points,
                             output=str(root/'predictions'/ident),score=str(root/'scores'/ident))
                    validate_scope(job);planned.append(job)
    require(len(planned)==1932 and len({j['id'] for j in planned})==1932,'Wrong Tweak task matrix')
    for old in prior:
        require(not any(j.get('profile')=='tweak_m1' for j in old['jobs']),'Prior queue already includes Tweak')
    return dict(entry_module='satellite_tweak',input_condition='satellite_received',run_root=str(root),
                release_root=str(release),runtime_file=str(Path(runtime_file).resolve()),wait_run=str(Path(wait_run).resolve()),
                prior_configs=list(map(str,prior_configs)),allowed_gpus=gpus,max_per_gpu=7,minimum_free_MiB=6000,
                worker_reserve_MiB=3000,jobs=planned)


def run_worker(config,job):
    from .clean_additional_worker import run
    validate_scope(job);rt=read(config['runtime_file']);require(rt.get('input_condition')=='satellite_received','Wrong input')
    run(rt,job)


def run_score(config,job):
    from .episodic_score import run
    verify_prediction(job);run(read(config['runtime_file']),dict(prediction=job['output'],output=job['score']))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','dispatch','worker','score'))
    p.add_argument('--config',required=True);p.add_argument('--job-id');p.add_argument('--source-runtime')
    p.add_argument('--run-root');p.add_argument('--release-root');p.add_argument('--wait-run')
    p.add_argument('--prior-config',action='append',default=[]);a=p.parse_args()
    if a.action=='prepare':
        require(all((a.source_runtime,a.run_root,a.release_root,a.wait_run)) and len(a.prior_config)==3,'Missing preparation inputs')
        path=Path(a.config).resolve();require(not path.exists() and not path.parent.exists(),'Plan exists')
        path.parent.mkdir(parents=True,exist_ok=False);rt=read(a.source_runtime)
        rt.pop('clean_root',None);rt.pop('restored_root',None);rt['input_condition']='satellite_received'
        rt['release_root']=str(Path(a.release_root).resolve());rt['runtime_file']=str(path.parent/'runtime.json')
        write(rt['runtime_file'],rt);config=make_plan(rt['runtime_file'],a.run_root,a.release_root,a.prior_config,a.wait_run)
        write(path,config);print(json.dumps(dict(status='PREPARED',jobs=len(config['jobs']),truth_read=False)));return
    config=read(a.config)
    if a.action=='dispatch':dispatch(config,a.config);return
    selected=[j for j in config['jobs'] if j['id']==a.job_id];require(len(selected)==1,'Unknown Tweak job')
    (run_worker if a.action=='worker' else run_score)(config,selected[0])


if __name__=='__main__':main()
