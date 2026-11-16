"""Long-budget native ISSL on satellite; each K/N scores independently."""
import argparse
import json
from pathlib import Path
from .io import read,require,write
from .satellite_native_da import RECEIVERS,SEEDS
from .satellite_native_c import EPISODES,dispatch


def validate_scope(job):
    parts=job.get('capsule','').split('/')
    require(len(parts)==4 and parts[0]=='capsules' and parts[1] in ('P01','P02','P03') and parts[2] in EPISODES and
            parts[3] in RECEIVERS and job.get('seed') in SEEDS and job.get('profile')=='issl_fixed_v1',
            'Invalid ISSL task identity')
    points=job.get('points');require(isinstance(points,list) and len(points)==1 and isinstance(points[0],list) and
        len(points[0])==2,'One ISSL K/N per task required')
    k,n=points[0];mix=parts[2].startswith('MIX')
    require((mix and k in (1,5,10,20) and n in (2,5,10,20)) or (not mix and (k,n)==(5,20)),
            'Wrong ISSL K/N scope')
    require(job.get('skip_a0') is (mix and n!=2),'A0 must execute once per K')
    return k,n


def verify_prediction(job):
    k,n=validate_scope(job);folder=Path(job['output']);done=read(folder/'complete.json');idx=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and done.get('query_fit') is False and
            done.get('profile')=='issl_fixed_v1' and done.get('model_seed')==job['seed'] and idx.get('capsule')==job['capsule'] and
            idx.get('model_seed')==job['seed'] and idx.get('capsule_id')==done.get('capsule_id'),'ISSL completion identity mismatch')
    arm='issl_fixed_v1_long_v1';expected={('C',k,n,'S0Q0',arm)}
    if not job['skip_a0']:expected.add(('A0',k,0,'S0Q0',arm))
    rows=idx['rows'];actual=[(v['stage'],v['k'],v['new_count'],v['input_path'],v['arm']) for v in rows]
    require(len(actual)==len(set(actual)) and set(actual)==expected and done['rows']==len(rows) and
            all(v.get('input_condition')=='satellite_received' and (folder/v['prediction']).is_file() for v in rows),
            'Wrong ISSL satellite prediction rows')


def make_plan(runtime_file,run_root,release_root,prior_configs,wait_runs):
    root=Path(run_root).resolve();release=Path(release_root).resolve();rt=read(runtime_file)
    require(not root.exists() and release.is_dir() and len(prior_configs)==4 and len(wait_runs)==2,'Wrong run or predecessor list')
    require(rt.get('input_condition')=='satellite_received' and not rt.get('clean_root') and
            Path(rt['data_root']).is_dir() and Path(rt['source_contract']).is_file() and Path(rt['source_cache']).is_dir(),
            'Wrong ISSL source or satellite input')
    sources=rt['additional_sources']['issl_fixed_v1']
    expected=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase2-external-threepool-m4-r01/source')
    require(all(Path(sources[str(seed)]).resolve()==(expected/f'issl_fixed_v1-s{seed}').resolve() and
                (Path(sources[str(seed)])/'source.pt').is_file() for seed in SEEDS),'Wrong ISSL source provenance')
    prior=[read(p) for p in prior_configs];gpus=[0,1,5,6,7]
    require(set(prior[2]['allowed_gpus'])==set(gpus) and set(prior[3]['allowed_gpus']).isdisjoint(gpus) and
            Path(prior[2]['run_root']).resolve()==Path(wait_runs[0]).resolve() and
            Path(prior[3]['run_root']).resolve()==Path(wait_runs[1]).resolve(),'Wrong ISSL predecessor/GPU allocation')
    require(not any(j.get('profile')=='issl_fixed_v1' for p in prior for j in p['jobs']),'Prior queue contains ISSL')
    planned=[]
    for pool in ('P01','P02','P03'):
        for episode in EPISODES:
            for rx in RECEIVERS:
                for seed in SEEDS:
                    cap=f'capsules/{pool}/{episode}/{rx}'
                    points=[(k,n) for k in (1,5,10,20) for n in (2,5,10,20)] if episode.startswith('MIX') else [(5,20)]
                    for k,n in points:
                        ident=f'issl_fixed_v1-s{seed}-capsules-{pool}-{episode}-{rx}-K{k}-N{n}'
                        job=dict(id=ident,profile='issl_fixed_v1',capsule=cap,seed=seed,points=[[k,n]],
                                 skip_a0=episode.startswith('MIX') and n!=2,
                                 output=str(root/'predictions'/ident),score=str(root/'scores'/ident))
                        validate_scope(job);planned.append(job)
    require(len(planned)==12012 and len({j['id'] for j in planned})==12012,'Wrong ISSL task matrix')
    return dict(entry_module='satellite_issl',input_condition='satellite_received',run_root=str(root),
                release_root=str(release),runtime_file=str(Path(runtime_file).resolve()),wait_runs=list(map(str,wait_runs)),
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
    p.add_argument('--run-root');p.add_argument('--release-root');p.add_argument('--wait-run',action='append',default=[])
    p.add_argument('--prior-config',action='append',default=[]);a=p.parse_args()
    if a.action=='prepare':
        require(all((a.source_runtime,a.run_root,a.release_root)) and len(a.wait_run)==2 and len(a.prior_config)==4,
                'Missing ISSL preparation inputs')
        path=Path(a.config).resolve();require(not path.exists() and not path.parent.exists(),'Plan exists')
        path.parent.mkdir(parents=True,exist_ok=False);rt=read(a.source_runtime)
        rt.pop('clean_root',None);rt.pop('restored_root',None);rt['input_condition']='satellite_received'
        rt['release_root']=str(Path(a.release_root).resolve());rt['runtime_file']=str(path.parent/'runtime.json')
        write(rt['runtime_file'],rt);config=make_plan(rt['runtime_file'],a.run_root,a.release_root,a.prior_config,a.wait_run)
        write(path,config);print(json.dumps(dict(status='PREPARED',jobs=len(config['jobs']),truth_read=False)));return
    config=read(a.config)
    if a.action=='dispatch':dispatch(config,a.config);return
    selected=[j for j in config['jobs'] if j['id']==a.job_id];require(len(selected)==1,'Unknown ISSL job')
    (run_worker if a.action=='worker' else run_score)(config,selected[0])


if __name__=='__main__':main()
