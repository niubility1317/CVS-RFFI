"""Paired short-budget learning-rate diagnostic on unused GPUs."""
import argparse
from pathlib import Path

from .clean_scheduler import completion_matches, validate_plan
from .io import read, require, write

ARMS=((50,25,3e-5),(50,25,1e-4),(100,50,3e-5),(100,50,1e-4))


def prepare(original_config, plan_root, run_root, release_root, gpus):
    original=read(original_config)
    plan_root,run_root,release_root=(Path(p).resolve() for p in (plan_root,run_root,release_root))
    require(not plan_root.exists() and not run_root.exists(),'Output already exists')
    require(gpus and len(gpus)==len(set(gpus)) and all(type(g)==int and 0<=g<=7 for g in gpus),'Invalid GPUs')
    originals=[]
    for entry in original['jobs']:
        job=read(entry['job'])
        if entry['group']=='native_cil' and job.get('profile')=='issl_fixed_v1':
            require(job['seed']==2026092701 and job['points']==[[5,20]] and completion_matches(entry),'Unfixed reference')
            originals.append((entry,job))
    require(len(originals)==5 and len({job['capsule'] for _,job in originals})==5,'Expected five paired capsules')
    plan_root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for ssl,transfer,lr in ARMS:
        for entry,job in originals:
            key=f"issl-ssl{ssl}-transfer{transfer}-lr{int(lr*1000000)}e6-{entry['id'].rsplit('-',1)[-1]}"
            output=run_root/'tasks'/key
            runtime=read(entry['runtime']);runtime_file=plan_root/'runtime'/(key+'.json');job_file=plan_root/'jobs'/(key+'.json')
            runtime.update(release_root=str(release_root),runtime_file=str(runtime_file))
            write(runtime_file,runtime)
            write(job_file,dict(job,output=str(output),issl_budget=[ssl,transfer],issl_ssl_lr=lr))
            jobs.append(dict(id=key,group='native_cil',runtime=str(runtime_file),job=str(job_file),
                             completion=str(output/'complete.json'),priority=ssl))
    config=dict(original,run_root=str(run_root),release_root=str(release_root),jobs=jobs,allowed_gpus=gpus,max_per_gpu=2)
    config['auto_score']=dict(preparation=str(plan_root/'preparation.json'),config=str(plan_root/'scheduler.json'),
                              output=str(run_root/'scores'),workers=1)
    validate_plan(config)
    write(plan_root/'scheduler.json',config)
    write(plan_root/'preparation.json',dict(jobs=len(jobs),blocked=[],arms=[list(a) for a in ARMS],
          scope='five_paired_clean_capsules_one_seed_K5_N20',target_test_previously_exposed=True))
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('original-config','plan-root','run-root','release-root'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--gpu',type=int,action='append',required=True)
    a=p.parse_args()
    print('PREPARED',len(prepare(a.original_config,a.plan_root,a.run_root,a.release_root,a.gpu)['jobs']))


if __name__=='__main__':main()
