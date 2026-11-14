"""Prepare paired short-budget ISSL jobs without launching them."""
import argparse
from pathlib import Path

from .clean_scheduler import completion_matches, validate_plan
from .io import read, require, write

BUDGETS=((50,25),(100,50),(200,100),(400,100))


def prepare(original_config, plan_root, run_root, release_root, gpus):
    original=read(original_config)
    plan_root=Path(plan_root).resolve()
    run_root=Path(run_root).resolve()
    release_root=Path(release_root).resolve()
    require(not plan_root.exists() and not run_root.exists(),'Output already exists')
    require(len(gpus)==2 and len(set(gpus))==2 and all(type(g)==int and 0<=g<=7 for g in gpus),
            'Two explicit GPUs required')
    originals=[]
    for entry in original['jobs']:
        job=read(entry['job'])
        if entry['group']=='native_cil' and job.get('profile')=='issl_fixed_v1':
            require(job['seed']==2026092701 and job['points']==[[5,20]] and completion_matches(entry),
                    'Long-budget paired prediction incomplete')
            originals.append((entry,job))
    require(len(originals)==5 and len({j['capsule'] for _,j in originals})==5,'Five completed paired tasks required')
    plan_root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for ssl,transfer in BUDGETS:
        for entry,job in originals:
            key=f"issl-ssl{ssl}-transfer{transfer}-{entry['id'].rsplit('-',1)[-1]}"
            output=run_root/'tasks'/key
            runtime=read(entry['runtime'])
            runtime_file=plan_root/'runtime'/(key+'.json')
            job_file=plan_root/'jobs'/(key+'.json')
            runtime.update(release_root=str(release_root),runtime_file=str(runtime_file))
            write(runtime_file,runtime)
            write(job_file,dict(job,output=str(output),issl_budget=[ssl,transfer]))
            jobs.append(dict(id=key,group='native_cil',runtime=str(runtime_file),
                             job=str(job_file),completion=str(output/'complete.json'),priority=ssl))
    config=dict(original,run_root=str(run_root),release_root=str(release_root),jobs=jobs,
                allowed_gpus=gpus,max_per_gpu=2)
    validate_plan(config)
    write(plan_root/'scheduler.json',config)
    write(plan_root/'preparation.json',dict(jobs=len(jobs),blocked=[],budgets=[list(v) for v in BUDGETS],
          reference_config=str(Path(original_config).resolve()),reference_budget=[2000,300],
          scope='five_paired_clean_capsules_one_model_seed_K5_N20',target_test_previously_exposed=True))
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('original-config','plan-root','run-root','release-root'):
        parser.add_argument('--'+arg,required=True)
    parser.add_argument('--gpu',type=int,action='append',required=True)
    args=parser.parse_args()
    print('PREPARED',len(prepare(args.original_config,args.plan_root,args.run_root,args.release_root,args.gpu)['jobs']))


if __name__=='__main__':main()
