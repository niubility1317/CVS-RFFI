"""Write a bounded, immutable clean pilot queue without launching workers."""
import argparse
from pathlib import Path

from .clean_comparison_plan import tasks
from .io import read,require,write


def prepare(plan_root,run_root,release_root,clean_root,pamf_template,group1_template,
            additional_template,source_root,scalr_root,*,capsules,seed,allowed_gpus,restored_root=None):
    plan_root=Path(plan_root).resolve();run_root=Path(run_root).resolve();release_root=Path(release_root).resolve()
    require(seed in (2026092701,2026092702,2026092703,2026092704),'Unknown seed')
    require(capsules and len(capsules)<=5 and len(set(capsules))==len(capsules),'Choose at most five distinct pilot capsules')
    allowed={t['capsule'] for t in tasks() if t['seed']==seed}
    require(set(capsules)<=allowed,'Pilot capsule is outside fixed matrix')
    require(allowed_gpus and len(set(allowed_gpus))==len(allowed_gpus) and all(type(g) is int and 0<=g<=7 for g in allowed_gpus),'Wrong GPU list')
    require(not run_root.exists(),'Run output root already exists')
    pamf=read(pamf_template);native=read(group1_template);additional=read(additional_template)
    plan_root.mkdir(parents=True,exist_ok=False);(plan_root/'runtime').mkdir();(plan_root/'jobs').mkdir()
    native_file=plan_root/'runtime/native.json'
    native.update(data_root=pamf['data_root'],clean_root=str(clean_root),run_root=str(source_root),source_run_root=str(source_root),
                  release_root=str(release_root),runtime_file=str(native_file),additional_sources=additional['additional_sources'])
    write(native_file,native)
    jobs=[];blocked=[]
    b_profiles=('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft')
    def add(key,group,rt_file,job,out):
        job_file=plan_root/'jobs'/(key+'.json');write(job_file,job)
        jobs.append(dict(id=key,group=group,runtime=str(rt_file),job=str(job_file),
                         completion=str(out/'complete.json'),priority=0))
    for i,cap in enumerate(capsules):
        common=dict(seed=seed,capsule=cap,points=[[5,20]])
        key=f'pamf-{i:02d}';task_root=run_root/'tasks'/key
        rtf=plan_root/'runtime'/(key+'.json')
        prt=dict(pamf,clean_root=str(clean_root),run_root=str(task_root),model_seeds=[seed],
                 release_root=str(release_root),runtime_file=str(rtf),scalr_source_root=str(Path(scalr_root)/f's{seed}'))
        if restored_root:prt['restored_root']=str(restored_root)
        else:prt.pop('restored_root',None)
        write(rtf,prt)
        out=task_root/'predictions'/cap.split('/')[-1]
        add(key,'pamf',rtf,dict(common),out)
        for profile in (*b_profiles,'tweak_m1'):
            key=f'da-{profile}-{i:02d}';out=run_root/'tasks'/key
            add(key,'native_da',native_file,dict(common,profile=profile,output=str(out)),out)
        for profile in ('lora_triplet_knn','issl_fixed_v1'):
            key=f'cil-{profile}-{i:02d}';out=run_root/'tasks'/key
            add(key,'native_cil',native_file,dict(common,profile=profile,output=str(out)),out)
        for profile in b_profiles:
            key=f'hybrid-{profile}-{i:02d}';out=run_root/'tasks'/key
            add(key,'hybrid',native_file,dict(common,profile=profile,methods=['orthogonal','lora'],output=str(out)),out)
        key=f'hybrid-tweak_m1-{i:02d}';out=run_root/'tasks'/key
        add(key,'hybrid',native_file,dict(common,profile='tweak_m1',methods=['lora'],output=str(out)),out)
        blocked.extend([dict(capsule=cap,profile='orthogonal_fixed',reason='matching repaired source weights unavailable'),
                        dict(capsule=cap,profile='mopc_hr_public_code',reason='genuine nonfinite failure unresolved'),
                        dict(capsule=cap,profile='tweak_m1_to_mopc_or_orthogonal',reason='compatible state bridge unverified')])
    queue=dict(phase='pilot',run_root=str(run_root),release_root=str(release_root),jobs=jobs,allowed_gpus=allowed_gpus,
        max_per_gpu=2,minimum_free_MiB=6000,poll_seconds=5,input_condition='clean_ground_pre_satellite_normalized')
    write(plan_root/'scheduler.json',queue)
    summary=dict(status='PARTIAL_PREPARED_NOT_STARTED',jobs=len(jobs),blocked=blocked,capsules=capsules,seed=seed,
                 restoration_cache_configured=bool(restored_root),truth_read=False,formal_experiment_started=False,
                 global_prediction_barrier=False)
    write(plan_root/'preparation.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan-root','run-root','release-root','clean-root','pamf-template','group1-template',
                 'additional-template','source-root','scalr-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--capsule',action='append',required=True);p.add_argument('--seed',type=int,default=2026092701)
    p.add_argument('--gpu',action='append',type=int,required=True);p.add_argument('--restored-root')
    a=p.parse_args()
    result=prepare(a.plan_root,a.run_root,a.release_root,a.clean_root,a.pamf_template,a.group1_template,
        a.additional_template,a.source_root,a.scalr_root,capsules=a.capsule,seed=a.seed,
        allowed_gpus=a.gpu,restored_root=a.restored_root)
    print(result)


if __name__=='__main__':main()
