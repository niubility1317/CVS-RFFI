"""Prepare the non-overlapping, currently executable clean comparison jobs."""
import argparse
from pathlib import Path

from .clean_comparison_plan import tasks
from .clean_scheduler import completion_matches
from .clean_score import expected_rows, key
from .io import read, require, write

DA=('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft','tweak_m1')
CIL=('lora_triplet_knn','issl_fixed_v1')


def prior_pilot_identities(config_paths):
    identities=set();covered=set()
    for config_path in config_paths:
        config=read(config_path)
        for entry in config['jobs']:
            require(completion_matches(entry),'Previous pilot is not fully fixed')
            job=read(entry['job'])
            payload=read(Path(entry['completion']).parent/'predictions.json')
            require(payload['capsule']==job['capsule'] and payload['model_seed']==job['seed'],
                    'Previous pilot identity mismatch')
            actual=[key(row) for row in payload['rows']]
            require(len(actual)==len(set(actual)) and set(actual)==expected_rows(entry['group'],job),
                    'Previous pilot rows incomplete')
            require(all((Path(entry['completion']).parent/row['prediction']).is_file()
                        and row['input_condition']=='clean_ground_pre_satellite_normalized'
                        for row in payload['rows']),'Previous pilot prediction missing or wrong input')
            identities.add((job['capsule'],job['seed']))
            identity=(entry['group'],job.get('profile','R42'),job['capsule'],job['seed'])
            require(identity not in covered,'Duplicate previous pilot task')
            covered.add(identity)
    require(len(identities)==5,'Expected five previously fixed pilot identities')
    expected={(group,profile,capsule,seed)
              for capsule,seed in identities
              for group,profiles in (('pamf',('R42',)),('native_da',DA),
                                     ('native_cil',CIL),('hybrid',DA))
              for profile in profiles}
    require(covered==expected,'Previously fixed pilot is incomplete')
    return identities


def prepare(plan_root, run_root, release_root, template_root, prior_configs, *, gpus):
    plan_root=Path(plan_root).resolve();run_root=Path(run_root).resolve();release_root=Path(release_root).resolve()
    require(not plan_root.exists() and not run_root.exists(),'Preparation or run output exists')
    require(len(gpus)==3 and len(set(gpus))==3 and all(type(g) is int and 0<=g<=7 for g in gpus),
            'Three distinct GPUs required')
    prior=prior_pilot_identities(prior_configs)
    template_root=Path(template_root)
    native=read(template_root/'runtime/native.json')
    pamf=read(template_root/'runtime/pamf-00.json')
    source_selection=Path(pamf['scalr_source_root']).parent
    require(read(Path(pamf['restored_root'])/'completion.json')['status']=='CLEAN_DURA_COMPLETE',
            'DURA clean bank incomplete')
    plan_root.mkdir(parents=True);(plan_root/'jobs').mkdir();(plan_root/'runtime').mkdir()
    native_file=plan_root/'runtime/native.json'
    native.update(release_root=str(release_root),runtime_file=str(native_file))
    write(native_file,native)
    jobs=[];excluded=[]
    for i,task in enumerate(tasks()):
        if (task['capsule'],task['seed']) in prior:
            excluded.append(dict(capsule=task['capsule'],seed=task['seed'],reason='pilot_fixed_points_need_row_level_continuation'))
            continue
        common=dict(capsule=task['capsule'],seed=task['seed'],points=task['points'])
        for group,profiles in (('native_da',DA),('native_cil',CIL),('pamf',('R42',))):
            for profile in profiles:
                name=f'{i:04d}-{group}-{profile}';output=run_root/'tasks'/name
                jobfile=plan_root/'jobs'/(name+'.json')
                if group=='pamf':
                    runtime_file=plan_root/'runtime'/(name+'.json')
                    runtime=dict(pamf,run_root=str(output),release_root=str(release_root),
                                 runtime_file=str(runtime_file),model_seeds=[task['seed']],
                                 scalr_source_root=str(source_selection/f"s{task['seed']}"))
                    write(runtime_file,runtime)
                    job=common
                    completion=output/'predictions'/task['capsule'].split('/')[-1]/'complete.json'
                else:
                    runtime_file=native_file
                    job=dict(common,profile=profile,output=str(output))
                    completion=output/'complete.json'
                write(jobfile,job)
                jobs.append(dict(id=name,group=group,runtime=str(runtime_file),job=str(jobfile),
                                 completion=str(completion),priority=0 if profile=='issl_fixed_v1' else 1))
    require(len(excluded)==5 and len(jobs)==(len(tasks())-5)*9,'Unexpected full queue size')
    config=dict(phase='full',run_root=str(run_root),release_root=str(release_root),jobs=jobs,
                allowed_gpus=gpus,max_per_gpu=5,minimum_free_MiB=6000,poll_seconds=5,
                input_condition='clean_ground_pre_satellite_normalized')
    write(plan_root/'scheduler.json',config)
    write(plan_root/'preparation.json',dict(status='READY_PARTIAL_MATRIX',jobs=len(jobs),
        excluded_pilot_identities=excluded,excluded_methods=['orthogonal','mopc_hr','all_hybrids'],
        global_prediction_barrier=False,truth_read=False))
    return len(jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('plan-root','run-root','release-root','template-root'):
        p.add_argument('--'+field,required=True)
    p.add_argument('--prior-config',action='append',required=True)
    p.add_argument('--gpu',action='append',type=int,required=True)
    a=p.parse_args()
    print(prepare(a.plan_root,a.run_root,a.release_root,a.template_root,a.prior_config,gpus=a.gpu))


if __name__=='__main__':main()
