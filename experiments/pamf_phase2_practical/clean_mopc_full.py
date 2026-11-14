"""Launch clean native and B-backbone MoPC after predeclared stability checks."""
import argparse
import json
from pathlib import Path

from .clean_comparison_plan import tasks
from .clean_scheduler import completion_matches, dispatch, validate_plan
from .clean_score import expected_rows, key
from .io import read, require, write

PROFILES=('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft','tweak_m1')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def verify_stability_evidence(synthetic,real):
    found=set()
    for gpu in (5,6,7):
        for event in lines(Path(synthetic)/f'gpu{gpu}.log'):
            require(event.get('scope')=='synthetic_only' and event.get('semantics')=='public_code' and
                    event.get('query_read') is False and event.get('truth_read') is False and
                    event.get('predictions_written') is False, 'Invalid synthetic diagnostic')
            seed=event['seed'];require(seed in range(1731,1737) and seed not in found,'Duplicate synthetic diagnostic')
            variants={row['variant']:row for row in event['results']}
            require(set(variants)=={'no_clip','clip1'} and
                    variants['no_clip']['status']=='COMPLETE' and variants['no_clip']['completed_updates']==160 and
                    variants['clip1']['status']=='COMPLETE' and variants['clip1']['completed_updates']==160,
                    'Synthetic diagnostic did not complete')
            found.add(seed)
    require(found==set(range(1731,1737)),'Missing synthetic diagnostics')
    for gpu,key_name,k,updates in ((5,'protonet_cda-s2026092702-c0195',20,560),
                                   (6,'dadda_sda-s2026092703-c0332',10,320),
                                   (7,'dadda_sda-s2026092703-c0332',20,560)):
        events=lines(Path(real)/f'gpu{gpu}.log')
        require(len(events)==3 and events[0].get('event')=='SOURCE_SMOKE_PASS' and
                events[0].get('query_rows')==0 and events[0].get('truth_read') is False and
                events[1].get('event')=='SAVED_B_SUPPORT_READY' and events[1].get('key')==key_name and
                events[1].get('k')==k and events[1].get('n')==20 and events[1].get('query_rows')==0 and
                events[1].get('truth_read') is False and events[2].get('event')=='REGISTRATION_RETURNED' and
                events[2].get('observed_nonfinite') is False and events[2].get('completed_updates')==updates,
                'Saved-B diagnostic did not complete')
    return 9


def fixed_pilot_refs(config_file):
    refs={};config=read(config_file)
    for entry in config['jobs']:
        job=read(entry['job']);require(entry['group']=='native_cil' and job['profile']=='mopc_hr' and
            completion_matches(entry),'Pilot prediction is not fixed')
        root=Path(entry['completion']).parent;payload=read(root/'predictions.json')
        actual=[key(row) for row in payload['rows']]
        require(payload['capsule']==job['capsule'] and payload['model_seed']==job['seed'] and
                len(actual)==len(set(actual)) and set(actual)==expected_rows('native_cil',job) and
                all(row['input_condition']=='clean_ground_pre_satellite_normalized' and
                    (root/row['prediction']).is_file() for row in payload['rows']), 'Pilot rows incomplete')
        identity=(job['capsule'],job['seed']);require(identity not in refs,'Duplicate pilot')
        refs[identity]=dict(root=str(root),job=entry['job'],runtime=entry['runtime'])
    require(len(refs)==3,'Expected three completed pilot jobs')
    return refs


def verify_no_other_mopc_plans(preparations,pilot_config):
    pilot=Path(pilot_config).resolve()
    for file in Path(preparations).glob('*clean*/scheduler.json'):
        if file.resolve()==pilot:continue
        for entry in read(file)['jobs']:
            if entry['group'] not in ('native_cil','hybrid'):continue
            job=read(entry['job'])
            require(not (entry['group']=='native_cil' and job.get('profile')=='mopc_hr') and
                    not (entry['group']=='hybrid' and 'mopc_hr' in job.get('methods',[])),
                    'Another clean MoPC plan exists; reconcile before launch')


def run(synthetic,real,pilot_config,runtime_file,source_root,plan_root,run_root,release_root):
    plan=Path(plan_root).resolve();root=Path(run_root).resolve();release=Path(release_root).resolve()
    source=Path(source_root).resolve();rt=read(runtime_file);project=Path(rt['project']).resolve()
    require(not plan.exists() and not root.exists() and release.is_dir(),'Unsafe full MoPC paths')
    verify_stability_evidence(synthetic,real);pilots=fixed_pilot_refs(pilot_config)
    verify_no_other_mopc_plans(project/'preparations',pilot_config)
    plan.mkdir(parents=True);(plan/'jobs').mkdir();(plan/'runtime').mkdir()
    hybrid=plan/'runtime/hybrid.json';native=plan/'runtime/native.json'
    common=dict(rt,release_root=str(release));write(hybrid,dict(common,runtime_file=str(hybrid)))
    write(native,dict(common,run_root=str(source),source_run_root=str(source),runtime_file=str(native)))
    jobs=[];task_list=tasks()
    require(len(task_list)==1932,'Unexpected fixed comparison matrix')
    for i,task in enumerate(task_list):
        for group,profiles in (('native_cil',('mopc_hr',)),('hybrid',PROFILES)):
            for profile in profiles:
                name=f'{i:04d}-{group}-{profile}';out=root/'tasks'/name;jf=plan/'jobs'/(name+'.json')
                job=dict(task,profile=profile,output=str(out))
                job.pop('pilot_point',None)
                if group=='native_cil':
                    identity=(task['capsule'],task['seed'])
                    if identity in pilots:job['reuse']=pilots[identity]
                    rtf=native
                else:
                    job.update(key=name,methods=['mopc_hr'],save_B_for_mopc=True);rtf=hybrid
                write(jf,job)
                jobs.append(dict(id=name,group=group,runtime=str(rtf),job=str(jf),
                                 completion=str(out/'complete.json'),priority=0))
    require(len(jobs)==13524,'Unexpected MoPC full queue')
    config=dict(phase='full',run_root=str(root),release_root=str(release),jobs=jobs,
                allowed_gpus=[5,6,7],max_per_gpu=7,minimum_free_MiB=6000,worker_reserve_MiB=3000,
                one_per_snapshot=True,poll_seconds=5,input_condition='clean_ground_pre_satellite_normalized')
    write(plan/'scheduler.json',config)
    write(plan/'preparation.json',dict(jobs=len(jobs),reused_pilot_identities=3,
          methods=['public_code_native','public_code_B_to_C'],old_quarantines_retried=False,
          global_prediction_barrier=False,truth_read=False))
    validate_plan(config)
    reservation=project/'preparations'/'phase2-clean-mopc-full-owner';reservation.mkdir()
    write(reservation/'owner.json',dict(plan_root=str(plan),run_root=str(root),pilot_config=str(pilot_config)))
    dispatch(config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('synthetic','real','pilot-config','runtime','source-root','plan-root','run-root','release-root'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.synthetic,a.real,a.pilot_config,a.runtime,a.source_root,a.plan_root,a.run_root,a.release_root)


if __name__=='__main__':main()
