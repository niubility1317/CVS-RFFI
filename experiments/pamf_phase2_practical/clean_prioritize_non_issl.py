"""Prepare a successor from unstarted non-ISSL jobs after the parent has drained."""
import argparse
import os
from pathlib import Path

from .io import read, require, write


def prepare(parent_config, plan_root, run_root, release_root, orthogonal_root):
    parent=read(parent_config)
    old_root=Path(parent['run_root']).resolve()
    control=old_root/'scheduler'
    state=read(control/'state.json')
    owner=read(control/'owner.json')
    require(not Path('/proc',str(owner['pid'])).exists(), 'Parent controller still exists')
    require(state['pending']==0 and not state['active'] and (control/'USER_STOP.json').is_file(),
            'Parent did not drain after requested stop')
    require(all('issl_fixed_v1' in row['id'] for row in state['failed']),'Unexpected parent failure')
    proc=Path('/proc')
    if proc.is_dir():
        for process in proc.iterdir():
            if not process.name.isdigit():continue
            try:
                if process.stat().st_uid!=os.getuid():continue
                command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                require(not (str(old_root.name) in command and
                        ('clean_external_worker' in command or 'clean_scalr_worker' in command
                         or 'clean_scheduler' in command)), 'Parent process remains')
            except (FileNotFoundError,PermissionError,ProcessLookupError):
                continue
    plan=Path(plan_root).resolve();run=Path(run_root).resolve();release=Path(release_root).resolve()
    orthogonal=Path(orthogonal_root).resolve()
    require(not plan.exists() and not run.exists(),'Successor output already exists')
    for seed in (2026092701,2026092702,2026092703,2026092704):
        folder=orthogonal/'source'/f'orthogonal-s{seed}'
        require((folder/'source.pt').is_file()
                and read(folder/'group1_complete.json')['status']=='GROUP1_SOURCE_READY'
                and read(folder/'recipe.json')['schema']=='group1_source_orthogonal_tempered_PL_v2',
                'Corrected Orthogonal source incomplete')
    entries=[];previous=set();profiles={}
    for entry in parent['jobs']:
        job=read(entry['job']);profile=job.get('profile','R42')
        if profile=='issl_fixed_v1':continue
        prior_output=Path(entry['completion']).parent
        require(not (old_root/'tasks'/entry['id']).exists() and not prior_output.exists()
                and not (control/'logs'/(entry['id']+'.log')).exists(),
                'Non-ISSL task has prior output or launch log; reconcile before continuation')
        identity=(entry['group'],profile,job['capsule'],job['seed'])
        require(identity not in previous,'Duplicate parent task identity')
        previous.add(identity);entries.append((entry,job))
        if profile=='lora_triplet_knn':
            profiles[(job['capsule'],job['seed'])]=job
    require(len(entries)==15416 and len(profiles)==1927,'Unexpected unstarted parent partition')
    reservation=control/'non_issl_successor'
    reservation.mkdir()
    write(reservation/'owner.json',dict(plan_root=str(plan),run_root=str(run),parent=str(old_root)))
    plan.mkdir(parents=True);(plan/'jobs').mkdir();(plan/'runtime').mkdir()
    native_source=read(next(entry['runtime'] for entry,job in entries if entry['group']=='native_da'))
    native_file=plan/'runtime/native.json'
    native_source.update(release_root=str(release),runtime_file=str(native_file))
    write(native_file,native_source)
    orthogonal_file=plan/'runtime/orthogonal.json'
    write(orthogonal_file,dict(native_source,run_root=str(orthogonal),runtime_file=str(orthogonal_file)))
    jobs=[]
    for entry,job in entries:
        output=run/'tasks'/entry['id']
        job_file=plan/'jobs'/(entry['id']+'.json')
        if entry['group']=='pamf':
            runtime_file=plan/'runtime'/(entry['id']+'.json')
            runtime=read(entry['runtime'])
            runtime.update(run_root=str(output),release_root=str(release),runtime_file=str(runtime_file))
            write(runtime_file,runtime)
            completion=output/'predictions'/job['capsule'].split('/')[-1]/'complete.json'
        else:
            runtime_file=native_file
            job=dict(job,output=str(output))
            completion=output/'complete.json'
        write(job_file,job)
        jobs.append(dict(entry,job=str(job_file),runtime=str(runtime_file),completion=str(completion),priority=0))
    for i,job in enumerate(profiles.values()):
        name=f'orthogonal-{i:04d}';output=run/'tasks'/name
        file=plan/'jobs'/(name+'.json')
        write(file,dict(job,profile='orthogonal',output=str(output)))
        jobs.append(dict(id=name,group='native_cil',job=str(file),runtime=str(orthogonal_file),
                         completion=str(output/'complete.json'),priority=0))
    require(len(jobs)==17343,'Unexpected successor queue size')
    config=dict(parent,run_root=str(run),release_root=str(release),jobs=jobs,
                allowed_gpus=[1,2,3],max_per_gpu=5)
    config.pop('auto_score',None)
    write(plan/'scheduler.json',config)
    write(plan/'preparation.json',dict(status='OTHER_METHODS_PRIORITIZED',jobs=len(jobs),
          issl_deferred=1927,parent=str(old_root),parent_partial_retained=True,
          excluded=['ISSL','MoPC','hybrid','five_pilot_remaining_points'],
          global_prediction_barrier=False,truth_read=False))
    return len(jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-config','plan-root','run-root','release-root','orthogonal-root'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    print(prepare(a.parent_config,a.plan_root,a.run_root,a.release_root,a.orthogonal_root))


if __name__=='__main__':main()
