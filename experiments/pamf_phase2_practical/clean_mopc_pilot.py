"""Run a small clean native-MoPC pilot after independent support diagnostics."""
import argparse
from pathlib import Path
import json
import time

from .clean_scheduler import dispatch, validate_plan
from .io import read, require, write

CAPSULES=('capsules/P01/MIX01/1-1','capsules/P02/MIX05/20-1','capsules/P03/MIX01/7-7')
DIAGNOSTICS=((5,CAPSULES[0],2026092702,10,20),
             (6,CAPSULES[1],2026092702,10,20),
             (7,CAPSULES[2],2026092703,1,20))


def verify_diagnostics(root,source_root):
    root=Path(root);tasks=read(root/'owner.json')['tasks']
    expected={gpu:(cap,seed,k,n) for gpu,cap,seed,k,n in DIAGNOSTICS}
    require(len(tasks)==3 and {t['gpu'] for t in tasks}==set(expected), 'Diagnostic task set mismatch')
    for task in tasks:
        cap,seed,k,n=expected[task['gpu']]
        require((task.get('capsule'),task.get('seed'),task.get('k'),task.get('n'))==(cap,seed,k,n),
                'Diagnostic task identity mismatch')
        path=root/('gpu'+str(task['gpu'])+'.log')
        require(path.is_file(), 'Diagnostic log missing')
        events=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: raise ValueError('Diagnostic log contains an error')
        done=[event for event in events if event.get('event')=='DIAGNOSTIC_COMPLETE']
        require(len(done)==1 and events[-1]==done[0] and done[0].get('semantics')=='public_code' and
                done[0].get('updates')==20*((5*k+15)//16)*4 and done[0].get('query_read') is False and
                done[0].get('truth_read') is False and done[0].get('predictions_written') is False and
                done[0].get('capsule')==cap and done[0].get('seed')==seed and
                done[0].get('k')==k and done[0].get('n')==n and done[0].get('mopc_epochs')==20 and
                done[0].get('stage_size')==5 and done[0].get('gradient_clipping') is False and
                Path(done[0].get('source_checkpoint','')).resolve()==(Path(source_root)/'source'/f'mopc_hr-s{seed}'/'source.pt').resolve(),
                'Diagnostic did not complete legally')
    return len(tasks)


def process_live(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def reserve_pilot(project,plan,root):
    reservation=Path(project)/'preparations'/'phase2-clean-native-mopc-pilot-owner'
    reservation.mkdir()
    write(reservation/'owner.json',dict(plan_root=str(plan),run_root=str(root)))


def run(diagnostic_root, runtime_file, source_root, plan_root, run_root, release_root):
    diagnostic=Path(diagnostic_root).resolve();plan=Path(plan_root).resolve();root=Path(run_root).resolve()
    release=Path(release_root).resolve();source=Path(source_root).resolve()
    require(not plan.exists() and not root.exists() and release.is_dir(), 'Unsafe pilot paths')
    owner=read(diagnostic/'owner.json');tasks=owner['tasks']
    require(len(tasks)==3 and {t['gpu'] for t in tasks}=={5,6,7}, 'Diagnostic task set mismatch')
    rt=read(runtime_file)
    plan.mkdir(parents=True)
    write(plan/'owner.json',dict(diagnostic_root=str(diagnostic),run_root=str(root)))
    while any(process_live(int(t['pid'])) for t in tasks):time.sleep(5)
    verify_diagnostics(diagnostic,source)
    rt.update(run_root=str(source),source_run_root=str(source),release_root=str(release))
    (plan/'jobs').mkdir();(plan/'runtime').mkdir()
    rtf=plan/'runtime/native.json';rt['runtime_file']=str(rtf);write(rtf,rt)
    jobs=[];seed=2026092702
    for i,capsule in enumerate(CAPSULES):
        name=f'mopc-public-{i:02d}';out=root/'tasks'/name;jf=plan/'jobs'/(name+'.json')
        write(jf,dict(profile='mopc_hr',seed=seed,capsule=capsule,points=[[5,20]],output=str(out)))
        jobs.append(dict(id=name,group='native_cil',runtime=str(rtf),job=str(jf),
                         completion=str(out/'complete.json'),priority=0))
    config=dict(phase='pilot',run_root=str(root),release_root=str(release),jobs=jobs,
                allowed_gpus=[5,6,7],max_per_gpu=2,minimum_free_MiB=6000,poll_seconds=5,
                one_per_snapshot=True,input_condition='clean_ground_pre_satellite_normalized')
    write(plan/'scheduler.json',config)
    write(plan/'preparation.json',dict(jobs=3,blocked=['full_retained_comparison_unfinished'],
          diagnostics=str(diagnostic),query_read=False,truth_read=False))
    validate_plan(config)
    reserve_pilot(rt['project'],plan,root)
    dispatch(config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic-root','runtime','source-root','plan-root','run-root','release-root'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.diagnostic_root,a.runtime,a.source_root,a.plan_root,a.run_root,a.release_root)


if __name__=='__main__':main()
