"""Launch a versioned MoPC matrix, preserving and reusing prior source/B states."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .clean_scheduler import completion_matches, dispatch, numeric_quarantine_matches, validate_plan
from .io import read, require, write


def select_entries(parent,successor):
    originals={x['id']:x for x in parent};replacements={x['id']:x for x in successor}
    require(len(originals)==len(parent) and len(replacements)==len(successor),'duplicate parent or successor id')
    require(set(replacements)<=set(originals),'unknown successor id')
    for key,entry in replacements.items():require(entry['group']==originals[key]['group'],'successor group differs')
    return [replacements.get(x['id'],x) for x in parent]


def retained_partial_ids(state, entries):
    by_id={x['id']:x for x in entries};failed=state.get('failed',[]);quarantined=state.get('quarantined',[])
    identifiers=[x['id'] for x in failed+quarantined]
    require(len(identifiers)==len(set(identifiers)) and set(identifiers)<=set(by_id),
            'Unknown or duplicate retained failure')
    for item in quarantined:
        require(numeric_quarantine_matches(by_id[item['id']]),'Retained numerical quarantine differs')
    return set(identifiers)


def drained(config):
    root=Path(config['run_root']).resolve();control=root/'scheduler';state=read(control/'state.json')
    require(state.get('status') in ('FAILED','STOPPED','COMPLETED_WITH_QUARANTINE') and state.get('active')==[],
            'Parent is not failed/stopped and drained')
    owner=read(control/'owner.json');proc=Path('/proc')/str(owner['pid'])
    if proc.exists():
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            require(root.name not in cmd, 'Parent controller still exists')
        except FileNotFoundError:pass
    return state


def start_watcher_and_reserve(plan,root,release,score,reservation):
    cmd=[sys.executable,'-u','-m',__package__+'.clean_auto_score',
        '--preparations',str(plan.parent),'--output',str(score),'--workers','4',
        '--mopc-stabilization','clip1000_v1']
    process=None;created=False
    try:
        with (plan/'auto_score.log').open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,cwd=release,env=dict(os.environ,OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        require(process.poll() is None,'Automatic scorer exited at startup')
        write(plan/'auto_score_owner.json',dict(pid=process.pid,output=str(score),stabilization='clip1000_v1'))
        reservation.mkdir();created=True
        write(reservation/'owner.json',dict(plan_root=str(plan),run_root=str(root),score_output=str(score)))
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=10)
        if created:
            owner=reservation/'owner.json'
            if owner.exists():owner.unlink()
            reservation.rmdir()
        raise


def run(parent_file,successor_file,plan_root,run_root,release_root,score_output):
    parent=read(parent_file);successor=read(successor_file)
    old_state=drained(parent);new_state=drained(successor)
    plan=Path(plan_root).resolve();root=Path(run_root).resolve();release=Path(release_root).resolve()
    score=Path(score_output).resolve()
    require(plan.name.startswith('mopc-stable-') and not plan.exists() and not root.exists() and
            not score.exists() and release.is_dir(),'Unsafe stable output or scorer path')
    selected=select_entries(parent['jobs'],successor['jobs'])
    require(len(selected)==13524,'Unexpected fixed MoPC matrix')
    orig_by_id={x['id']:x for x in parent['jobs']};new_ids={x['id'] for x in successor['jobs']}
    failed=retained_partial_ids(new_state,successor['jobs'])
    plan.mkdir(parents=True);(plan/'jobs').mkdir();(plan/'runtime').mkdir()
    jobs=[];runtimes={};reused_b=reused_a0=partial=0
    for entry in selected:
        old_job=read(entry['job']);original=read(orig_by_id[entry['id']]['job'])
        require(all(old_job.get(k)==original.get(k) for k in ('capsule','profile','seed','points')),
                'Successor task identity differs')
        folder=Path(entry['completion']).parent;done=Path(entry['completion']).is_file()
        if done:require(completion_matches(entry),'Invalid retained completion')
        else:
            prior=Path(successor['run_root']) if entry['id'] in new_ids else Path(parent['run_root'])
            started=folder.exists() or (prior/'scheduler'/'logs'/(entry['id']+'.log')).exists()
            require(not started or (entry['id'] in failed and entry['id'] in new_ids), 'Unclassified retained partial')
            partial+=int(started)
        job=dict(old_job,output=str(root/'tasks'/entry['id']),mopc_stabilization='clip1000_v1')
        if entry['group']=='hybrid' and folder.is_dir():
            job['reuse_saved_b']=dict(job=entry['job'],runtime=entry['runtime']);reused_b+=1
        if entry['group']=='native_cil' and done:
            job.pop('reuse',None);job['reuse_a0']=dict(job=entry['job'],runtime=entry['runtime']);reused_a0+=1
        old_runtime=entry['runtime']
        if old_runtime not in runtimes:
            file=plan/'runtime'/(str(len(runtimes))+'.json')
            rt=read(old_runtime);rt.update(release_root=str(release),runtime_file=str(file))
            write(file,rt);runtimes[old_runtime]=str(file)
        jf=plan/'jobs'/(entry['id']+'.json');write(jf,job)
        jobs.append(dict(entry,job=str(jf),runtime=runtimes[old_runtime],
                         completion=str(Path(job['output'])/'complete.json')))
    config=dict(phase='full',run_root=str(root),release_root=str(release),jobs=jobs,
        allowed_gpus=[0,1,5,6,7],max_per_gpu=7,minimum_free_MiB=6000,worker_reserve_MiB=3000,
        one_per_snapshot=True,poll_seconds=5,input_condition='clean_ground_pre_satellite_normalized',
        numerical_policy='quarantine')
    write(plan/'scheduler.json',config)
    write(plan/'preparation.json',dict(jobs=len(jobs),reused_b=reused_b,reused_a0=reused_a0,
        prior_partial=partial,stabilization='clip1000_v1',old_outputs_preserved=True,truth_read=False))
    validate_plan(config)
    reservation=Path(successor['run_root'])/'scheduler'/'stable_successor'
    start_watcher_and_reserve(plan,root,release,score,reservation)
    dispatch(config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-config','successor-config','plan-root','run-root','release-root','score-output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.parent_config,a.successor_config,a.plan_root,a.run_root,a.release_root,a.score_output)


if __name__=='__main__':main()
