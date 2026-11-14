"""Safely continue clean MoPC with an explicitly versioned ProtoNet bridge."""
import argparse
from pathlib import Path

from .clean_scheduler import completion_matches, dispatch, validate_plan
from .io import read, require, write


def partition(root, entries, failed_ids):
    result = dict(completed_other=[], pending_other=[], replace_proto=[], pending_proto=[])
    found_failed = set()
    for entry in entries:
        job = read(entry['job'])
        proto = entry['group'] == 'hybrid' and job.get('profile') == 'protonet_cda'
        output = Path(entry['completion']).parent
        log = Path(root)/'scheduler'/'logs'/(entry['id']+'.log')
        completed = Path(entry['completion']).is_file()
        if completed:
            require(entry['id'] not in failed_ids and completion_matches(entry), 'Invalid old completion')
            result['replace_proto' if proto else 'completed_other'].append(entry)
        else:
            started = output.exists() or log.exists()
            if started:
                require(proto and entry['id'] in failed_ids, 'Unclassified old partial')
                found_failed.add(entry['id'])
                result['replace_proto'].append(entry)
            else:
                require(entry['id'] not in failed_ids, 'Failed job has no retained evidence')
                result['pending_proto' if proto else 'pending_other'].append(entry)
    require(found_failed == set(failed_ids), 'Unclassified failed jobs')
    return result


def run(parent_config, plan_root, run_root, release_root):
    parent_file=Path(parent_config).resolve();parent=read(parent_file)
    old=Path(parent['run_root']).resolve();control=old/'scheduler'
    plan=Path(plan_root).resolve();root=Path(run_root).resolve();release=Path(release_root).resolve()
    require(not plan.exists() and not root.exists() and release.is_dir(), 'Unsafe successor paths')
    state=read(control/'state.json')
    require(state.get('status')=='FAILED' and state.get('active')==[] and len(state.get('failed',[]))==1,
            'Original controller is not failed and drained')
    failed_ids={item['id'] for item in state['failed']}
    owner=read(control/'owner.json');proc=Path('/proc')/str(owner['pid'])
    if proc.exists():
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            require(str(old) not in cmd, 'Original controller is still running')
        except FileNotFoundError:
            pass
    parts=partition(old,parent['jobs'],failed_ids)
    require(len(parent['jobs'])==13524 and len(parts['completed_other'])+len(parts['replace_proto'])-1==432 and
            len(parts['replace_proto'])+len(parts['pending_proto'])==1932,
            'Unexpected old queue inventory')
    plan.mkdir(parents=True);(plan/'jobs').mkdir();(plan/'runtime').mkdir()
    jobs=[];runtimes={}
    for category in ('replace_proto','pending_proto','pending_other'):
        for entry in parts[category]:
            old_job=read(entry['job']);job=dict(old_job,output=str(root/'tasks'/entry['id']))
            if category in ('replace_proto','pending_proto'):
                job['mopc_bridge']='proto_affine_v1'
                if category=='replace_proto':
                    job['reuse_original_b']=dict(job=entry['job'],runtime=entry['runtime'])
            old_runtime=entry['runtime']
            if old_runtime not in runtimes:
                runtime_file=plan/'runtime'/(str(len(runtimes))+'.json')
                runtime=read(old_runtime);runtime.update(release_root=str(release),runtime_file=str(runtime_file))
                write(runtime_file,runtime);runtimes[old_runtime]=str(runtime_file)
            jf=plan/'jobs'/(entry['id']+'.json');write(jf,job)
            jobs.append(dict(entry,job=str(jf),runtime=runtimes[old_runtime],
                             completion=str(Path(job['output'])/'complete.json')))
    require(len(jobs)==len(parent['jobs'])-len(parts['completed_other']), 'Wrong successor matrix')
    config=dict(parent,run_root=str(root),release_root=str(release),jobs=jobs,
                allowed_gpus=[5,6,7],max_per_gpu=7,minimum_free_MiB=6000,
                worker_reserve_MiB=3000,one_per_snapshot=True)
    config.pop('auto_score',None)
    write(plan/'scheduler.json',config)
    write(plan/'preparation.json',dict(jobs=len(jobs),completed_other=len(parts['completed_other']),
          replace_proto=len(parts['replace_proto']),pending_proto=len(parts['pending_proto']),
          pending_other=len(parts['pending_other']),parent=str(old),truth_read=False,
          proto_version='proto_affine_v1',old_proto_results_retained=True))
    validate_plan(config)
    reservation=control/'affine_successor';reservation.mkdir()
    write(reservation/'owner.json',dict(plan_root=str(plan),run_root=str(root)))
    dispatch(config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-config','plan-root','run-root','release-root'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.parent_config,a.plan_root,a.run_root,a.release_root)


if __name__=='__main__':
    main()
