"""Prepare only missing clean main-scope native DA A0/B rows; never dispatch."""
import argparse
from fnmatch import fnmatchcase
from pathlib import Path
import re

from .clean_scheduler import validate_plan
from .io import read, require, write


def match_templates(targets, templates):
    matches=[]
    for target in targets:
        found=[(entry,job) for entry,job in templates if entry['group']=='native_da' and
               all(job.get(k)==target[k] for k in ('profile','capsule','seed'))]
        require(len(found)==1,'Missing or ambiguous native DA pilot template')
        matches.append((target,*found[0]))
    return matches


def prepare(parent_file, targets_file, missing_file, plan_root, run_root, release_root):
    parent=read(parent_file);targets=read(targets_file);coverage=read(missing_file)
    plan,run,release=map(lambda x:Path(x).resolve(),(plan_root,run_root,release_root))
    from .clean_auto_score import EXCLUDED_PLANS
    require(plan.parent==Path(parent_file).resolve().parent.parent and fnmatchcase(plan.name,'*phase2*clean*') and
            re.fullmatch(r'[A-Za-z0-9_-]+',plan.name) is not None and plan.name not in EXCLUDED_PLANS,
            'Core plan must be discoverable by the incremental scorer')
    require(len(targets)==30 and len({(t['profile'],t['capsule'],t['seed'],t['k']) for t in targets})==30,
            'Wrong native DA target set')
    require(all(t['k']==10 and '/MIX' in t['capsule'] for t in targets),'Wrong clean main-scope target')
    require(coverage.get('scope')=='MIX01-08 K5/10 N5/10','Wrong missing-results scope')
    require(not plan.exists() and not run.exists() and release.is_dir(),'Output exists or release is missing')
    templates=[(e,read(e['job'])) for e in parent['jobs'] if e['group']=='native_da']
    selected=match_templates(targets,templates)
    plan.mkdir(parents=True);(plan/'jobs').mkdir();(plan/'runtime').mkdir()
    reduced={name:rows for name,rows in coverage['missing'].items() if name.startswith('native_da/')}
    write(plan/'missing.json',dict(scope=coverage['scope'],missing=reduced))
    runtimes={};jobs=[]
    for target,old,oldjob in selected:
        parts=target['capsule'].split('/')
        ident=f"core-{target['profile']}-{parts[1]}-{parts[2]}-{parts[3]}-s{target['seed']}-K10"
        runtime=old['runtime']
        if runtime not in runtimes:
            dest=plan/'runtime'/f'{len(runtimes)}.json';rt=read(runtime);rt['release_root']=str(release)
            write(dest,rt);runtimes[runtime]=str(dest)
        job=dict(oldjob,points=[[10,5]],output=str(run/'tasks'/ident))
        path=plan/'jobs'/(ident+'.json');write(path,job)
        jobs.append(dict(id=ident,group='native_da',runtime=runtimes[runtime],job=str(path),
                         completion=str(Path(job['output'])/'complete.json'),priority=0))
    config=dict(phase='core_native_da',run_root=str(run),release_root=str(release),jobs=jobs,
        allowed_gpus=[0,1,5,6,7],max_per_gpu=7,minimum_free_MiB=6000,worker_reserve_MiB=3000,
        one_per_snapshot=True,poll_seconds=5,input_condition='clean_ground_pre_satellite_normalized',
        numerical_policy='quarantine',core_missing_file=str(plan/'missing.json'))
    write(plan/'scheduler.json',config)
    return validate_plan(config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','targets','missing','plan-root','run-root','release-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(prepare(a.parent,a.targets,a.missing,a.plan_root,a.run_root,a.release_root))


if __name__=='__main__':main()
