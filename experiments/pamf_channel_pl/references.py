"""Resolve historical CPU score recoveries without changing scientific outputs."""
import argparse
import json
import os
from pathlib import Path

SCIENCE_RUN='20261025-phase1-PAMF-ChannelPL-m33-r01'
OWNER='codex-channel-pl-training-20261025'
CONTROLS=('supervised','ordinary_PL','REPL')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_reference_score_paths(job_file,wanted):
    result={}
    for job in read(job_file):
        if job['kind']!='score':continue
        c=read(job['config']);rid=c['row_id']
        if rid not in wanted:continue
        complete=Path(job['completion'])
        if not complete.exists():continue
        d=read(complete)
        if d.get('status')!='SCORED_COMPLETE':continue
        stem,seed=rid.rsplit('-s',1);protocol,method=stem.rsplit('-',1)
        if (method not in CONTROLS or c['model_seed']!=int(seed) or c['protocol_id']!=protocol
            or d['row_id']!=rid or d['run_id']!=c['run_id'] or Path(c['output_root'])!=complete.parent):
            raise ValueError('Wrong recorded reference row/seed/output')
        score=complete.parent/'scores.json'
        if not score.is_file():raise ValueError('Recorded completed score is missing: '+rid)
        if rid in result and result[rid]!=str(score):raise ValueError('Ambiguous completed reference: '+rid)
        result[rid]=str(score)
    if set(result)!=set(wanted):raise ValueError('Missing completed reference rows: '+repr(sorted(set(wanted)-set(result))))
    return result


def repair(run_root,job_file,expected_commit,*,apply=False):
    root=Path(run_root).resolve();jobs=Path(job_file).resolve()
    if root.name!=SCIENCE_RUN or jobs.name!='jobs.json' or not jobs.is_relative_to(root.parent):
        raise ValueError('Exact owned run and same-project historical jobs required')
    state=read(root/'pipeline_state.json')
    if state['code_commit']!=expected_commit or state['launch_owner']!=OWNER:raise ValueError('Scientific owner/version differs')
    current=read(root/'reference_scores.json');wanted=set();bindings={}
    for row,controls in current.items():
        protocol,seed=row.rsplit('-ChannelPL-s',1)
        if set(controls)!=set(CONTROLS):raise ValueError('Comparison method matrix differs')
        for method in controls:
            rid=protocol+'-'+method+'-s'+seed;wanted.add(rid);bindings[row,method]=rid
    resolved=resolve_reference_score_paths(jobs,wanted);updated={row:dict(v) for row,v in current.items()};changes=[]
    for (row,method),rid in bindings.items():
        old=current[row][method];new=resolved[rid]
        header=read(new)
        if (header['status']!='SCORED' or header['truth_last'] is not True or header['row_id']!=rid
            or header['model_seed']!=int(rid.rsplit('-s',1)[1])):raise ValueError('Wrong independently completed score header')
        if old==new:continue
        if Path(old).exists():raise ValueError('Refuse replacing an existing different score')
        updated[row][method]=new;changes.append(dict(row=row,method=method,old=old,new=new))
    result=dict(status='VERIFIED_PREVIEW' if changes else 'VERIFIED_ALREADY_CORRECT',changes=changes,
        reference_rows=len(current),reference_files=len(wanted),scientific_execution_commit=expected_commit,
        model_prediction_score_files_modified=0,training_configuration_modified=False)
    if apply and changes:
        backup=root/'reference_scores_before_path_repair.json'
        if backup.exists():
            if read(backup)!=current:raise ValueError('Reconcile existing reporting-path backup before retry')
        else:
            with backup.open('x',encoding='utf-8') as stream:json.dump(current,stream,indent=2);stream.write('\n')
        temporary=root/'reference_scores_path_repair_pending.json'
        if temporary.exists():
            if read(temporary)!=updated:raise ValueError('Unexpected pending metadata repair')
        else:
            with temporary.open('x',encoding='utf-8') as stream:json.dump(updated,stream,indent=2);stream.write('\n')
        temporary.replace(root/'reference_scores.json')
        if read(root/'reference_scores.json')!=updated:raise ValueError('Reporting path readback differs')
        result['status']='VERIFIED_REPAIRED'
        with (root/'reference_score_path_repair.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--reference-jobs',required=True)
    p.add_argument('--expected-commit',required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    if hasattr(os,'geteuid') and os.geteuid()==0:raise ValueError('Ordinary account required')
    print(json.dumps(repair(a.run_root,a.reference_jobs,a.expected_commit,apply=a.apply)))
