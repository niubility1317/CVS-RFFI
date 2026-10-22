"""Append frozen external+DURA readouts with a healthy-worker owner handoff."""
import argparse
from pathlib import Path
from .external_dura import CONTROL,validate_rows
from .io import read
from .publish_teacher_acceleration import publish


def runtime_for(previous,commit,manifest,external_releases):
    if previous.get('worker_commit')!='3e8806900449e97f3e8fa62822322932168c6447':
        raise ValueError('Unexpected original scientific worker; reconcile owner')
    validate_rows(manifest['rows']);project=previous['project'];root=project+'/runs/'+CONTROL
    # Pin only routing metadata; original model implementations and checkpoints
    # remain in their immutable original releases and are never modified.
    return dict(previous,run_id=CONTROL,release_root=project+'/releases/'+CONTROL,control_root=root,
        runtime_file=project+'/releases/'+CONTROL+'/runtime.json',code_commit=commit,
        launch_owner='codex-external-DURA-frozen-20261023',source_jobs=previous['jobs'],jobs=root+'/resolved_jobs.json',
        previous_state=previous['control_root']+'/pipeline_state.json',previous_runtime=previous['runtime_file'],
        previous_control_root=previous['control_root'],previous_controller_cwd=previous['release_root'],
        previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
        external_dura_extension=True,external_manifest=manifest,external_releases=external_releases,
        F_source_binding_recovery=False,G_teacher_acceleration=False,supervised_extension=False,core_extension=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','output','previous','ssh-config','manifest','external-runtime','count-runtime'):p.add_argument('--'+key,required=True)
    a=p.parse_args();m=read(a.manifest)
    # The preflight uses existing validated handles and no target scores.
    if m['status']!='PASS' or m['target_score_reads']!=0:raise ValueError('Only score-free existing source mapping is permitted')
    templates={pid:{k:v for k,v in t.items() if k in ('protocol_id','source_contract','source_cache','source_days','readouts','tests')}
        for pid,t in m['G_templates'].items()}
    for t in templates.values():t['tests']={name:{'views_root':v['views_root']} for name,v in t['tests'].items()}
    manifest=dict(rows=m['source_rows'],G_templates=templates)
    releases=[read(a.external_runtime)['release_root'],read(a.count_runtime)['release_root']]
    publish(a.repo,a.output,a.previous,a.ssh_config,
        runtime_factory=lambda previous,commit:runtime_for(previous,commit,manifest,releases))
