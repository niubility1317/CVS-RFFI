"""Immutable supervised supplement; preserve the current mixed-release queue."""
import argparse
from .supervised_extension import CONTROL
from .publish_teacher_acceleration import publish


def runtime_for(previous,commit):
    if previous['run_id']!='20261022-phase1-T1-DURA-F-binding-control-r01':
        raise ValueError('Reconcile current owner before supervised supplement')
    project=previous['project'];control=project+'/runs/'+CONTROL
    return dict(previous,run_id=CONTROL,release_root=project+'/releases/'+CONTROL,control_root=control,
        runtime_file=project+'/releases/'+CONTROL+'/runtime.json',code_commit=commit,
        launch_owner='codex-PAMF-Lonly-physics-20261022',source_jobs=previous['jobs'],jobs=control+'/resolved_jobs.json',
        previous_state=previous['control_root']+'/pipeline_state.json',previous_runtime=previous['runtime_file'],
        previous_control_root=previous['control_root'],previous_controller_cwd=previous['release_root'],
        F_source_binding_recovery=False,G_teacher_acceleration=False,supervised_extension=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','output','previous','ssh-config'):p.add_argument('--'+key,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.previous,a.ssh_config,runtime_factory=runtime_for)
