"""One immutable repair owner for the diagnosed69 F rows and downstream tests."""
import argparse
from .f_binding_recovery import CONTROL
from .publish_teacher_acceleration import publish


def runtime_for(previous,commit):
    if previous['run_id']!='20261022-phase1-DURA-teacher-cache-control-r01':
        raise ValueError('Unexpected prior resource owner; reconcile before repair')
    project=previous['project'];root=project+'/runs/'+CONTROL
    return dict(previous,run_id=CONTROL,release_root=project+'/releases/'+CONTROL,control_root=root,
        runtime_file=project+'/releases/'+CONTROL+'/runtime.json',code_commit=commit,
        launch_owner='codex-F-source-ID-binding-repair-20261022',source_jobs=previous['jobs'],jobs=root+'/resolved_jobs.json',
        previous_state=previous['control_root']+'/pipeline_state.json',previous_runtime=previous['runtime_file'],
        previous_control_root=previous['control_root'],previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
        previous_controller_cwd=previous['release_root'],G_teacher_acceleration=False,F_source_binding_recovery=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','output','previous','ssh-config'):p.add_argument('--'+key,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.previous,a.ssh_config,runtime_factory=runtime_for)
