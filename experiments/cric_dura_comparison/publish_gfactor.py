"""Release the already-registered51 frozen PL/G-L fairness readouts."""
import argparse
from .io import read
from .pamf_gfactor import CONTROL,validate_cells
from .publish_teacher_acceleration import publish


def runtime_for(previous,commit,manifest):
    if previous.get('run_id')!='20261023-phase1-external-DURA-control-r01':
        raise ValueError('Reconcile current owner before the fairness handoff')
    if manifest.get('target_score_reads')!=0 or manifest.get('target_truth_reads')!=0 or manifest.get('status')!='PASS':
        raise ValueError('Only score-free source dependency mapping is permitted')
    validate_cells(manifest['cells']);project=previous['project'];root=project+'/runs/'+CONTROL
    return dict(previous,run_id=CONTROL,release_root=project+'/releases/'+CONTROL,control_root=root,
        runtime_file=project+'/releases/'+CONTROL+'/runtime.json',code_commit=commit,
        launch_owner='codex-PAMF-PL-Gfactor-20261023',source_jobs=previous['jobs'],jobs=root+'/resolved_jobs.json',
        previous_state=previous['control_root']+'/pipeline_state.json',previous_runtime=previous['runtime_file'],
        previous_control_root=previous['control_root'],previous_controller_cwd=previous['release_root'],
        previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
        pamf_gfactor_extension=True,pamf_gfactor_manifest=dict(cells=manifest['cells']),
        F_source_binding_recovery=False,G_teacher_acceleration=False,supervised_extension=False,
        core_extension=False,external_dura_extension=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','previous','ssh-config','manifest'):p.add_argument('--'+k,required=True)
    a=p.parse_args();m=read(a.manifest)
    publish(a.repo,a.output,a.previous,a.ssh_config,runtime_factory=lambda previous,commit:runtime_for(previous,commit,m))
