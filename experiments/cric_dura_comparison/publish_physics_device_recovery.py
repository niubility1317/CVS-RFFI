"""Publish the bounded WRONG_REFERENCE CUDA construction recovery."""
import argparse
from .io import read
from .physics_device_recovery import CONTROL
from .publish_teacher_acceleration import publish


def runtime_for(previous, commit):
    if previous.get('run_id') != '20261023-phase1-PAMF-PL-Gfactor-control-r02':
        raise ValueError('Reconcile the current fairness owner before recovery')
    project = previous['project']; root = project + '/runs/' + CONTROL
    return dict(previous, run_id=CONTROL, release_root=project + '/releases/' + CONTROL,
        control_root=root, runtime_file=project + '/releases/' + CONTROL + '/runtime.json',
        code_commit=commit, launch_owner='codex-physics-device-recovery-20261023',
        source_jobs=previous['jobs'], jobs=root + '/resolved_jobs.json',
        previous_state=previous['control_root'] + '/pipeline_state.json',
        previous_runtime=previous['runtime_file'], previous_control_root=previous['control_root'],
        previous_controller_cwd=previous['release_root'],
        previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
        physics_device_recovery=True, pamf_gfactor_extension=False,
        F_source_binding_recovery=False, G_teacher_acceleration=False,
        supervised_extension=False, core_extension=False, external_dura_extension=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('repo', 'output', 'previous', 'ssh-config'):
        p.add_argument('--' + k, required=True)
    a = p.parse_args()
    publish(a.repo, a.output, a.previous, a.ssh_config, runtime_factory=runtime_for)
