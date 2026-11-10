"""Publish only after the matched fixed-budget run is fully analyzed."""
import argparse
from experiments.pamf_branch_training import publish as base
from experiments.pamf_physics_branch_balanced.plan import RUN as PREVIOUS
from .plan import RUN


def publish(repo, output, context, ssh_config):
    old_run, old_remote = base.RUN, base.REMOTE
    base.RUN = RUN
    check = "previous=json.loads((project/'runs'/" + repr(PREVIOUS) + "/'pipeline_state.json').read_text())\n"
    check += "if previous['status'] not in ('ANALYZED','PARTIAL_TECHNICAL_FAILURE') or any(v.get('status')=='RUNNING' for v in previous['jobs'].values()):raise ValueError('Fixed reference-budget owner still active')\n"
    base.REMOTE = old_remote.replace('release.mkdir()', check + 'release.mkdir()').replace(
        "str(release/'experiments/pamf_branch_training')",
        "str(release/'experiments/pamf_physics_branch_adaptive'),str(release/'experiments/pamf_physics_branch')").replace(
        'experiments.pamf_branch_training.runner', 'experiments.pamf_physics_branch_adaptive.runner')
    try:
        base.publish(repo, output, context, ssh_config, launch_owner='codex-pamf-adaptive-20261110')
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh_config'):
        parser.add_argument('--' + option.replace('_', '-'), required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config)
