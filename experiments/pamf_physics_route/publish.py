"""Publish the practical-only candidate using one verified release archive."""
import argparse
from experiments.pamf_branch_training import publish as base
from .plan import RUN


def publish(repo, output, context, ssh_config):
    old_run, old_remote = base.RUN, base.REMOTE
    base.RUN = RUN
    base.REMOTE = (old_remote.replace("rt=p['runtime']; project=", "rt=p['runtime'];rt['allowed_gpus']=[1]; project=")
        .replace("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]",
                 "[rt['python'],*rt['frozen_sources'].values()]")
        .replace("str(release/'experiments/pamf_branch_training')",
                 "str(release/'experiments/pamf_physics_route'),str(release/'experiments/pamf_physics_branch')")
        .replace('experiments.pamf_branch_training.runner', 'experiments.pamf_physics_route.runner'))
    try:
        base.publish(repo, output, context, ssh_config, launch_owner='codex-pamf-route-20261110')
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('repo', 'output', 'context', 'ssh_config'):
        p.add_argument('--' + key.replace('_', '-'), required=True)
    a = p.parse_args()
    publish(a.repo, a.output, a.context, a.ssh_config)
