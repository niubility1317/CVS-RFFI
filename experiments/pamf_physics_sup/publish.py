"""Publish the fixed three-seed PAMF-Sup physical continuation."""
import argparse

from experiments.pamf_branch_training import publish as base
from .plan import RUN, SEEDS


def publish(repo, output, context, ssh_config):
    old_run, old_remote = base.RUN, base.REMOTE
    base.RUN = RUN
    base.REMOTE = (old_remote.replace("rt=p['runtime']; project=",
        f"rt=p['runtime']; rt.update(allowed_gpus=[2,3,4], parent_seeds={list(SEEDS)!r}, parent_kinds=['F20']); project=")
        .replace("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]",
                 "[rt['python'],*rt['frozen_sources'].values()]")
        .replace("str(release/'experiments/pamf_branch_training')",
                 "str(release/'experiments/pamf_physics_sup'),str(release/'experiments/pamf_physics_tri')")
        .replace('experiments.pamf_branch_training.runner', 'experiments.pamf_physics_sup.runner'))
    try:
        base.publish(repo, output, context, ssh_config, launch_owner='codex-pamf-sup-label-tangent-20261110')
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('repo', 'output', 'context', 'ssh_config'):
        p.add_argument('--' + key.replace('_', '-'), required=True)
    a = p.parse_args()
    publish(a.repo, a.output, a.context, a.ssh_config)
