"""Publish the fixed three-seed Tri-SIG-TIME physical-L continuation."""
import argparse

from experiments.pamf_branch_training import publish as base
from experiments.pamf_repl_view_shuffle.plan import RUN as TRI_RUN
from .plan import RUN, SEEDS


def publish(repo, output, context, ssh_config):
    old_run, old_remote = base.RUN, base.REMOTE
    base.RUN = RUN
    preflight = (
        f"for seed in {SEEDS!r}:\n"
        f" path=project/'runs'/{TRI_RUN!r}/'F'/f'main_split-repl-shuffled_time-3view-s{{seed}}'/'source'/'segment20'\n"
        " if not (path/'final_ssdg.pth').is_file() or not (path/'complete.json').is_file():raise FileNotFoundError(path)\n"
    )
    base.REMOTE = (old_remote.replace("rt=p['runtime']; project=", "rt=p['runtime']; rt['parent_seeds']=[]; project=")
        .replace("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]",
                 "[rt['python'],*rt['frozen_sources'].values()]")
        .replace("for v in ctx['caches']['F0'].values():", preflight + "for v in ctx['caches']['F0'].values():")
        .replace("str(release/'experiments/pamf_branch_training')",
                 "str(release/'experiments/pamf_physics_tri'),str(release/'experiments/pamf_physics_branch')")
        .replace('experiments.pamf_branch_training.runner', 'experiments.pamf_physics_tri.runner'))
    try:
        base.publish(repo, output, context, ssh_config, launch_owner='codex-pamf-tri-label-tangent-20261110')
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('repo', 'output', 'context', 'ssh_config'):
        p.add_argument('--' + key.replace('_', '-'), required=True)
    a = p.parse_args()
    publish(a.repo, a.output, a.context, a.ssh_config)
