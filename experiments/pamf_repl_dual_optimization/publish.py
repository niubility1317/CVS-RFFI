"""Publish the dual-reference follow-up to the existing N607 executor."""
import argparse
from unittest.mock import patch
from experiments.pamf_branch_training import publish as base
from experiments.pamf_repl_attribution_factorial import publish as previous
from .plan import RUN

def publish(repo, output, context, ssh_config):
    old_run, old_remote = previous.RUN, base.REMOTE
    original = base.publish
    def submit(*args, **kwargs):
        base.REMOTE = base.REMOTE.replace("rt['allowed_gpus']=[2,3]", "rt['allowed_gpus']=[0,1,2,3,4,5,6,7]")
        base.REMOTE = base.REMOTE.replace('experiments.pamf_repl_attribution_factorial.runner',
                                           'experiments.pamf_repl_dual_optimization.runner')
        base.REMOTE = base.REMOTE.replace("str(release/'experiments/pamf_repl_attribution_factorial')",
                                           "str(release/'experiments/pamf_repl_dual_optimization')")
        return original(*args, **kwargs)
    previous.RUN = RUN
    try:
        with patch.object(base, 'publish', submit):
            return previous.publish(repo, output, context, ssh_config)
    finally:
        previous.RUN, base.REMOTE = old_run, old_remote

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('repo', 'output', 'context', 'ssh-config'): p.add_argument('--' + name, required=True)
    a = p.parse_args(); publish(a.repo, a.output, a.context, a.ssh_config)
