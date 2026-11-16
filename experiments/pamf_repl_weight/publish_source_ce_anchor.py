"""Publish the source-only CE anchor diagnostic using the existing CPU owner."""
import argparse
from pathlib import Path
from . import publish_source_adamw as base


RUN = '20261116-diagnostic-pamf-tri-ce-anchor-e20-s2026092701-r01'


def publish(repo, output, ssh_config):
    Path(output).mkdir(parents=True, exist_ok=True)
    old_run, old_remote = base.RUN, base.REMOTE
    before = 'experiments.pamf_repl_weight.source_adamw_diagnostic'
    if before not in old_remote:
        raise ValueError('Source diagnostic release template differs')
    base.RUN = RUN
    base.REMOTE = old_remote.replace(before, 'experiments.pamf_repl_weight.source_ce_anchor')
    try:
        return base.publish(repo, output, ssh_config)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ssh-config', required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.ssh_config)
