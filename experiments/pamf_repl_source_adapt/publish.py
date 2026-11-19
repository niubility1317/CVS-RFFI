"""Publish the source-only trial without target handles."""
import argparse
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_branch_training import publish as base
from .pilot import RUN, SETTINGS, paper


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    protocol_ids = [paper.PROTOCOLS[setting][0] for setting in SETTINGS]
    source_context = dict(
        runtime={key: ctx['runtime'][key] for key in ('project', 'python', 'frozen_sources')},
        handles={pid: {key: ctx['handles'][pid][key] for key in ('source_cache', 'source_contract', 'classes')}
                 for pid in protocol_ids},
        protocols={pid: {'source_rxs': ctx['protocols'][pid]['source_rxs']} for pid in protocol_ids},
        templates=ctx['templates'],
    )
    path = Path(output)/'source_context.json'
    write(path, source_context)
    old_run, old_remote = base.RUN, base.REMOTE
    script = old_remote.replace("rt=p['runtime']; project=", "rt=p['runtime']; rt['allowed_gpus']=list(range(7)); rt.pop('weak_root'); rt.pop('weak_truth'); project=")
    start = script.index('for path in [rt[\'python\']')
    end = script.index('gpu=subprocess', start)
    preflight = """for path in [rt['python'], *rt['frozen_sources'].values(),
             *(h[k] for h in ctx['handles'].values() for k in ('source_cache','source_contract'))]:
 if not Path(path).exists():raise FileNotFoundError(path)
"""
    script = (script[:start] + preflight + script[end:]).replace('experiments/pamf_branch_training', 'experiments/pamf_repl_source_adapt')
    script = script.replace('experiments.pamf_branch_training.runner', 'experiments.pamf_repl_source_adapt.runner')
    base.RUN, base.REMOTE = RUN, script
    try:
        return base.publish(repo, output, path, ssh_config, launch_owner='codex-'+RUN)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config)
