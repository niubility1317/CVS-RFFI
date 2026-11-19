"""Publish the frozen source decisions to the clean-only transfer verification."""
import argparse

from experiments.pamf_branch_training import publish as base
from .transfer import DEVELOPMENT_RUN, RUN


def publish(repo, output, context, ssh_config):
    old_run, old_remote = base.RUN, base.REMOTE
    script = old_remote.replace("rt=p['runtime']; project=", "rt=p['runtime']; rt['allowed_gpus']=list(range(7)); rt.pop('weak_root'); rt.pop('weak_truth'); project=")
    start = script.index("for path in [rt['python']")
    end = script.index('gpu=subprocess', start)
    source = f"d=project/'runs'/{DEVELOPMENT_RUN!r}/'analysis'/'source_decisions.json'\n"
    preflight = source + """for path in [rt['python'], *rt['frozen_sources'].values(), d]:
 if not Path(path).exists():raise FileNotFoundError(path)
decision=json.loads(d.read_text())
if decision['status']!='SOURCE_ADAPTATION_DECISIONS_FROZEN' or decision['target_access'] or decision['U_truth_read']:
 raise ValueError('Frozen source-only decisions required')
"""
    script = (script[:start] + preflight + script[end:]).replace('experiments/pamf_branch_training', 'experiments/pamf_repl_source_adapt')
    script = script.replace('experiments.pamf_branch_training.runner', 'experiments.pamf_repl_source_adapt.transfer')
    base.RUN, base.REMOTE = RUN, script
    try:
        return base.publish(repo, output, context, ssh_config, launch_owner='codex-'+RUN)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config)
