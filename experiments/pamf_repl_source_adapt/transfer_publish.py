"""Publish the frozen source decisions to the clean-only transfer verification."""
import argparse

from experiments.pamf_branch_training import publish as base
from .transfer import DEVELOPMENT_RUN, RUN


def publish(repo, output, context, ssh_config, *, run_id=RUN,
            module='experiments.pamf_repl_source_adapt.transfer', decision_run=DEVELOPMENT_RUN):
    old_run, old_remote = base.RUN, base.REMOTE
    script = old_remote.replace("rt=p['runtime']; project=", "rt=p['runtime']; rt['allowed_gpus']=list(range(7)); rt.pop('weak_root'); rt.pop('weak_truth'); project=")
    start = script.index("for path in [rt['python']")
    end = script.index('gpu=subprocess', start)
    preflight = """for path in [rt['python'], *rt['frozen_sources'].values()]:
 if not Path(path).exists():raise FileNotFoundError(path)
"""
    if decision_run is not None:
        preflight += f"d=project/'runs'/{decision_run!r}/'analysis'/'source_decisions.json'\n"
        preflight += """decision=json.loads(d.read_text())
if decision['status']!='SOURCE_ADAPTATION_DECISIONS_FROZEN' or decision['target_access'] or decision['U_truth_read']:
 raise ValueError('Frozen source-only decisions required')
"""
    script = (script[:start] + preflight + script[end:]).replace('experiments/pamf_branch_training', 'experiments/pamf_repl_source_adapt')
    script = script.replace('experiments.pamf_branch_training.runner', module)
    base.RUN, base.REMOTE = run_id, script
    try:
        return base.publish(repo, output, context, ssh_config, launch_owner='codex-'+run_id)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--receiver-balance', action='store_true')
    variant.add_argument('--receiver-balance-confirm', action='store_true')
    args = parser.parse_args()
    if args.receiver_balance or args.receiver_balance_confirm:
        from .receiver_balance import RUN as balance_run, CONFIRM_RUN, MODULE

        run_id = CONFIRM_RUN if args.receiver_balance_confirm else balance_run
        publish(args.repo, args.output, args.context, args.ssh_config, run_id=run_id, module=MODULE, decision_run=None)
    else:
        publish(args.repo, args.output, args.context, args.ssh_config)
