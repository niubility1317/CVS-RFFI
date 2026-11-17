"""Transfer a reviewed dispatcher without changing the original worker release."""
import argparse
from unittest.mock import patch
from experiments.pamf_branch_training import publish as base
from experiments.pamf_repl_attribution_factorial import publish as previous

RUN='20261117-phase1-pamf-priority-dispatch-takeover-r01'
ORIGINAL='20261117-phase1-pamf-repl-priority-clean-m20-r01'


def publish(repo,output,context,ssh_config):
    old_run,old_remote=previous.RUN,base.REMOTE;original=base.publish
    def submit(*args,**kwargs):
        marker="rt['allowed_gpus']=[2,3]; project="
        if marker not in base.REMOTE:raise ValueError('Expected release template missing')
        replacement=("rt['allowed_gpus']=[0,1,2,3,5,7]; "
            f"rt['original_runtime_file']=rt['project']+'/releases/{ORIGINAL}-release/runtime.json'; project=")
        base.REMOTE=base.REMOTE.replace(marker,replacement)
        base.REMOTE=base.REMOTE.replace('experiments.pamf_repl_attribution_factorial.runner',
                                       'experiments.pamf_repl_priority_takeover.runner')
        base.REMOTE=base.REMOTE.replace("str(release/'experiments/pamf_repl_attribution_factorial')",
                                       "str(release/'experiments/pamf_repl_priority_takeover')")
        return original(*args,**kwargs)
    previous.RUN=RUN
    try:
        with patch.object(base,'publish',submit):return previous.publish(repo,output,context,ssh_config)
    finally:previous.RUN,base.REMOTE=old_run,old_remote


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','output','context','ssh-config'):p.add_argument('--'+name,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,a.ssh_config)
