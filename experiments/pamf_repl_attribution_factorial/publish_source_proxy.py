"""Publish the fixed source-only CE-head proxy screen on CPU."""
import argparse
from pathlib import Path
from experiments.pamf_repl_weight import publish_source_adamw as base

RUN='20261116-diagnostic-pamf-repl-source-ce-proxy-m16-r01'


def publish(repo,output,ssh_config):
    Path(output).mkdir(parents=True,exist_ok=True)
    old_run,old_remote=base.RUN,base.REMOTE
    before='experiments.pamf_repl_weight.source_adamw_diagnostic'
    compile_before="str(release/'experiments/pamf_repl_weight')"
    if before not in old_remote or compile_before not in old_remote:
        raise ValueError('Source diagnostic release template differs')
    base.RUN=RUN
    base.REMOTE=old_remote.replace(before,'experiments.pamf_repl_attribution_factorial.source_proxy_diagnostic').replace(
        compile_before,compile_before+",str(release/'experiments/pamf_repl_attribution_factorial')")
    try:return base.publish(repo,output,ssh_config)
    finally:base.RUN,base.REMOTE=old_run,old_remote


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','output','ssh-config'):p.add_argument('--'+name,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.ssh_config)
