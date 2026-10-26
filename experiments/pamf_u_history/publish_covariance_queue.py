import argparse
from .covariance_queue import RUN,RELEASE,POLICY,validate_context
from .publish_vat_source import publish


def required(c):
    root=c['project'];old=root+'/runs/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01'
    return [c['original_runtime'],old+'/jobs.json',old+'/pipeline_state.json']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.covariance_queue',
        context_validator=validate_context,required_builder=required,runtime_extra=dict(cpu_only=True,queue_policy=POLICY))
