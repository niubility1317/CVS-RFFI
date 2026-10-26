import argparse
from .receiver_response_probe import RUN,RELEASE,validate_context
from .publish_vat_source import publish


def required(c):
    p=c['U_channel_cache']
    return [c['source_contract'],c['source_cache'],p+'/complete.json',p+'/fresh1/complete.json',p+'/fresh1/physical_index.npz',p+'/fresh1/channel_U.npy']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.receiver_response_probe',
        context_validator=validate_context,required_builder=required,runtime_extra=dict(cpu_only=True))
