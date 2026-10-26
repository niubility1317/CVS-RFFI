import argparse
from experiments.pamf_channel_pl.sig1_matched_baselines import prefix
from .bounded_response_source import RUN,RELEASE,SEEDS,validate_context
from .publish_vat_source import publish


def required(c):
    result=[c['source_cache'],c['source_contract'],c['channel_code'],*c['frozen_sources'].values()]
    for seed in SEEDS:result.extend([prefix(c['project'],seed)+'/complete.json',prefix(c['project'],seed)+'/final_ssdg.pth'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.bounded_response_source',context_validator=validate_context,required_builder=required,runtime_extra=dict(cpu_only=True))
