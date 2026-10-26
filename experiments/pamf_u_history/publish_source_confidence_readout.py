import argparse
from .source_confidence_readout import RUN,RELEASE,SEEDS,MEAN,validate_context,source_folder
from .publish_vat_source import publish


def required(c):
    values=[*c['frozen_sources'].values(),c['channel_code'],c['source_contract'],c['source_cache'],c['project']+'/runs/'+MEAN]
    for method in ('SIG1-PL','SIG1-Sup'):
        for seed in SEEDS:values.extend([source_folder(c,method,seed)+'/complete.json',source_folder(c,method,seed)+'/final_ssdg.pth'])
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.source_confidence_readout',
        context_validator=validate_context,required_builder=required,runtime_extra=dict(cpu_only=True))
