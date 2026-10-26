import argparse
from .covariance_seed_masks import RUN,RELEASE,validate_context
from .publish_vat_source import publish


def required(c):
    values=[*c['frozen_sources'].values(),c['channel_code'],c['source_contract'],c['source_cache'],c['seed01_masks']]
    for row in c['cases']:
        p=row['parent']['path'];values.extend([p+'/complete.json',p+'/final_ssdg.pth'])
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.covariance_seed_masks',context_validator=validate_context,required_builder=required)
