import argparse
from .covariance_fit import RUN,RELEASE,RECIPE
from .publish_vat_source import publish


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args()
    publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.covariance_fit',runtime_extra=dict(recipe=RECIPE))
