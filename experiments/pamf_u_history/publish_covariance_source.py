import argparse
from .covariance_source import RUN,RELEASE,FIT_RUN,BASELINE_RUN
from .publish_vat_source import publish


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args()
    required=['runs/'+FIT_RUN+'/statistics.pt','runs/'+FIT_RUN+'/result.json',
        'runs/20261026-diagnostic-PAMF-URealizations-cache-m3-r01/complete.json',
        'runs/'+BASELINE_RUN+'/F/F0-SIG1-PL-s2026092701/source/final_ssdg.pth']
    publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.covariance_source',required_relative=required)
