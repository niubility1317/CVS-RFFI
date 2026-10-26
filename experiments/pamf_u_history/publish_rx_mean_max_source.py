import argparse
import json
from pathlib import Path
from .rx_mean_max_source import RUN,RELEASE,PREDECESSOR,SEEDS,validate_context
from .publish_vat_source import publish
from .covariance_queue import POLICY
from .publish_rx_mean_source import required as mean_required


def required(c):
    values=mean_required(c);p=c['project']
    values.extend([p+'/runs/'+PREDECESSOR,p+'/runs/'+PREDECESSOR+'/mean_bank/bank.npz',p+'/runs/'+PREDECESSOR+'/mean_bank/complete.json'])
    # The predecessor complete.json is a future resource dependency; requiring
    # it at publication would prevent fixing this matrix before its tests end.
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();c=json.loads(Path(a.context).read_text(encoding='utf-8'))
    publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.rx_mean_max_source',
        context_validator=validate_context,required_builder=required,runtime_extra=dict(cpu_only=True,queue_policy=POLICY,
            evaluation_context=c['project']+'/releases/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-release-r03/context.json'))
