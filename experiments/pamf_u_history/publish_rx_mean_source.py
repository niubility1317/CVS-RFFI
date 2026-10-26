import argparse
import json
from pathlib import Path
from .rx_mean_source import RUN,RELEASE,SEEDS,BASELINE,REUSE_BANK_RUN,validate_context
from .covariance_queue import POLICY
from .publish_vat_source import publish
from experiments.pamf_channel_pl.sig1_matched_baselines import prefix


def required(c):
    p=c['project'];result=[*c['frozen_sources'].values(),c['channel_code'],c['source_contract'],c['source_cache'],
        p+'/releases/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-release-r03/context.json',
        p+'/runs/20261026-diagnostic-PAMF-URealizations-cache-m3-r01/complete.json',
        p+'/runs/'+REUSE_BANK_RUN+'/mean_bank/complete.json',p+'/runs/'+REUSE_BANK_RUN+'/mean_bank/bank.npz']
    for seed in SEEDS:
        result.extend([prefix(p,seed)+'/complete.json',prefix(p,seed)+'/final_ssdg.pth'])
        for method in ('SIG1-PL','SIG1-Sup'):result.append(p+'/runs/'+BASELINE+'/readouts/F0-'+method+'-s'+str(seed)+'/scores/complete.json')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();c=json.loads(Path(a.context).read_text(encoding='utf-8'))
    publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.rx_mean_source',
        context_validator=validate_context,required_builder=required,runtime_extra=dict(cpu_only=True,queue_policy=POLICY,
            evaluation_context=c['project']+'/releases/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-release-r03/context.json'))
