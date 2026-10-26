import argparse
from .covariance_three_seed import RUN,RELEASE,OLD_SOURCE,OLD_CONFIRM,BASELINE,validate_context
from .covariance_seed_masks import RUN as MASK_RUN,NEW_SEEDS
from .publish_vat_source import publish


def required(c):
    p=c['project'];paths=[*c['frozen_sources'].values(),c['channel_code'],c['source_contract'],c['source_cache']]
    for seed in NEW_SEEDS:paths.extend([p+'/runs/'+MASK_RUN+'/s'+str(seed)+'/masks.pt',p+'/runs/'+MASK_RUN+'/s'+str(seed)+'/result.json'])
    paths.append(p+'/releases/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-release-r03/context.json')
    for row in c['cases']:paths.extend([row['parent']['path']+'/complete.json',row['parent']['path']+'/final_ssdg.pth',row['completed_reference']+'/complete.json'])
    for method in ('CovSel-U','CovRandom-U','CovFull-U','CovSel-LU'):paths.append(p+'/runs/'+OLD_SOURCE+'/F/F0-'+method+'-s2026092701/source/final_ssdg.pth')
    for method in ('CovSel-U','CovRandom-U','CovSel-LU'):paths.append(p+'/runs/'+OLD_CONFIRM+'/readouts/F0-'+method+'-s2026092701/scores/complete.json')
    return paths


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args()
    import json
    from pathlib import Path
    ctx=json.loads(Path(a.context).read_text(encoding='utf-8'))
    publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.covariance_three_seed',context_validator=validate_context,required_builder=required,
        runtime_extra=dict(evaluation_context=ctx['project']+'/releases/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-release-r03/context.json'))
