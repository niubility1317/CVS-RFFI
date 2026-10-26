import argparse
from .fonly_confirm import RUN,RELEASE,METHODS,validate_context
from .publish_vat_source import publish


def required(c):
    result=[*c['frozen_sources'].values(),c['source_contract']]
    for method in METHODS:
        p=c['source_run_root']+'/F/F0-'+method+'-s2026092701'
        result.extend([p+'/source/complete.json',p+'/source/final_ssdg.pth',p+'/source_validation/complete.json'])
    result.append(c['source_run_root']+'/baseline_source/complete.json')
    result.extend(v['views_root']+'/manifest.json' for v in c['tests'].values())
    result.extend(p+'/complete.json' for p in c['reference_predictions'].values())
    result.extend(c['reference_scores'].values());return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.fonly_confirm',context_validator=validate_context,required_builder=required)
