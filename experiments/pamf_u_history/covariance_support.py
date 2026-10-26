"""Explicit natural-support candidate from saved L statistics; zero new inference."""
import argparse
import json
from pathlib import Path
import torch
from .selective_covariance import MaskStatistics
from .covariance_fit import RUN as FIT_RUN,RECIPE,LEVELS

RUN='20261026-analysis-PAMF-SIG1-CovSupport-L-s2026092701-r01'


def derive(statistics_path,result_path,output,code_commit):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'masks.pt').exists() or (out/'result.json').exists():raise FileExistsError('Immutable source mask derivation output exists')
    r=json.loads(Path(result_path).read_text(encoding='utf-8'))
    payload=torch.load(statistics_path,map_location='cpu',weights_only=True)
    if (r['status']!='SOURCE_MASK_FIT_COMPLETE' or r['run_id']!=FIT_RUN or r['recipe']!=RECIPE
        or payload['schema']!='pamf_covariance_L_statistics_v1' or payload['code_commit']!=r['code_commit']
        or r['source_L_seen']!=6300 or r['As_physical_views']!=18900 or not r['parameters_and_registered_buffers_unchanged']
        or any(r[k] for k in ('F_optimizer_updates','G_weight_reads','G_updates','U_IQ_reads','U_truth_read','V_reads','target_reads'))):
        raise ValueError('Completed original L-only statistics and matching fit result required')
    contract=payload['source_contract']
    if (contract['counts']!=dict(L_s=6300,U_s=56700,V=27000) or contract['source_rxs']!=[1,3,4,6,8]
        or contract['source_days']!=[1,2,3] or set(payload['statistics'])!=set(LEVELS)):
        raise ValueError('Original fixed physical source roles required')
    masks={};diagnostics={}
    for level in LEVELS:
        saved=payload['statistics'][level];s=MaskStatistics(saved['channels'])
        for name in ('count','total','square','sensitivity','rx_count'):
            value=saved[name]
            if value.shape!=getattr(s,name).shape or value.dtype!=torch.float64 or not torch.isfinite(value).all():
                raise ValueError('Complete finite original source moments required')
            setattr(s,name,value)
        s.samples=saved['samples'];s.paired_views=saved['paired_views']
        if s.samples!=6300 or s.paired_views!=18900 or not bool((s.count==70).all()) or not bool((s.rx_count==1260).all()):
            raise ValueError('All90 source cells from the completed fit required')
        masks[level],diagnostics[level]=s.masks(fraction=.1,random_seed=2026102623,support_limited=True)
    result=dict(status='NATURAL_SUPPORT_MASKS_DERIVED',run_id=RUN,code_commit=code_commit,input_fit_run=FIT_RUN,
        input_fit_commit=r['code_commit'],actual_checkpoint_epoch=20,actual_checkpoint_updates=4440,
        selection='same top25percent sensitivity/bottom50percent conditionalTX association;up to10percent edges;no ineligible fill',
        diagnostics=diagnostics,new_F_forwards=0,new_IQ_reads=0,F_training_updates=0,G_reads=0,U_truth_reads=0,V_reads=0,target_reads=0,
        empirical_performance='UNTESTED',scope='independent explicitly revised source-support rule;original strict10percent failed result unchanged')
    with (out/'masks.pt').open('xb') as f:torch.save(dict(schema='pamf_sig1_selective_covariance_masks_v2_natural_support',
        code_commit=code_commit,source_contract=contract,source_prefix=r['source_prefix'],recipe=dict(RECIPE,selection_mode='natural_support_capped'),
        masks=masks,diagnostics=diagnostics,fit_role='L_s',U_truth_read=False,V_used=False,target_used=False),f)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--statistics',required=True);p.add_argument('--fit-result',required=True);p.add_argument('--output',required=True);p.add_argument('--code-commit',required=True)
    a=p.parse_args();derive(a.statistics,a.fit_result,a.output,a.code_commit)
