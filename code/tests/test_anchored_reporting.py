import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_reporting import count_changes, class_flow, risk_coverage, profile_stages,assess_source_promotion


def test_known_counts_and_class_flow():
    y=torch.tensor([0,0,1,1]);base=torch.tensor([0,1,1,0]);pred=torch.tensor([1,0,1,0])
    stats=count_changes(y,base,pred)
    assert stats['rescue']==1 and stats['harm']==1 and stats['net_correct']==0 and stats['utility']==-1
    flow=class_flow(y,pred,2)
    assert flow[0]['precision']==.5 and flow[1]['recall']==.5
    assert sum(r['inflow_errors'] for r in flow)==2


def test_risk_curve_keeps_ties_and_profile_is_measured():
    curve=risk_coverage(torch.tensor([.9,.9,.5]),torch.tensor([True,False,True]))
    assert len(curve)==2 and abs(curve[0]['coverage']-2/3)<1e-8
    result=profile_stages({'head':lambda:torch.ones(2).sum()},warmup=1,repeats=2)
    assert result['stages']['head']['repeats']==2 and result['stages']['head']['seconds_mean']>=0


def test_source_promotion_requires_complete_three_seed_group_evidence():
    records=[dict(candidate='A6',group='all',value='all',weighted_net=.01)]
    records += [dict(candidate='A6',group='RX',value=str(rx),weighted_net=.01) for rx in (1,3,4,6,8)]
    records += [dict(candidate='A6',group='TX',value=str(y),weighted_net=.01) for y in range(6)]
    records += [dict(candidate='A6',group='view',value='clean',weighted_net=0.)]
    assert not assess_source_promotion({392005:records})['passed']
    result=assess_source_promotion({seed:records for seed in (392005,392006,392007)})
    assert result['passed'] and not result['confirmation']
    bad=[dict(r,weighted_net=-.006) if r['group']=='TX' and r['value']=='2' else r for r in records]
    assert not assess_source_promotion({392005:records,392006:records,392007:bad})['passed']
