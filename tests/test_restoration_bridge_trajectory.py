import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.direct_bridge import scratch_direct_bridge
from experiments.pamf_phase3_restoration.state_bridge import scratch_state_bridge
from experiments.pamf_phase3_restoration.bridge_trajectory import trajectory,summarize,METHODS,METRICS


@pytest.mark.parametrize('factory',[scratch_direct_bridge,scratch_state_bridge])
def test_recorded_trajectory_exactly_matches_actual_deployment_and_first_estimate(factory):
    torch.set_num_threads(2);m=factory(33).eval().requires_grad_(False)
    with torch.no_grad():m.output.weight.normal_(std=.003)
    y=torch.randn(2,2,256);before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.inference_mode():
        q=trajectory(m,y);direct=m(y);first=m.estimate(y,y,torch.ones(2))
    assert len(q)==4 and [x[0] for x in q]==[1.,.75,.5,.25]
    assert torch.equal(q[-1][1],direct) and torch.equal(q[0][1],first)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def records():
    a={k:[] for k in ['method','step','epoch','source_index','scene','lock',*METRICS]}
    for method in METHODS:
        for step in range(1,5):
            for epoch in range(1,21):
                for index in range(512):
                    z=dict(method=method,step=step,epoch=epoch,source_index=index,scene='clean',lock='not_applicable')
                    z.update({k:0. for k in METRICS});z['clean_decision_agreement']=float(index>=2 or step>=2)
                    for k,v in z.items():a[k].append(v)
    return {k:np.asarray(v) for k,v in a.items()}


def test_summary_checks_same_source_cases_and_repairs_against_first_step():
    a=records();q=summarize(a)
    last=next(x for x in q if x['method']==METHODS[0] and x['step']==4 and x['dimension']=='overall')
    assert last['count']==10240 and last['paired_against_first']==dict(repairs=40,damage=0,net=40)
    mask=np.ones(len(a['step']),bool);mask[-1]=False
    with pytest.raises(ValueError):summarize({k:v[mask] for k,v in a.items()})
