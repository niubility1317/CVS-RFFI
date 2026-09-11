import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi.cross_response.decision import decision_margin_loss_reference, decision_margin_loss_vectorized
from cvsrffi.cross_response.decision_calibration import (DecisionCalibrationConfig,estimate_source_noise,
    select_source_competitors,calibrated_decision_tensors)
from cvsrffi.cross_response.schema import SampleRecord


@pytest.mark.parametrize('receivers',[1,2,3,4,5])
@pytest.mark.parametrize('amp',[False,True])
def test_vectorized_loss_and_gradient_match_reference(receivers,amp):
    g=torch.Generator().manual_seed(18)
    n=receivers*8
    labels=torch.arange(n)%3
    x=torch.randn(n,3,generator=g)
    x[torch.arange(n),labels]+=2
    rx=[i//8 for i in range(n)]
    physical=[i if i%11 else 'duplicate' for i in range(n)]
    cond=[i%2 for i in range(n)]
    args=(labels,rx,physical,cond)
    if amp:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            x=x @ torch.eye(3)
    a=x.clone().requires_grad_();b=x.clone().requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        loss_a,da=decision_margin_loss_reference(a,*args,min_records=1)
        loss_b,db=decision_margin_loss_vectorized(b,*args,min_records=1)
    loss_a.backward();loss_b.backward()
    tolerance = 0 if amp else 2e-6
    torch.testing.assert_close(da['reference'],db['reference'],rtol=tolerance,atol=tolerance)
    assert torch.equal(da['valid_mask'],db['valid_mask'])
    torch.testing.assert_close(loss_a,loss_b,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(a.grad,b.grad,rtol=tolerance,atol=tolerance)


def test_midpoint_and_unavailable_noise_detached_weights():
    logits=torch.tensor([[1.,0.],[3.,0.],[5.,0.]],requires_grad=True)
    args=([0]*3,[0,1,2],[0,1,2],['clean']*3)
    _,d=decision_margin_loss_vectorized(logits,*args,min_records=1)
    assert d['reference'][0,1]==4
    delta=torch.zeros(3,2,requires_grad=True)
    weight=torch.ones(3,2,requires_grad=True)
    loss,d=decision_margin_loss_vectorized(logits,*args,min_records=1,delta=delta,pair_weights=weight)
    loss.backward()
    assert delta.grad is None and weight.grad is None
    assert d['positive_gap_fraction_valid']==pytest.approx(1/3)
    loss,d=decision_margin_loss_vectorized(logits,*args,min_records=1,delta=float('nan'))
    assert loss==0 and d['valid_comparisons']==0


def calibration_fixture():
    records=[SampleRecord(i,0,i//8,0,'clean',i) for i in range(16)]
    logits=torch.tensor([[float(1+i%4),0.,-.5] for i in range(16)])
    labels=[0]*16
    cfg=DecisionCalibrationConfig(quantile=.75,min_groups=2,group_size=2,max_delta=4,
                                 top_k=1,min_pair_support=4,boundary_margin=2)
    return records,logits,labels,cfg


def test_source_calibration_unfrozen_role_and_physical_safety():
    r,x,y,c=calibration_fixture()
    with pytest.raises(ValueError,match='UNFROZEN'):
        DecisionCalibrationConfig().validate()
    with pytest.raises(ValueError,match='L_s'):
        estimate_source_noise(x,y,r,c,source_role='target')
    with pytest.raises(ValueError,match='distinct physical'):
        estimate_source_noise(x,y,r[:15]+r[:1],c,source_role='L_s')
    noise=estimate_source_noise(x,y,r,c,source_role='L_s')
    pairs=select_source_competitors(x,y,r,c,source_role='V')
    assert noise['global_noise']['groups']==4
    assert noise['used_physical_records']==16
    for group in noise['matched_groups']:
        assert not set(group['group_a']) & set(group['group_b'])
        assert len(group['group_a'])==len(group['group_b'])==2
    with pytest.raises(ValueError,match='UNFROZEN'):
        calibrated_decision_tensors(noise,pairs,[0],[0],['clean'])
    noise['source_frozen']=pairs['source_frozen']=True
    delta,weights,diag=calibrated_decision_tensors(noise,pairs,[0],[99],['unseen'])
    assert weights.sum()==1 and diag['fallbacks'][(0,1)]['reason']=='pair'
    assert delta[0,1]>0


def test_sparse_noise_never_becomes_confident_zero():
    r,x,y,c=calibration_fixture()
    noise=estimate_source_noise(x[:2],y[:2],r[:2],c,source_role='L_s')
    pairs=select_source_competitors(x,y,r,c,source_role='V')
    noise['source_frozen']=pairs['source_frozen']=True
    delta,weights,diag=calibrated_decision_tensors(noise,pairs,[0],[0],['clean'])
    assert torch.isnan(delta).all() and weights.sum()==0
    assert diag['fallbacks'][(0,1)]['reason']=='insufficient_global_noise_evidence'


def test_delta_only_protects_all_competitors_without_v_selection():
    r,x,y,c=calibration_fixture()
    noise=estimate_source_noise(x,y,r,c,source_role='L_s')
    noise['source_frozen']=True
    delta,weights,diag=calibrated_decision_tensors(noise,None,[0],[0],['clean'])
    assert weights.sum()==2 and diag['source_roles']==('L_s',)


def test_invalid_references_wrong_predictions_zero_margins_and_minimum_support():
    logits=torch.tensor([[1.,0.,0.],[0.,2.,0.],[0.,0.,0.],[3.,0.,0.],[5.,0.,0.]],requires_grad=True)
    args=([0]*5,[0,1,2,3,3],['a','b','c','d','e'],['c']*5)
    for count in (1,2,3):
        a,da=decision_margin_loss_reference(logits,*args,min_records=count)
        b,db=decision_margin_loss_vectorized(logits,*args,min_records=count)
        torch.testing.assert_close(a,b)
        assert torch.equal(da['valid_mask'],db['valid_mask'])


if __name__=='__main__':
    import time,json
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(42)
    x=torch.randn(128,6,generator=g)
    y=torch.arange(128)%6
    x[torch.arange(128),y]+=2
    args=(y,(torch.arange(128)%5).tolist(),list(range(128)),['clean']*128)
    results={}
    for name,fn in [('reference',decision_margin_loss_reference),('vectorized',decision_margin_loss_vectorized)]:
        for _ in range(3):
            fn(x,*args)
        start=time.perf_counter()
        for _ in range(10):
            fn(x,*args)
        results[name+'_ms']=(time.perf_counter()-start)*100
    print(json.dumps(dict(results,device='cpu',threads=1,shape=[128,6],warmup=3,repeats=10,scope='forward and all diagnostics; no backward'),sort_keys=True))
