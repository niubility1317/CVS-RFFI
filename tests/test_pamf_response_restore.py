from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from experiments.pamf_response_restore import design as d,models as m,runtime,evaluation as e,response_evaluation as re
from experiments.pamf_restoration.models import pair,complex_iq
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_action_restore.models import ActionRestorer,SCALE
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_uncertainty_restore import design as original


def sample(count=4):return torch.randn(count,2,256,generator=torch.Generator().manual_seed(34))


def identity(count):
    ones=torch.ones(count,256,dtype=torch.complex64)
    return Action(torch.ones(count,1,dtype=torch.complex64),ones,torch.zeros_like(ones))


def parent():
    return dict(coarse_run=original.COARSE_RUN,coarse_commit=original.COARSE_COMMIT,
        action_run=original.ACTION_RUN,action_commit=original.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def test_capacity_lock_matches_existing_cross_owner_scheduler():
    from experiments.pamf_unrolled_restore import design as unrolled
    from experiments.pamf_rf_response import design as response
    assert d.configuration()['capacity_lock']==unrolled.configuration()['capacity_lock']==response.configuration()['capacity_lock']=='receiver_residual_capacity4.lock'


def test_direction_sampling_is_batched_and_epoch_deterministic():
    x=sample();delta,valid,codes=m.training_delta(x,[19,33,4,81],1)
    a,ok,c=m.training_delta(x[[3,1]],[81,33],1)
    torch.testing.assert_close(a,delta[[3,1]],atol=0,rtol=0);assert torch.equal(ok,valid[[3,1]])
    np.testing.assert_array_equal(c,codes[[3,1]])
    assert not np.array_equal(codes,m.training_delta(x,[19,33,4,81],2)[2])
    norm=delta.abs().square().sum(-1)/complex_iq(x).abs().square().sum(-1)
    expected=torch.tensor(np.where(codes[:,1]==0,.0001,.0004),dtype=norm.dtype)
    torch.testing.assert_close(norm,expected)


def test_training_random_direction_stream_is_disjoint_from_validation():
    from experiments.pamf_rf_response.models import directions
    index=next(i for i in range(100) if tuple(np.random.default_rng(np.random.SeedSequence([d.RESPONSE_SEED,1,i]))
                    .integers([0,0],[len(d.DIRECTIONS),2]))==(3,0))
    x=sample(1);actual,_,_=m.training_delta(x,[index],1)
    source=directions(x,[index],.01)['random_control'][0]
    expected=directions(x,[index+(1<<32)],.01)['random_control'][0]
    assert not torch.allclose(actual,source)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.parametrize('epoch',[0,21,-1])
def test_sampling_outside_budget_rejected(epoch):
    with pytest.raises(ValueError):m.training_delta(sample(),[0,1,2,3],epoch)


def test_teacher_is_noise_regularized_not_forced_unit_gain():
    x=sample();delta,valid,_=m.training_delta(x,[0,1,2,3],1)
    variance=torch.full((4,),1e-4)
    change,target,eligible,info=m.teacher_response(identity(4),delta,variance,valid)
    expected=delta*SCALE**2/(SCALE**2+variance[:,None])
    torch.testing.assert_close(target,expected,atol=1e-7,rtol=1e-5)
    assert not target.requires_grad and not eligible.requires_grad and eligible.all()
    assert not torch.equal(target,delta)
    _,_,unobservable,_=m.teacher_response(identity(4),delta,torch.ones(4),valid)
    assert not unobservable.any()


def test_masked_response_uses_fullbatch_mean_and_empty_mask_has_zero_grad():
    x=sample();delta=complex_iq(x)*.01;target=delta*.7
    plus=pair(delta).clone().requires_grad_();minus=(-pair(delta)).clone().requires_grad_()
    loss,error=m.response_objective(plus,minus,target,delta,torch.tensor([True,False,False,False]))
    torch.testing.assert_close(loss.mean(),error[0]/4)
    loss.mean().backward();assert plus.grad[0].abs().sum()>0 and plus.grad[1:].abs().sum()==0
    plus.grad=None;minus.grad=None
    empty,_=m.response_objective(plus,minus,target,delta,torch.zeros(4,dtype=torch.bool))
    empty.mean().backward();assert plus.grad.abs().sum()==0 and minus.grad.abs().sum()==0


def test_attached_teacher_rejected():
    x=sample();delta=complex_iq(x)*.01
    with pytest.raises(ValueError):m.response_objective(x,x,delta.clone().requires_grad_(),delta,torch.ones(4,dtype=torch.bool))


def test_three_paired_targets_are_correct_for_a_linear_restorer():
    x=sample();delta=complex_iq(x)*.01
    loss=m.paired_objective(x,x+pair(delta),x-pair(delta),x,delta)
    assert torch.count_nonzero(loss)==0


def test_microbatch_accumulation_matches_fullbatch_gradient():
    x=sample(7);parameter=torch.tensor(.6,requires_grad=True);target=complex_iq(x)*.007;delta=complex_iq(x)*.01
    def objective(a,b):
        xx=x[a:b];dd=delta[a:b];mask=torch.tensor([True,False,True,False,True,True,False])[a:b]
        plus=(xx+pair(dd))*parameter;minus=(xx-pair(dd))*parameter
        return m.paired_objective(xx*parameter,plus,minus,xx,dd)+.01*m.response_objective(plus,minus,target[a:b],dd,mask)[0]
    objective(0,7).mean().backward();full=parameter.grad.clone();parameter.grad=None
    for begin,end in [(0,3),(3,6),(6,7)]: (objective(begin,end).sum()/7).backward()
    torch.testing.assert_close(parameter.grad,full,atol=1e-6,rtol=1e-5)


def test_matched_models_have_same_full_state_and_three_priors():
    torch.manual_seed(77);one=m.ResponseRestorer(d.ROWS[0],ActionRestorer(ComplexTFUNet()))
    torch.manual_seed(77);two=m.ResponseRestorer(d.CANDIDATE,ActionRestorer(ComplexTFUNet()))
    assert len(one.prior)==len(two.prior)==3
    assert all(torch.equal(v,two.state_dict()[k]) for k,v in one.state_dict().items())
    assert all(not p.requires_grad for p in one.frozen.parameters())


def test_new_checkpoint_roundtrip_and_permission_negatives(tmp_path):
    model=m.ResponseRestorer(d.CANDIDATE,ActionRestorer(ComplexTFUNet()))
    path=tmp_path/'initial.pt';runtime.save_checkpoint(path,model,None,0,0,parent())
    fresh,payload=runtime.load_checkpoint(path,'cpu')
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    for key,value in [('schema','unrolled_prior_checkpoint_v1'),('training_F_access',True),('training_TX_labels_read',1),('target_training',True)]:
        wrong=copy.copy(payload);wrong[key]=value
        with pytest.raises(ValueError):runtime.validate_payload(wrong)
    wrong=copy.deepcopy(payload);wrong['parent']['data_contract']='differentphysicalroles'
    with pytest.raises(ValueError):runtime.validate_payload(wrong)


def response_summary(candidate,error=.5,count=180):
    return dict(status='ANALYZED',run=d.RUN,candidate=candidate,count=1800,all_predictions_before_truth=True,
        training_TX_F_access=False,target_access=False,source_V_state_updates=0,
        records=[dict(view='practical_low_urban',direction=direction,dimension='overall',value='all',observable_count=count,
                      observable_oracle_response_error_mean=error if count else None) for direction in d.DIRECTIONS[:-1]])


def test_response_priority_requires_improvement_and_finite_source_stats():
    control=response_summary(d.ROWS[0]);candidate=response_summary(d.CANDIDATE,.4)
    result=e.response_comparison(control,candidate)
    assert result['qualified'] and result['ratio']==pytest.approx(.8)
    assert not e.response_comparison(control,response_summary(d.CANDIDATE,.51))['qualified']
    wrong=response_summary(d.CANDIDATE,float('nan'))
    with pytest.raises(ValueError):e.response_comparison(control,wrong)


@pytest.mark.parametrize('count',[0,99])
def test_insufficient_observable_records_is_scientific_nonqualification(count):
    result=e.response_comparison(response_summary(d.ROWS[0],.5,count),response_summary(d.CANDIDATE,.4,count))
    assert not result['qualified']


def test_response_eligibility_must_match_between_rows():
    with pytest.raises(ValueError):e.response_comparison(response_summary(d.ROWS[0],.5,180),response_summary(d.CANDIDATE,.4,181))


@pytest.mark.parametrize('field,value',[('cells',11),('candidate','other'),('TX_truth_reads',1),('target_reads',1),('F_access',1)])
def test_incomplete_or_illegal_source_response_rejected(field,value):
    record=dict(status='SOURCE_RESPONSE_COMPLETE',run=d.RUN,candidate=d.CANDIDATE,count=1800,
        views=list(d.RESPONSE_VIEWS),directions=list(d.DIRECTIONS),amplitude=.01,cells=12,
        TX_truth_reads=0,F_access=0,target_reads=0,optimizer_updates=0,state_updates=0)
    record[field]=value
    with pytest.raises(ValueError):re.complete_contract(record,d.CANDIDATE)


def test_response_completion_checked_before_any_truth_scoring(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'source_reference',lambda *a:(tmp_path,np.arange(1800),None,None))
    monkeypatch.setattr(re,'load_predictions',lambda *a:(_ for _ in ()).throw(ValueError('incomplete response')))
    monkeypatch.setattr(e,'original_score',lambda *a:pytest.fail('Source truth scored too early'))
    with pytest.raises(ValueError,match='incomplete response'):
        e.score(SimpleNamespace(output=tmp_path,row=d.CANDIDATE,parent_action=tmp_path,source_cache=tmp_path),'source')


@pytest.mark.parametrize('row,qualified,selected',[(d.ROWS[0],True,d.CANDIDATE),(d.CANDIDATE,False,None)])
def test_target_access_rejected_before_qualification(monkeypatch,tmp_path,row,qualified,selected):
    monkeypatch.setattr(e,'read_json',lambda *a:dict(status='SOURCE_DECISION_FROZEN',qualified=qualified,selected=selected,target_rows=[d.CANDIDATE] if qualified else []))
    monkeypatch.setattr(e,'write_json',lambda *a:pytest.fail('Target preparation should remain closed'))
    with pytest.raises(ValueError):e.prepare_target(SimpleNamespace(output=tmp_path,row=row))
