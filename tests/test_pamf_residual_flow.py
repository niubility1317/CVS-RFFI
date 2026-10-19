import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_residual_flow import design as d,models as m,runner,target,evaluation
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_reception_restore.models import ComplexTFUNet

torch.set_num_threads(2)


def parent():
    p=ResidualDeterministic(ComplexTFUNet(),.1169459182403555)
    with torch.no_grad():p.correction.out.bias.fill_(.02)
    return p


@pytest.mark.parametrize('arm',d.ARMS)
def test_initial_full_output_exactly_parent_and_matching_state(arm):
    torch.manual_seed(5);p=parent();model=m.AnchoredFlow(p,arm);x=torch.randn(3,2,256)
    assert torch.equal(model(x),p(x))
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==d.TRAINABLE_PARAMETERS
    assert sum(v.numel() for v in model.parameters())==d.TOTAL_PARAMETERS
    model.train();assert not model.parent.training and not any(v.requires_grad for v in p.parameters())
    assert model.field.training


def test_two_arms_have_exact_same_initial_parent_and_field_state():
    p=parent();states=[]
    for arm in d.ARMS:
        torch.manual_seed(5);model=m.AnchoredFlow(p,arm);states.append(model.state_dict())
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())


def test_paired_state_endpoints_and_intermediate_are_correct():
    a=torch.randn(3,2,256);x=torch.randn_like(a);t=torch.tensor([0.,.25,1.])
    z=m.paired_state(a,x,t)
    assert torch.equal(z[0],a[0]) and torch.allclose(z[2],x[2],atol=5e-7)
    assert torch.allclose(z[1],.75*a[1]+.25*x[1],atol=3e-7)


@pytest.mark.parametrize('value',[-.1,1.01,float('nan'),float('inf')])
def test_forbidden_source_times_rejected(value):
    x=torch.randn(2,2,256)
    with pytest.raises(ValueError):m.paired_state(x,x,torch.tensor([value,.5]))


def test_control_has_no_midpath_clean_input():
    x=torch.randn(2,2,256);model=m.AnchoredFlow(parent(),d.ARMS[0])
    with pytest.raises(ValueError):model.training_prediction(x,x,torch.tensor([0.,.5]))


@pytest.mark.parametrize('arm',d.ARMS)
def test_actual_times_pair_source_and_cover_registered_distribution(arm):
    t=runner.sample_times(np.random.default_rng(d.TIME_SEED),63000,arm)
    assert np.array_equal(t[:63000],t[63000:]) and t.dtype==np.float32 and (t>=0).all() and (t<1).all()
    if arm==d.ARMS[0]:assert not np.any(t)
    else:assert .48<(t==0).mean()<.52 and .24<t.mean()<.26
    assert np.array_equal(t,runner.sample_times(np.random.default_rng(d.TIME_SEED),63000,arm))


@pytest.mark.parametrize('arm',d.ARMS)
def test_oracle_velocity_integrates_to_clean_and_has_declared_call_count(arm):
    x=torch.randn(3,2,256);model=m.AnchoredFlow(parent(),arm);a=model.anchor(x);delta=.1*torch.ones_like(x);calls=[]
    def field(received,anchor,point,time):
        assert torch.equal(received,x) and torch.equal(anchor,a)
        calls.append(time.clone());return delta
    model.velocity=field;y=model(x);steps=d.configuration(arm)['inference_steps']
    assert torch.allclose(y,a+delta,atol=5e-7) and len(calls)==steps
    assert [float(t[0]) for t in calls]==[i/steps for i in range(steps)]


@pytest.mark.parametrize('arm',d.ARMS)
def test_nonzero_field_is_current_frame_independent_and_deterministic(arm):
    torch.manual_seed(4);model=m.AnchoredFlow(parent(),arm).eval()
    with torch.no_grad():model.field.out.weight.normal_(0,.01)
    x=torch.randn(3,2,256);before={k:v.clone() for k,v in model.state_dict().items()};together=model(x)
    separate=torch.cat([model(v[None]) for v in x])
    assert torch.allclose(together,separate,atol=1e-6,rtol=1e-5) and torch.equal(together,model(x))
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize('arm',d.ARMS)
def test_velocity_training_lowers_displacement_error_and_never_updates_parent(arm):
    torch.manual_seed(8);model=m.AnchoredFlow(parent(),arm).train();y=torch.randn(4,2,256);x=y+.05
    times=torch.tensor(runner.sample_times(np.random.default_rng(d.TIME_SEED),2,arm));before={k:v.clone() for k,v in model.parent.state_dict().items()}
    opt=torch.optim.AdamW(model.field.parameters(),lr=2e-4);losses=[];encoder_received=False
    for _ in range(3):
        opt.zero_grad(set_to_none=True);estimate,error=model.training_prediction(y,x,times);loss=error.mean();loss.backward()
        assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.field.out.parameters())
        encoder_received|=any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.field.encoder.parameters())
        assert all(p.grad is None for p in model.parent.parameters());opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0] and encoder_received
    assert all(torch.equal(v,before[k]) for k,v in model.parent.state_dict().items())


@pytest.mark.parametrize('arm',d.ARMS)
def test_training_proxy_is_anchor_plus_velocity_with_normalized_error(arm):
    model=m.AnchoredFlow(parent(),arm);y=torch.randn(3,2,256);x=y+.07;a=model.anchor(y)
    times=torch.zeros(3) if arm==d.ARMS[0] else torch.tensor([0.,.2,.8]);velocity=torch.full_like(y,.02)
    model.velocity=lambda *args:velocity
    estimate,error=model.training_prediction(y,x,times)
    assert torch.equal(estimate,a+velocity)
    assert torch.allclose(error,(velocity-(x-a)).square().mean((1,2))/x.square().mean((1,2)).clamp_min(1e-4))


@pytest.mark.parametrize('arm',d.ARMS)
def test_real_architecture_checkpoint_roundtrip_and_strict_parent_metadata(tmp_path,arm):
    model=m.AnchoredFlow(parent(),arm).eval();p=tmp_path/'field.pt';m.save_checkpoint(p,model,None,arm,10,2470)
    loaded,record=m.load_checkpoint(p,arm,'cpu');x=torch.randn(2,2,256)
    assert torch.equal(model(x),loaded(x)) and not any(v.requires_grad for v in loaded.parameters())
    assert record['coarse_updates']==record['correction_updates']==0
    item=torch.load(p,weights_only=False);item['parent_run']='failed_refinement';torch.save(item,p)
    with pytest.raises(ValueError):m.load_checkpoint(p,arm,'cpu')


@pytest.mark.parametrize('key,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),('head_fit_updates',1),('coarse_updates',1),('correction_updates',1),('epoch',11),('updates',2471)])
def test_checkpoint_forbidden_permissions_and_budgets(tmp_path,key,value):
    model=m.AnchoredFlow(parent(),d.ARMS[0]);p=tmp_path/'field.pt';m.save_checkpoint(p,model,None,d.ARMS[0],10,2470)
    record=torch.load(p,weights_only=False);record[key]=value;torch.save(record,p)
    with pytest.raises(ValueError):m.load_checkpoint(p,d.ARMS[0],'cpu')


def test_registered_routes_use_original_clean_when_cache_slot_ungenerated():
    x=np.ones((3,2,256),np.float32);arrays=dict(residual=np.full_like(x,2),full=np.full_like(x,3))
    y,z,r=runner.training_batch(x,arrays,np.array([.01,.3,.99]),np.arange(3))
    assert y[:,0,0].tolist()==[1.,2.,3.,1.,1.,1.] and np.array_equal(z,np.concatenate((x,x)))


@pytest.mark.parametrize('update',[0,246,247,2469])
def test_fixed_newfield_optimizer_schedule(update):
    value=runner.learning_rate(update);assert 0<value<=2e-4
    if update==2469:assert abs(value-2e-5)<1e-10


def test_target_policy_all_seven_native_heads_without_source_gate():
    matrix=d.target_matrix();assert matrix['count']==168000 and len(matrix['views'])==7
    assert matrix['F_seeds']==[2026092701,2026092702,2026092703] and not matrix['source_selection'] and not matrix['target_feedback']
    for arm in d.ARMS:
        cfg=d.configuration(arm);assert cfg['component_epochs']==cfg['serial_inherited_passes']==50
        assert cfg['parent_coarse_epochs']==cfg['parent_correction_epochs']==20 and cfg['posterior_weight']==.3
    assert d.PARENT_RUN=='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'


@pytest.mark.parametrize('arm',d.ARMS)
def test_target_source_check_accepts_fixed_field_budget_then_rejects_parent_update(tmp_path,arm):
    source=tmp_path/d.RUN;row=source/arm;(row/'prediction').mkdir(parents=True)
    objects={source/'launch.json':dict(commit='method',fixed_target_arms=list(d.ARMS)),source/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,source_only=True,target_reads=0,epochs=10,updates=2470,checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=0,correction_updates=0,trainable_parameters=d.TRAINABLE_PARAMETERS,field_updates=2470,field_epochs=10,F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=list(target.VIEWS),role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0)}
    for p,value in objects.items():p.write_text(json.dumps(value),encoding='utf-8')
    args=SimpleNamespace(source_run=source,arm=arm,expected_commit='method');target.source_check(args)
    p=row/'training_complete.json';value=objects[p];value['coarse_updates']=2470;p.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError):target.source_check(args)


def test_source_truth_is_unreachable_when_predictions_missing(tmp_path):
    with pytest.raises(FileNotFoundError):evaluation.completed_predictions(tmp_path,np.arange(1800))


def test_target_truth_is_unreachable_when_predictions_missing(tmp_path):
    with pytest.raises(FileNotFoundError):target.completed_predictions(tmp_path,np.arange(168000))


def test_float32_rounding_cannot_move_sampled_time_to_one():
    class EdgeRng:
        def random(self,n):return np.full(n,1.-1e-12)
    times=runner.sample_times(EdgeRng(),2,d.ARMS[1]);assert (times<1).all() and (times>0).all()
