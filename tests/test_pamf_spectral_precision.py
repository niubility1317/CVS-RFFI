import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_spectral_precision import design as d,models,runner,target
from experiments.pamf_spectral_precision.losses import precision_weights,source_precision,pair_loss
from experiments.pamf_reception_restore.runner import pair_loss as old_pair_loss
from experiments.pamf_identity_priority.models import JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet
torch.set_num_threads(2)


def test_precision_bounded_mean_one_scale_invariant_and_emphasizes_weak_source_bins():
    power=np.geomspace(1e-5,100,256)
    weights=precision_weights(power,d.ARMS[1])
    assert weights.min()>=.25 and weights.max()<=4 and weights.mean()==pytest.approx(1,abs=1e-6)
    assert weights[0]>weights[-1] and np.allclose(weights,precision_weights(power*37,d.ARMS[1]))
    assert np.array_equal(precision_weights(power,d.ARMS[0]),np.ones(256,np.float32))
    assert np.array_equal(precision_weights(np.ones(256),d.ARMS[1]),np.ones(256,np.float32))


@pytest.mark.parametrize('bad',[np.zeros(256),np.ones(255),np.full(256,-1.),np.full(256,np.nan),np.full(256,np.inf)])
def test_invalid_source_spectrum_is_rejected(bad):
    with pytest.raises(ValueError):precision_weights(bad,d.ARMS[1])


def test_source_population_uses_IQ_only_and_records_no_label_V_or_target_access():
    t=np.arange(256);frame=np.stack((np.cos(2*np.pi*t/32),np.sin(2*np.pi*t/32))).astype(np.float32)
    clean=np.broadcast_to(frame,(63000,2,256))
    weights,state=source_precision(clean,d.ARMS[1]);models.validate_spectral_state(state,d.ARMS[1])
    assert state['count']==63000 and state['TX_labels_read']==state['V_reads']==state['target_reads']==0
    assert weights[8]<weights[100]


def test_uniform_complex_spectrum_is_parseval_IQNMSE_including_near_zero_energy_floor():
    torch.manual_seed(4)
    for magnitude in (1.,1e-4):
        clean=magnitude*torch.randn(3,2,256);estimate=(.8*clean+.001).requires_grad_()
        loss,terms=pair_loss(estimate,clean,torch.ones(256));old,original=old_pair_loss(estimate,clean)
        assert torch.allclose(terms['spectral_nmse'],original['iq_nmse'],atol=1e-5,rtol=1e-5)
        expected=old+.1*original['iq_nmse']
        assert torch.allclose(loss,expected,atol=1e-5,rtol=1e-5)
        actual_grad=torch.autograd.grad(loss.sum(),estimate,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected.sum(),estimate)[0]
        assert torch.allclose(actual_grad,expected_grad,atol=2e-5,rtol=2e-5)


def test_weighted_complex_loss_retains_phase_and_changes_frequency_gradient():
    clean=torch.randn(2,2,256);phase_rotated=torch.stack((-clean[:,1],clean[:,0]),1).requires_grad_()
    weights=torch.tensor(precision_weights(np.geomspace(.01,10,256),d.ARMS[1]))
    loss,terms=pair_loss(phase_rotated,clean,weights)
    _,old=old_pair_loss(phase_rotated,clean)
    assert old['log_power'].max()<1e-5 and terms['spectral_nmse'].min()>0
    gradient=torch.autograd.grad(loss.sum(),phase_rotated,retain_graph=True)[0]
    uniform,_=pair_loss(phase_rotated,clean,torch.ones(256))
    assert not torch.allclose(gradient,torch.autograd.grad(uniform.sum(),phase_rotated)[0])


def state(arm):return dict(arm=arm,count=63000,roles=['L_s','U_s'],TX_labels_read=0,V_reads=0,target_reads=0,
 rule=d.configuration(arm)['spectral_rule'],weight_min=1.,weight_max=1.,weight_mean=1.,source_global_power_mean=1.)


def test_original_parent_state_and_new_checkpoint_roundtrip_without_runtime_weights(tmp_path,monkeypatch):
    torch.manual_seed(8);parent=JointResidual(ComplexTFUNet(),.1);nn.init.normal_(parent.correction.out.weight,0,.001)
    monkeypatch.setattr(models,'load_restorer',lambda path:parent);initial=[];x=torch.randn(2,2,256)
    for arm in d.ARMS:
        model=models.initialize(tmp_path/'parent','cpu',arm);initial.append(model.state_dict())
        assert sum(p.numel() for p in model.parameters())==2140840
        with torch.no_grad():assert torch.equal(model(x),parent(x))
        path=tmp_path/(arm+'.pt');models.save_checkpoint(path,model,None,arm,10,2470,spectral_state=state(arm))
        fresh,item=models.load_checkpoint(path,arm,'cpu')
        with torch.no_grad():assert torch.equal(model(x),fresh(x))
        assert 'weights' not in fresh.state_dict() and not any(p.requires_grad for p in fresh.parameters())
    assert all(torch.equal(v,initial[1][k]) for k,v in initial[0].items())


@pytest.mark.parametrize('key,value',[('count',1800),('roles',['V']),('TX_labels_read',1),('V_reads',1),('target_reads',1),
 ('rule','query_based'),('weight_min',.249),('weight_max',4.001),('weight_mean',.9),('source_global_power_mean',0)])
def test_spectral_state_rejects_wrong_population_or_illegal_weighting(key,value):
    value_state=state(d.ARMS[1]);value_state[key]=value
    with pytest.raises(ValueError):models.validate_spectral_state(value_state,d.ARMS[1])


def fixed_source(root,arm):
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={root/'launch.json':dict(commit='x',fixed_target_arms=list(d.ARMS)),root/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=10,updates=2470,source_only=True,target_reads=0,checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=2470,trainable_parameters=2140840,gradient_updates=2470,gradient_epochs=10,
        L_physical_exposures=63000,U_physical_exposures=567000,IQ_examples=1260000,spectral_state=state(arm),F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=list(target.VIEWS),role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,diagnostic_accuracy=0.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value))
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x')


@pytest.mark.parametrize('arm',d.ARMS)
def test_fixed_rows_test_without_source_score_or_target_based_selection(tmp_path,arm):target.source_check(fixed_source(tmp_path,arm))


@pytest.mark.parametrize('key,value',[('updates',2469),('coarse_updates',0),('head_fit_updates',1),('target_reads',1),('spectral_state',None)])
def test_illegal_or_incomplete_source_cannot_test(tmp_path,key,value):
    args=fixed_source(tmp_path,d.ARMS[0]);path=args.source_run/args.arm/'training_complete.json'
    item=json.loads(path.read_text());item[key]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):target.source_check(args)


def test_source_gradients_use_both_objectives_and_bounded_cap():
    from experiments.pamf_spectral_precision.gradients import output_gradient
    clean=torch.randn(2,2,256);estimate=(.8*clean).requires_grad_();weights=torch.ones(256)
    paired,_=pair_loss(estimate,clean,weights);identity=50*(estimate-clean).square().sum((1,2))
    value,stats=output_gradient(estimate,paired,identity,d.ARMS[1])
    assert torch.isfinite(value).all() and value.abs().sum()>0 and stats['scaled'].all()
    assert torch.all(stats['identity_after_norm2']<=stats['wave_norm2']*(1+1e-5))


def test_GPU_queue_checks_actual_owned_handle_after_terminal_marker(tmp_path,monkeypatch):
    previous=tmp_path/d.GPU_PRECEDING_RUN;previous.mkdir();(previous/'complete.json').write_text('{}')
    output=tmp_path/d.RUN;output.mkdir();states=iter(([7],[]));sleeps=[]
    monkeypatch.setattr(runner,'previous_GPU_rows_live',lambda p:next(states));monkeypatch.setattr(runner.time,'sleep',lambda seconds:sleeps.append(seconds))
    runner.wait_for_GPU_predecessor(SimpleNamespace(output=output))
    assert sleeps==[5] and json.loads((output/'GPU_predecessor_released.json').read_text())['scores_read']==0


@pytest.mark.parametrize('bad',['earlytruth','pairedclean','missingview','stateupdate'])
def test_incomplete_target_packets_are_rejected_before_truth(tmp_path,bad):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(target.VIEWS),count=2,F_seeds=list(target.SEEDS),readouts=list(target.READOUTS),TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    if bad=='earlytruth':done['TX_truth_reads']=1
    elif bad=='pairedclean':done['paired_clean_in_practical_inference']=True
    elif bad=='missingview':done['views']=list(target.VIEWS[:-1])
    else:done['state_updates']=1
    (saved/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):target.completed_predictions(tmp_path,ids)
