import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_teacher_agreement import design as d
from experiments.pamf_teacher_agreement.models import initialize,save_checkpoint,load_checkpoint,JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet
torch.set_num_threads(2)


def test_frame_and_complex_time_granularity_have_distinct_registered_behavior():
    from experiments.pamf_teacher_agreement.gradients import teacher_agreement
    g=torch.zeros(3,1,2,256);g[:,:,0,:2]=1;g[2,0,0,0]=-3
    frame,a=teacher_agreement(g,d.ARMS[0]);time,b=teacher_agreement(g,d.ARMS[1])
    assert torch.equal(frame,torch.zeros_like(frame)) and a['units']==1 and a['agreed']==0
    assert time[0,0,0]==0 and time[0,0,1]==1 and b['units']==256 and b['agreed']==255 and b['active_agreed']==1
    assert a['pair_conflicts']==[0,1,1] and b['pair_conflicts']==[0,1,1]


@pytest.mark.parametrize('arm',d.ARMS)
def test_identical_teacher_gradients_and_all_zeros_have_defined_behavior(arm):
    from experiments.pamf_teacher_agreement.gradients import teacher_agreement
    torch.manual_seed(1);g=torch.randn(2,2,256);value,stats=teacher_agreement(g[None].repeat(3,1,1,1),arm)
    assert torch.allclose(value,g) and stats['agreed']==stats['units'] and stats['pair_conflicts']==[0,0,0]
    zero,stats=teacher_agreement(torch.zeros(3,2,2,256),arm)
    assert torch.equal(zero,torch.zeros_like(zero)) and stats['agreed']==stats['units'] and stats['active_agreed']==0
    assert stats['pair_valid']==[0,0,0] and stats['pair_cosine_sum']==[0,0,0]


@pytest.mark.parametrize('arm',d.ARMS)
def test_teacher_permutation_and_orthogonal_complex_rotation_invariance(arm):
    from experiments.pamf_teacher_agreement.gradients import teacher_agreement
    torch.manual_seed(2);g=torch.randn(3,4,2,256);value,stats=teacher_agreement(g,arm)
    permuted,_=teacher_agreement(g[torch.tensor([2,0,1])],arm)
    assert torch.allclose(value,permuted,atol=2e-7,rtol=1e-5)
    for rotation in (torch.tensor([[.6,-.8],[.8,.6]]),torch.tensor([[0.,1.],[1.,0.]])):
        rotated=torch.einsum('ij,tnjl->tnil',rotation,g);result,_=teacher_agreement(rotated,arm)
        assert torch.allclose(result,torch.einsum('ij,njl->nil',rotation,value),atol=3e-7,rtol=1e-5)
    assert stats['agreed_norm2']<=stats['raw_norm2']*(1+1e-6)
    assert (g*value[None]).sum((2,3)).min()>=-1e-5


@pytest.mark.parametrize('arm',d.ARMS)
def test_one_zero_teacher_does_not_create_opposed_identity_directions(arm):
    from experiments.pamf_teacher_agreement.gradients import teacher_agreement
    g=torch.ones(3,2,2,256);g[0].zero_();result,stats=teacher_agreement(g,arm)
    assert torch.allclose(result,torch.full_like(result,2/3)) and stats['agreed']==stats['units']
    assert stats['pair_valid'][0]==stats['pair_valid'][1]==0 and stats['pair_conflicts']==[0,0,0]


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('n',[1,3])
def test_all_agreed_inactive_cap_matches_original_mean3_loss_parameter_gradient_and_step(arm,n):
    from experiments.pamf_teacher_agreement.gradients import output_gradient
    torch.manual_seed(3);a=nn.Conv1d(2,2,1);b=nn.Conv1d(2,2,1);b.load_state_dict(a.state_dict());x=torch.randn(n,2,256)
    opta=torch.optim.AdamW(a.parameters(),lr=3e-5);optb=torch.optim.AdamW(b.parameters(),lr=3e-5)
    ya=a(x);yb=b(x);wavea=ya.square().mean((1,2));waveb=yb.square().mean((1,2))
    ta=tuple(weight*wavea for weight in (.01,.02,.03));tb=torch.stack([weight*waveb for weight in (.01,.02,.03)])
    merged,stats=output_gradient(ya,wavea,ta,arm);ya.backward(merged);(waveb+tb.mean(0)).mean().backward()
    assert stats['agreement']['agreed']==stats['agreement']['units'] and not stats['scaled'].any()
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)
    opta.step();optb.step()
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa,pb,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('arm',d.ARMS)
def test_actual_three_pullbacks_wave_cap_and_accumulated_units(arm):
    from experiments.pamf_teacher_agreement.gradients import output_gradient
    from experiments.pamf_teacher_agreement.runner import empty_gradient_state,accumulate_gradients
    model=nn.Conv1d(2,2,1);x=torch.randn(4,2,256);y=model(x);wave=y.square().mean((1,2))
    teachers=tuple(weight*wave for weight in (10,20,30));merged,stats=output_gradient(y,wave,teachers,arm)
    y.backward(merged);state=empty_gradient_state();accumulate_gradients(state,stats,2)
    assert state['calls']==1 and state['IQ_examples']==4 and (stats['identity_after_norm2']<=stats['wave_norm2']+1e-4).all()
    a=state['agreement'];assert a['units']==4*(1 if arm==d.ARMS[0] else 256) and a['agreed']==a['units']
    assert a['agreed_norm2']<=a['raw_norm2']*(1+1e-6) and a['teacher_negative_after']==0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('bad',['shape','dtype','nan','arm'])
def test_invalid_teacher_gradient_contract(bad):
    from experiments.pamf_teacher_agreement.gradients import teacher_agreement
    g=torch.randn(3,2,2,256);arm=d.ARM
    if bad=='shape':g=g[:2]
    elif bad=='dtype':g=g.double()
    elif bad=='nan':g[0,0,0,0]=float('nan')
    else:arm='unregistered'
    with pytest.raises(ValueError):teacher_agreement(g,arm)


def test_guidance_preserves_three_original_losses_and_mean_before_custom_backward():
    from experiments.pamf_teacher_agreement.guidance import guidance
    from experiments.pamf_teacher_agreement.models import feature_distance,posterior_distance
    class FakeF(nn.Module):
        def features(self,x):return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    models={seed:FakeF().eval() for seed in d.TEACHER_SEEDS};x=torch.randn(2,2,256,requires_grad=True);refs=torch.randn(2,3,160)
    mean,feature,posterior,teachers=guidance(models,x,refs,d.ARM)
    expected=[]
    for i,model in enumerate(models.values()):
        f=model.features(x);r=refs[:,i]
        expected.append(feature_distance(f,r)+.3*posterior_distance(model.classify_features(f),model.classify_features(r)))
    assert isinstance(teachers,tuple) and torch.allclose(torch.stack(teachers),torch.stack(expected)) and torch.equal(mean,torch.stack(teachers).mean(0))
    assert torch.allclose(mean,feature+.3*posterior)


def test_parent_state_batch_independence_checkpoint_and_external_control(monkeypatch,tmp_path):
    from experiments.pamf_teacher_agreement import models
    from experiments.pamf_residual_restore.models import ResidualDeterministic
    parent=ResidualDeterministic(ComplexTFUNet(),.1).eval();nn.init.normal_(parent.correction.out.weight,0,.02);opened=[]
    monkeypatch.setattr(models,'load_restorer',lambda path:(opened.append(path),parent)[1]);x=torch.randn(2,2,256);values=[]
    for arm in d.ARMS:
        model=initialize('original_R03_only','cpu',arm);values.append(model);assert sum(p.numel() for p in model.parameters())==2140840
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():assert torch.equal(model(x),parent(x));assert torch.allclose(model(x),torch.cat([model(v[None]) for v in x]),atol=1e-6,rtol=1e-5)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        path=tmp_path/(arm+'.pt');save_checkpoint(path,model,None,arm,10,2470);loaded,_=load_checkpoint(path,arm,'cpu')
        with torch.no_grad():assert torch.equal(model(x),loaded(x))
    assert opened==['original_R03_only']*2 and all(torch.equal(v,values[1].state_dict()[k]) for k,v in values[0].state_dict().items())
    assert d.CONTROL_ARM=='per_frame_cap' and d.target_matrix()['external_control']['new_training'] is False
    assert all(d.configuration(arm)['control_run']==d.CONTROL_RUN for arm in d.ARMS)


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),('head_fit_updates',1),
 ('parent_corrector_epochs',0),('parent_coarse_epochs',0),('coarse_updates',1),('updates',2471),('gradient_updates',1),('gradient_epochs',1)])
def test_checkpoint_permissions_budget_and_parent(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def fixed_source(root,arm):
    from experiments.pamf_teacher_agreement import design as d
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={
      root/'launch.json':dict(commit='x',fixed_target_arms=list(d.ARMS)),root/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=10,updates=2470,source_only=True,target_reads=0,
         checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=2470,trainable_parameters=2140840,gradient_updates=2470,gradient_epochs=10,
         F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=d.target_matrix()['views'],
         role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,
         diagnostic_accuracy=1.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value),encoding='utf-8')
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x'),values


def test_both_fixed_rows_may_test_without_source_score_selection_or_other_row_completion(tmp_path):
    from experiments.pamf_teacher_agreement.target import source_check
    for arm in d.ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[d.ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),('training_complete','gradient_updates',0),
 ('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_teacher_agreement.target import source_check
    args,_=fixed_source(tmp_path,d.ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_teacher_agreement.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_teacher_agreement.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_teacher_agreement.target import VIEWS,SEEDS,completed_predictions
    saved,ids,done,values=packets(tmp_path)
    if bad=='missingview':done['views']=list(VIEWS[:-1])
    elif bad=='earlytruth':done['TX_truth_reads']=1
    elif bad=='pairedclean':done['paired_clean_in_practical_inference']=True
    elif bad=='stateupdate':done['state_updates']=1
    elif bad=='badseed':values['seeds']=np.asarray(SEEDS[:1])
    elif bad=='badclass':values['refined'][0,0]=6
    elif bad=='ids':values['ids']=ids[::-1]
    elif bad=='missingIQ':np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.zeros((1,2,256),np.float32))
    else:np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.full((2,2,256),np.nan,np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(VIEWS[-1]+'.npz'),**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,ids)


def test_controller_progress_can_update_through_source_target_completion(tmp_path):
    from experiments.pamf_teacher_agreement.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {d.ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
