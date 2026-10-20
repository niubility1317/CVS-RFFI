import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_tangent_guidance import design as d
from experiments.pamf_tangent_guidance.models import initialize,save_checkpoint,load_checkpoint,JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet
torch.set_num_threads(2)


class LinearF(nn.Module):
    def __init__(self,seed):
        super().__init__();generator=torch.Generator().manual_seed(seed)
        self.feature_weight=nn.Parameter(torch.randn(160,512,generator=generator)/20,requires_grad=False)
        self.head_weight=nn.Parameter(torch.randn(6,160,generator=generator)/10,requires_grad=False)
        self.eval()
    def features(self,x):return x.flatten(1)@self.feature_weight.T
    def classify_features(self,x):return x@self.head_weight.T


def readers():return {seed:LinearF(seed) for seed in d.TEACHER_SEEDS}


@pytest.mark.parametrize('arm',d.ARMS)
def test_actual_jacobian_matches_finite_difference_and_is_batch_independent(arm):
    from experiments.pamf_tangent_guidance.guidance import source_jacobians,coordinates
    models=readers();torch.manual_seed(7);x=torch.randn(2,2,256);direction=torch.randn_like(x)/20
    features,jac=source_jacobians(models,x,arm)
    assert features.shape==(2,3,160) and jac.shape==(2,3,6,2,256) and not jac.requires_grad
    for index,(seed,model) in enumerate(models.items()):
        numeric=(coordinates(model,model.features(x+1e-3*direction),arm,seed)-coordinates(model,model.features(x-1e-3*direction),arm,seed))/(2e-3)
        analytic=(jac[:,index]*direction[:,None]).sum((2,3))
        assert torch.allclose(numeric,analytic,atol=3e-4,rtol=.01)
    singles=[source_jacobians(models,value[None],arm) for value in x]
    assert torch.allclose(jac,torch.cat([j for _,j in singles]),atol=1e-7,rtol=1e-4)
    assert not any(p.grad is not None for model in models.values() for p in model.parameters())


def test_projection_is_fixed_six_probe_math_without_fitted_parameters():
    from experiments.pamf_tangent_guidance.guidance import feature_projection
    torch.manual_seed(5);before=torch.get_rng_state().clone();p=feature_projection(d.TEACHER_SEEDS[0],'cpu',torch.float32)
    assert p.shape==(6,160) and not p.requires_grad and torch.equal(before,torch.get_rng_state())
    assert torch.equal(p,feature_projection(d.TEACHER_SEEDS[0],'cpu',torch.float32))
    assert not torch.equal(p,feature_projection(d.TEACHER_SEEDS[1],'cpu',torch.float32))
    assert torch.allclose(p.abs(),torch.full_like(p,1/(6**.5)))


def test_posterior_coordinate_is_native_all_class_symmetric_and_stable():
    from experiments.pamf_tangent_guidance.guidance import coordinates
    model=LinearF(7);x=torch.randn(2,160);permutation=torch.tensor([5,3,1,4,0,2]);before=coordinates(model,x,d.ARMS[1],0)
    model.head_weight.data=model.head_weight.data[permutation]
    assert torch.allclose(coordinates(model,x,d.ARMS[1],0),before[:,permutation])
    huge=(1000*x).requires_grad_();value=coordinates(model,huge,d.ARMS[1],0)
    assert torch.isfinite(value).all() and torch.isfinite(torch.autograd.grad(value.sum(),huge)[0]).all()


@pytest.mark.parametrize('arm',d.ARMS)
def test_exactly_one_identity_component_is_replaced_and_joint_IQ_gradient_is_capped(arm):
    from experiments.pamf_tangent_guidance.guidance import guidance,source_jacobians,tangent_distance
    from experiments.pamf_tangent_guidance.models import feature_distance,posterior_distance
    from experiments.pamf_tangent_guidance.gradients import output_gradient
    models=readers();torch.manual_seed(5);clean=torch.randn(3,2,256);estimate=(clean+.1*torch.randn_like(clean)).requires_grad_()
    refs,jac=source_jacobians(models,clean,arm);local=tangent_distance(estimate,clean,jac).mean(0)
    combined,feature,posterior=guidance(models,estimate,refs,arm,clean,jac)
    exact_features=[];exact_posteriors=[]
    for index,model in enumerate(models.values()):
        embed=model.features(estimate)
        exact_features.append(feature_distance(embed,refs[:,index]))
        exact_posteriors.append(posterior_distance(model.classify_features(embed),model.classify_features(refs[:,index])))
    assert torch.allclose(feature,local if arm==d.ARMS[0] else torch.stack(exact_features).mean(0))
    assert torch.allclose(posterior,torch.stack(exact_posteriors).mean(0) if arm==d.ARMS[0] else local)
    assert torch.allclose(combined,feature+.3*posterior)
    paired=(estimate-clean).square().mean((1,2));gradient,stats=output_gradient(estimate,paired,combined,arm)
    assert not gradient.requires_grad and torch.isfinite(gradient).all()
    assert (stats['identity_after_norm2']<=stats['wave_norm2']+1e-5).all()


def test_quadratic_gradient_is_two_J_transpose_J_delta_without_updating_clean_or_J():
    from experiments.pamf_tangent_guidance.guidance import tangent_distance
    torch.manual_seed(3);estimate=torch.randn(2,2,256,requires_grad=True);clean=torch.randn_like(estimate,requires_grad=True)
    jac=torch.randn(2,3,6,2,256,requires_grad=True);value=tangent_distance(estimate,clean,jac).sum()
    value.backward();matrix=jac.detach().reshape(2,18,512);delta=(estimate-clean).detach().flatten(1)[:,:,None]
    expected=(2*matrix.transpose(1,2).bmm(matrix.bmm(delta))).reshape_as(estimate)
    assert torch.allclose(estimate.grad,expected,atol=1e-3,rtol=1e-4) and clean.grad is None and jac.grad is None


@pytest.mark.parametrize('arm',d.ARMS)
def test_selected_source_tangents_follow_observed_clean_pair_and_original_route(arm):
    from experiments.pamf_tangent_guidance.runner import training_batch
    from experiments.pamf_tangent_guidance.guidance import array_tensor
    clean=np.arange(5*512,dtype=np.float32).reshape(5,2,256);selection=np.array([4,1,3])
    values,targets,route=training_batch(clean,dict(residual=clean+10,full=clean+20),np.array([0,.5,0,.9,.1]),selection)
    assert route.tolist()==['clean','residual','full'] and np.array_equal(targets,np.concatenate([clean[selection]]*2))
    jac=np.arange(5*3*6*512,dtype=np.float32).reshape(5,3,6,2,256)
    paired=array_tensor(jac[selection],'cpu').repeat(2,1,1,1,1)
    assert torch.equal(paired[:3],paired[3:]) and torch.equal(paired[0],array_tensor(jac[4],'cpu'))


def test_ABI_independent_array_bridge_handles_readonly_noncontiguous_and_owns_buffer():
    from experiments.pamf_tangent_guidance.guidance import array_tensor
    x=np.arange(24,dtype=np.float64).reshape(2,3,4)[:,:,::2];x.flags.writeable=False
    tensor=array_tensor(x,'cpu');expected=torch.tensor(x.tolist(),dtype=torch.float32)
    assert torch.equal(tensor,expected);tensor.add_(1);assert torch.equal(tensor,expected+1)
    assert np.array_equal(x,np.arange(24,dtype=np.float64).reshape(2,3,4)[:,:,::2])


@pytest.mark.parametrize('bad',['train','parameters','shape','arm'])
def test_source_tangent_rejects_unfrozen_or_wrong_input_contract(bad):
    from experiments.pamf_tangent_guidance.guidance import source_jacobians
    models=readers();x=torch.randn(1,2,256);arm=d.ARM
    if bad=='train':next(iter(models.values())).train()
    elif bad=='parameters':next(iter(models.values())).requires_grad_(True)
    elif bad=='shape':x=x[:,:,:255]
    else:arm='unregistered'
    with pytest.raises(ValueError):source_jacobians(models,x,arm)


def test_parent_state_batch_independence_checkpoint_and_external_control(monkeypatch,tmp_path):
    from experiments.pamf_tangent_guidance import models
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
    from experiments.pamf_tangent_guidance import design as d
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
    from experiments.pamf_tangent_guidance.target import source_check
    for arm in d.ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[d.ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),('training_complete','gradient_updates',0),
 ('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_tangent_guidance.target import source_check
    args,_=fixed_source(tmp_path,d.ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_tangent_guidance.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_tangent_guidance.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_tangent_guidance.target import VIEWS,SEEDS,completed_predictions
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
    from experiments.pamf_tangent_guidance.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {d.ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
