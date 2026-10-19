import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_gradient_balance import design as d
from experiments.pamf_gradient_balance.gradients import merge_gradients,output_gradient
from experiments.pamf_gradient_balance.models import initialize,save_checkpoint,load_checkpoint,JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet
torch.set_num_threads(2)


def test_global_and_per_frame_caps_have_distinct_declared_scope():
    w=torch.zeros(2,2,256);g=w.clone();w[:,0,0]=torch.tensor([1.,10.]);g[:,0,1]=torch.tensor([10.,1.])
    _,global_stats=merge_gradients(w,g,d.ARMS[0]);_,frame_stats=merge_gradients(w,g,d.ARMS[1])
    assert torch.equal(global_stats['scale'],torch.ones(2))
    assert torch.allclose(frame_stats['scale'],torch.tensor([.1,1.]))
    assert frame_stats['scaled'].tolist()==[True,False]


@pytest.mark.parametrize('arm',d.ARMS)
def test_caps_only_reduce_identity_norm_and_match_declared_bound(arm):
    torch.manual_seed(5);w=torch.randn(4,2,256);g=10*torch.randn_like(w)
    value,stats=merge_gradients(w,g,arm);adjusted=value-w
    assert ((stats['scale']>=0)&(stats['scale']<=1)).all() and stats['scaled'].all()
    assert torch.allclose(adjusted,g*stats['scale'][:,None,None],atol=1e-6,rtol=1e-5)
    if arm==d.ARMS[0]:assert stats['identity_after_norm2'].sum()<=stats['wave_norm2'].sum()+1e-3
    else:assert (stats['identity_after_norm2']<=stats['wave_norm2']+1e-3).all()


@pytest.mark.parametrize('arm',d.ARMS)
def test_zero_identity_wave_and_near_zero_inputs_remain_finite(arm):
    zero=torch.zeros(2,2,256);one=torch.ones_like(zero)
    value,stats=merge_gradients(zero,one,arm);assert torch.equal(value,zero) and torch.equal(stats['scale'],torch.zeros(2))
    value,stats=merge_gradients(one,zero,arm);assert torch.equal(value,one) and torch.equal(stats['scale'],torch.ones(2))
    value,stats=merge_gradients(zero,zero,arm);assert torch.equal(value,zero) and torch.isfinite(stats['scale']).all()
    value,_=merge_gradients(1e-15*one,1e-15*one,arm);assert torch.isfinite(value).all()


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('n',[1,3])
def test_inactive_cap_matches_scalar_parameter_gradient_and_step(arm,n):
    torch.manual_seed(2);original=nn.Conv1d(2,2,1);manual=nn.Conv1d(2,2,1);manual.load_state_dict(original.state_dict());x=torch.randn(n,2,256)
    opt1=torch.optim.AdamW(original.parameters(),lr=3e-5);opt2=torch.optim.AdamW(manual.parameters(),lr=3e-5)
    for model,optimizer,custom in ((original,opt1,False),(manual,opt2,True)):
        value=model(x);paired=value.square().mean((1,2));identity=.01*paired.clone()
        if custom:
            merged,stats=output_gradient(value,paired,identity,arm);assert not stats['scaled'].any();value.backward(merged)
        else:(paired+identity).mean().backward()
        optimizer.step()
    for a,b in zip(original.parameters(),manual.parameters()):assert torch.allclose(a.grad,b.grad,atol=1e-6,rtol=1e-5) and torch.allclose(a,b,atol=1e-7,rtol=1e-6)


def test_actual_balanced_backward_and_state_counts():
    from experiments.pamf_gradient_balance.runner import empty_gradient_state,accumulate_gradients
    model=nn.Conv1d(2,2,1);x=torch.randn(4,2,256);value=model(x);paired=value.square().mean((1,2));identity=100*paired.clone()
    merged,stats=output_gradient(value,paired,identity,d.ARMS[1]);value.backward(merged);state=empty_gradient_state();accumulate_gradients(state,stats,2)
    assert state['calls']==1 and state['IQ_examples']==state['scaled']==4 and state['observed_scaled']==state['clean_scaled']==2
    assert 0<state['scale_min']<=state['scale_max']<1 and state['identity_after_norm2_sum']<=state['wave_norm2_sum']+1e-5
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('bad',['shape','dtype','nan','arm'])
def test_invalid_output_gradient_contract(bad):
    w=torch.randn(1,2,256);g=w.clone();arm=d.ARM
    if bad=='shape':g=g[:,:,:255]
    elif bad=='dtype':g=g.double()
    elif bad=='nan':g[0,0,0]=float('nan')
    else:arm='unknown'
    with pytest.raises(ValueError):merge_gradients(w,g,arm)


def test_parent_state_batch_independence_checkpoint_and_external_control(monkeypatch,tmp_path):
    from experiments.pamf_gradient_balance import models
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
    assert d.CONTROL_ARM=='scalar_sum' and d.target_matrix()['external_control']['new_training'] is False
    assert all(d.configuration(arm)['control_run']==d.CONTROL_RUN for arm in d.ARMS)


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),('head_fit_updates',1),
 ('parent_corrector_epochs',0),('parent_coarse_epochs',0),('coarse_updates',1),('updates',2471),('gradient_updates',1),('gradient_epochs',1)])
def test_checkpoint_permissions_budget_and_parent(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def fixed_source(root,arm):
    from experiments.pamf_gradient_balance import design as d
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
    from experiments.pamf_gradient_balance.target import source_check
    for arm in d.ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[d.ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),('training_complete','gradient_updates',0),
 ('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_gradient_balance.target import source_check
    args,_=fixed_source(tmp_path,d.ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_gradient_balance.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_gradient_balance.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_gradient_balance.target import VIEWS,SEEDS,completed_predictions
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
    from experiments.pamf_gradient_balance.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {d.ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
