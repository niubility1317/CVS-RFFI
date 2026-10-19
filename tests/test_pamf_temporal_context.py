import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_temporal_context import design as d
from experiments.pamf_temporal_context.models import TemporalAttention,ContextCore,JointContext,initialize,save_checkpoint,load_checkpoint
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet

torch.set_num_threads(2)


def test_zero_attention_exactly_preserves_full_parent_and_matching_parameters(monkeypatch):
    from experiments.pamf_temporal_context import models
    from experiments.pamf_residual_restore.models import ResidualDeterministic
    torch.manual_seed(1);parent=ResidualDeterministic(ComplexTFUNet(),.1).eval()
    nn.init.normal_(parent.correction.out.weight,0,.02)
    monkeypatch.setattr(models,'load_restorer',lambda path:parent)
    x=torch.randn(2,2,256);values=[]
    for arm in d.ARMS:
        torch.manual_seed(3);model=initialize('unused','cpu',arm);values.append(model)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==d.TRAINABLE_PARAMETERS==2207144
        assert sum(p.numel() for p in model.correction.attention.parameters())==d.ATTENTION_PARAMETERS==66304
        with torch.no_grad():assert torch.equal(model(x),parent(x))
    assert all(torch.equal(v,values[1].state_dict()[k]) for k,v in values[0].state_dict().items())


def test_explicit_attention_window_has_actual_narrow_and_full_context():
    torch.manual_seed(2);narrow=TemporalAttention(d.ARMS[0]).eval();wide=TemporalAttention(d.ARMS[1]).eval()
    wide.load_state_dict(narrow.state_dict(),strict=True)
    for module in (narrow,wide):nn.init.eye_(module.out.weight)
    assert narrow.allowed_pairs==154 and wide.allowed_pairs==1024
    assert narrow.mask[0,:3].tolist()==[False]*3 and narrow.mask[0,3:].all() and not wide.mask.any()
    x=torch.randn(2,128,32);changed=x.clone();changed[:,:,31]*=-5
    with torch.no_grad():
        assert torch.equal(narrow(x)[:,:,0],narrow(changed)[:,:,0])
        assert not torch.allclose(wide(x)[:,:,0],wide(changed)[:,:,0])


def test_zero_output_attention_gets_output_gradient_then_learns_QKV():
    module=TemporalAttention(d.ARMS[1]);x=torch.randn(2,128,32,requires_grad=True)
    optimizer=torch.optim.AdamW(module.parameters(),lr=2e-4);initial=module.qkv.weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);module(x).square().mean().backward()
        assert module.out.weight.grad.abs().sum()>0 and torch.isfinite(module.out.weight.grad).all()
        if step==0:assert torch.count_nonzero(module.qkv.weight.grad)==0
        else:assert module.qkv.weight.grad.abs().sum()>0 and torch.isfinite(module.qkv.weight.grad).all()
        optimizer.step()
    assert not torch.equal(module.qkv.weight,initial)
    assert module.training_calls==2 and module.training_IQ_examples==4 and module.training_context_pairs==4096


def test_IQ_only_inference_batch_independence_and_checkpoint_roundtrip(tmp_path):
    for arm in d.ARMS:
        model=JointContext(ComplexTFUNet(),.1,arm).eval();nn.init.normal_(model.correction.out.weight,0,.02)
        nn.init.normal_(model.correction.attention.out.weight,0,.02)
        x=torch.randn(2,2,256);before={k:v.clone() for k,v in model.state_dict().items()}
        buffers={k:v.clone() for k,v in model.named_buffers()}
        with torch.no_grad():y=model(x);alone=torch.cat([model(v[None]) for v in x])
        assert torch.allclose(y,alone,atol=1e-6,rtol=1e-5)
        assert model.correction.attention.training_calls==0
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
        path=tmp_path/(arm+'.pt');save_checkpoint(path,model,None,arm,d.EPOCHS,d.UPDATES)
        loaded,_=load_checkpoint(path,arm,'cpu')
        assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
        with torch.no_grad():assert torch.equal(y,loaded(x))


@pytest.mark.parametrize('bad',['shape','dtype','arm'])
def test_invalid_current_frame_contract(bad):
    if bad=='arm':
        with pytest.raises(ValueError):TemporalAttention('unknown')
        return
    module=TemporalAttention(d.ARM);x=torch.randn(1,128,32)
    if bad=='shape':x=x[:,:,:31]
    else:x=x.double()
    with pytest.raises(ValueError):module(x)


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
 ('head_fit_updates',1),('parent_corrector_epochs',0),('parent_coarse_epochs',0),('coarse_updates',1),('updates',2471),('attention_updates',1),('attention_epochs',1)])
def test_checkpoint_permissions_budget_and_parent(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointContext(ComplexTFUNet(),.1,d.ARM),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def fixed_source(root,arm):
    from experiments.pamf_temporal_context import design as d
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={
      root/'launch.json':dict(commit='x',fixed_target_arms=list(d.ARMS)),root/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=10,updates=2470,source_only=True,target_reads=0,
         checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=2470,trainable_parameters=2207144,attention_updates=2470,attention_epochs=10,
         F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=d.target_matrix()['views'],
         role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,
         diagnostic_accuracy=1.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value),encoding='utf-8')
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x'),values


def test_both_fixed_rows_may_test_without_source_score_selection_or_other_row_completion(tmp_path):
    from experiments.pamf_temporal_context.target import source_check
    for arm in d.ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[d.ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),('training_complete','attention_updates',0),
 ('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_temporal_context.target import source_check
    args,_=fixed_source(tmp_path,d.ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_temporal_context.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_temporal_context.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_temporal_context.target import VIEWS,SEEDS,completed_predictions
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
    from experiments.pamf_temporal_context.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {d.ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
