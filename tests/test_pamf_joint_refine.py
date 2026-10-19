import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_joint_refine import design as d
from experiments.pamf_joint_refine.models import JointResidual,save_checkpoint,load_checkpoint
from experiments.pamf_joint_refine.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_reception_restore.models import ComplexTFUNet


def test_joint_context_backpropagates_to_both_original_modules():
    torch.set_num_threads(2);model=JointResidual(ComplexTFUNet(),.1);model.train()
    assert model.coarse.training and all(p.requires_grad for p in model.parameters())
    model(torch.randn(2,2,256)).square().mean().backward()
    for module in (model.coarse,model.correction):
        grad=[p.grad for p in module.parameters() if p.grad is not None]
        assert grad and all(torch.isfinite(g).all() for g in grad) and sum(float(g.abs().sum()) for g in grad)>0


def test_joint_eval_preserves_original_parent_inference_graph():
    old=ResidualDeterministic(ComplexTFUNet(),.1).eval();new=JointResidual(ComplexTFUNet(),.1).eval()
    new.load_state_dict(old.state_dict(),strict=True);x=torch.randn(2,2,256)
    with torch.no_grad():assert torch.equal(old(x),new(x))


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
 ('head_fit_updates',1),('coarse_updates',1),('updates',2471),('parent_coarse_epochs',50)])
def test_joint_checkpoint_permissions_and_budget(tmp_path,field,value):
    path=tmp_path/'m.pt';model=JointResidual(ComplexTFUNet(),.1);save_checkpoint(path,model,None,d.ARM,0,0)
    v=torch.load(path,weights_only=False);v[field]=value;torch.save(v,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def summary(gain=0,nmse=.03,drop=0):
    records=[]
    for view in VIEWS:
        for name in ('identity','refined'):
            acc=98.5 if view=='clean' else 80.+(gain if name=='refined' else 0)
            if view=='clean' and name=='refined':acc-=drop
            records.append(dict(view=view,readout=name,baseline='identity',dimension='overall',accuracy=acc))
    return dict(source_only=True,count=1800,target_reads=0,records=records,waveform=[dict(view=v,NMSE=dict(refined=nmse)) for v in VIEWS])


def test_source_selection_can_choose_joint_feature_when_it_is_better():
    result=d.source_rule(dict(zip(d.ARMS,(summary(2),summary(1)))),summary())
    assert result['qualified'] and result['selected_arm']==d.ARMS[0]
    result=d.source_rule(dict(zip(d.ARMS,(summary(.2),summary(1)))),summary())
    assert result['selected_arm']==d.ARM


def test_source_selector_keeps_waveform_and_clean_limits():
    result=d.source_rule(dict(zip(d.ARMS,(summary(1,.032),summary(1,.03,.2)))),summary())
    assert not result['qualified'] and result['selected_arm'] is None
    with pytest.raises(ValueError):d.source_rule({d.ARM:summary(1)},summary())


def packets(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_all_seven_views_before_truth(tmp_path):
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    saved,ids,done,values=packets(tmp_path)
    if bad=='missingview':done['views']=list(VIEWS[:-1])
    elif bad=='earlytruth':done['TX_truth_reads']=1
    elif bad=='pairedclean':done['paired_clean_in_practical_inference']=True
    elif bad=='stateupdate':done['state_updates']=1
    elif bad=='badseed':values['seeds']=np.asarray(SEEDS[:1])
    elif bad=='badclass':values['refined'][0,0]=6
    elif bad=='ids':values['ids']=ids[::-1]
    else:np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.zeros((1,2,256),np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(VIEWS[-1]+'.npz'),**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,ids)


def test_unselected_joint_arm_cannot_access_target(tmp_path):
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='x')))
    (root/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=True,selected_arm=d.ARMS[0],target_reads=0)))
    row=root/d.ARM;row.mkdir();(row/'training_complete.json').write_text('{}')
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARM))
