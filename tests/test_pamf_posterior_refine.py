import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_posterior_refine import design as d
from experiments.pamf_posterior_refine.losses import posterior_distance
from experiments.pamf_posterior_refine.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS
from experiments.pamf_posterior_refine.models import enable_correction,save_checkpoint,load_checkpoint
from experiments.pamf_posterior_refine.runner import training_batch
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_reception_restore.models import ComplexTFUNet


def test_native_posterior_reference_is_detached_but_estimate_has_gradient():
    e=torch.tensor([[2.,1.,0.,-1.,-2.,-3.]],requires_grad=True)
    r=torch.tensor([[0.,1.,3.,-1.,-2.,-3.]],requires_grad=True)
    loss=posterior_distance(e,r);assert loss.shape==(1,) and loss.item()>0
    loss.mean().backward();assert e.grad.abs().sum()>0 and r.grad is None
    assert posterior_distance(r,r).abs().max()<1e-6


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
    ('head_fit_updates',1),('coarse_updates',1),('updates',4941),('parent_corrector_epochs',50)])
def test_checkpoint_permissions_and_budget_are_strict(tmp_path,field,value):
    path=tmp_path/'model.pt';model=ResidualDeterministic(ComplexTFUNet(),.1)
    save_checkpoint(path,model,None,d.ARM,0,0);v=torch.load(path,weights_only=False);v[field]=value;torch.save(v,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def test_only_original_correction_trainable_and_frozen_coarse_stays_eval():
    model=enable_correction(ResidualDeterministic(ComplexTFUNet(),.1));model.train()
    assert not model.coarse.training
    assert all(p.requires_grad==name.startswith('correction.') for name,p in model.named_parameters())


def test_uncached_clean_slots_remain_clean():
    x=np.ones((3,2,256),np.float32);arrays=dict(residual=np.full_like(x,2),full=np.full_like(x,3))
    arrays['residual'][0]=0;arrays['full'][0]=0
    values,targets,route=training_batch(x,arrays,np.array([.1,.5,.9]),np.arange(3))
    assert route.tolist()==['clean','residual','full'] and np.all(values[0]==1) and np.all(values[1]==2) and np.all(values[2]==3)
    assert np.all(values[3:]==1) and np.all(targets==1)


def summary(gain=0,clean_drop=0,nmse=.03):
    records=[]
    for view in VIEWS:
        for name in ('identity','refined'):
            accuracy=98.5 if view=='clean' else 80.+(gain if name=='refined' else 0)
            if view=='clean' and name=='refined':accuracy-=clean_drop
            records.append(dict(view=view,readout=name,baseline='identity',dimension='overall',accuracy=accuracy))
    return dict(source_only=True,count=1800,target_reads=0,records=records,
        waveform=[dict(view=view,NMSE=dict(refined=nmse)) for view in VIEWS])


def test_source_rule_reuses_fixed_control_and_requires_both_urban_gains():
    assert d.source_rule(summary(.6),summary())['qualified']
    assert not d.source_rule(summary(.4),summary())['qualified']
    assert not d.source_rule(summary(.6,.2),summary())['qualified']
    assert not d.source_rule(summary(.6,nmse=.032),summary())['qualified']
    bad=summary(.6);bad['target_reads']=1
    with pytest.raises(ValueError):d.source_rule(bad,summary())


def packets(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done))
    values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_seven_view_target_complete_before_scoring(tmp_path):
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ'])
def test_invalid_target_predictions_fail_before_truth(tmp_path,bad):
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


def test_unqualified_source_cannot_access_new_target(tmp_path):
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='x')))
    (root/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=False,target_reads=0)))
    row=root/d.ARM;row.mkdir();(row/'training_complete.json').write_text('{}')
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x'))
