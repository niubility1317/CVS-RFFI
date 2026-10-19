import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_hardness_refine import design as d
from experiments.pamf_hardness_refine.weighting import ObservedFocus
from experiments.pamf_hardness_refine.models import JointResidual,save_checkpoint,load_checkpoint
from experiments.pamf_hardness_refine.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS
from experiments.pamf_reception_restore.models import ComplexTFUNet


def test_mean_control_uses_all_observed_losses_and_records_exposures():
    loss=torch.tensor([1.,2.,3.,4.],requires_grad=True);rx=torch.tensor([1,3,4,8])
    focus=ObservedFocus(d.ARMS[0]);value=focus(loss,rx);value.backward()
    assert value.item()==2.5 and torch.equal(loss.grad,torch.full((4,),.25))
    state=focus.state();assert state['observed_exposures']==state['selected_exposures']==4
    assert state['updates']==0 and sum(state['RX_exposures'].values())==4
    assert state['global_RX_weights']==pytest.approx([.2]*5)


def test_cvar_only_selected_quartile_receives_additional_gradient():
    loss=torch.arange(1.,9.,requires_grad=True);rx=torch.tensor([1,3,4,6,8,1,3,4])
    focus=ObservedFocus(d.ARMS[1]);value=focus(loss,rx);value.backward()
    assert value.item()==7.5 and torch.equal(loss.grad,torch.tensor([0.,0.,0.,0.,0.,0.,.5,.5]))
    state=focus.state();assert state['selected_fraction']==.25 and state['selected_exposures']==2
    assert state['selected_RX_exposures']['3']==state['selected_RX_exposures']['4']==1
    assert state['updates']==0 and state['TX_labels_read']==state['target_reads']==0


def test_focus_excludes_clean_half_and_handles_small_last_batch():
    all_pairs=torch.tensor([1.,4.,100.,200.],requires_grad=True)
    focus=ObservedFocus(d.ARMS[1]);value=focus(all_pairs[:2],torch.tensor([1,3]));value.backward()
    assert torch.equal(all_pairs.grad,torch.tensor([0.,1.,0.,0.]))
    assert focus.state()['selected_exposures']==1


def test_rx_weights_detach_and_renormalize_only_present_source_groups():
    loss=torch.tensor([1.,1.,5.,5.],requires_grad=True);rx=torch.tensor([1,1,3,3])
    focus=ObservedFocus(d.ARMS[2]);value=focus(loss,rx);value.backward()
    weights=torch.softmax(torch.tensor([.01,.05]),0)
    assert value.item()==pytest.approx(float((weights*torch.tensor([1.,5.])).sum()))
    assert loss.grad.tolist()==pytest.approx([float(weights[0]/2)]*2+[float(weights[1]/2)]*2)
    assert not focus.logq.requires_grad and focus.logq.grad is None
    state=focus.state();assert state['updates']==1 and state['selected_exposures']==4
    assert state['global_RX_weights'][1]>state['global_RX_weights'][0]
    single=torch.tensor([2.,4.],requires_grad=True)
    result=focus(single,torch.tensor([8,8]));result.backward()
    assert result.item()==3. and single.grad.tolist()==[.5,.5]


@pytest.mark.parametrize('kind',['bad_RX','length','dtype','nonfinite','empty'])
def test_invalid_source_loss_or_rx_rejected(kind):
    loss=torch.ones(2);rx=torch.tensor([1,3])
    if kind=='bad_RX':rx[0]=2
    elif kind=='length':rx=rx[:1]
    elif kind=='dtype':rx=rx.float()
    elif kind=='nonfinite':loss[0]=float('nan')
    else:loss=loss[:0];rx=rx[:0]
    with pytest.raises(ValueError):ObservedFocus(d.ARMS[2])(loss,rx)


def test_strategy_configuration_has_matched_budget_and_no_inference_rx():
    for arm in d.ARMS:
        c=d.configuration(arm)
        assert c['epochs']==10 and c['updates']==2470 and c['focus_weight']==.5 and c['posterior_weight']==.3
        assert c['teacher_seeds']==[2026092701,2026092702,2026092703]
        assert c['source_RX']==[1,3,4,6,8] and c['coarse_updates']==2470
    assert len(set(d.GPUS.values()))==3
    with pytest.raises(ValueError):ObservedFocus('unknown')


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
 ('head_fit_updates',1),('parent_corrector_epochs',0),('coarse_updates',1),('updates',2471),('parent_coarse_epochs',50)])
def test_checkpoint_permissions_and_budget(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def test_checkpoint_preserves_focus_audit_and_freezes_model_for_inference(tmp_path):
    path=tmp_path/'m.pt';focus=ObservedFocus(d.ARMS[2]);focus(torch.tensor([1.,2.]),torch.tensor([1,3]))
    save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARMS[2],0,0,focus.state())
    model,item=load_checkpoint(path,d.ARMS[2],'cpu')
    assert item['weighting_state']['updates']==1 and not model.training and all(not p.requires_grad for p in model.parameters())
    item['config']['focus_weight']=1.;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARMS[2],'cpu')


def summary(gain=0,nmse=.028,drop=0):
    records=[]
    for view in VIEWS:
        for name in ('identity','refined'):
            accuracy=98.5 if view=='clean' else 80.+(gain if name=='refined' else 0)
            if view=='clean' and name=='refined':accuracy-=drop
            records.append(dict(view=view,readout=name,baseline='identity',dimension='overall',accuracy=accuracy))
    return dict(source_only=True,count=1800,target_reads=0,records=records,waveform=[dict(view=v,NMSE=dict(refined=nmse)) for v in VIEWS])


def test_three_arm_source_selection_and_no_relaxed_threshold():
    result=d.source_rule(dict(zip(d.ARMS,(summary(1),summary(2),summary(3)))),summary())
    assert result['selected_arm']==d.ARMS[2]
    result=d.source_rule(dict(zip(d.ARMS,(summary(.2),summary(1,.034),summary(1,.03,.2)))),summary())
    assert not result['qualified'] and result['selected_arm'] is None
    with pytest.raises(ValueError):d.source_rule({d.ARM:summary(1)},summary())
    bad=summary(1)
    for row in bad['records']:
        if row['view']=='practical_mid_urban' and row['readout']=='refined':row['accuracy']=79.
    assert not d.source_rule({a:bad for a in d.ARMS},summary())['qualified']


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


def test_unselected_third_arm_cannot_access_target(tmp_path):
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='x')))
    (root/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=True,selected_arm=d.ARMS[0],target_reads=0)))
    row=root/d.ARMS[2];row.mkdir();(row/'training_complete.json').write_text('{}')
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[2]))
