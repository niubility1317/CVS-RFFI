import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_semantic_feedback import design as d
from experiments.pamf_semantic_feedback.models import FeedbackCore,JointFeedback,save_checkpoint,load_checkpoint
from experiments.pamf_semantic_feedback.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet

class TinyReader(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv1d(2,4,1);self.bn=nn.BatchNorm1d(4);self.head=nn.Linear(160,6)

    def features(self,x):return torch.nn.functional.adaptive_avg_pool1d(self.bn(self.conv(x)),40).flatten(1)


def test_zero_modulation_exactly_preserves_parent_core_and_has_identical_trainable_budget():
    torch.manual_seed(1);parent=Core(incoming=4);nn.init.normal_(parent.out.weight,0,.02)
    core=FeedbackCore();result=core.load_state_dict(parent.state_dict(),strict=False)
    assert set(result.missing_keys)=={'modulation.'+k for k in core.modulation.state_dict()} and not result.unexpected_keys
    x=torch.randn(2,4,256);code=torch.randn(2,160)
    assert torch.equal(parent(x),core(x,code))
    for arm in d.ARMS:
        model=JointFeedback(ComplexTFUNet(),.1,arm,TinyReader())
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==d.TRAINABLE_PARAMETERS
    assert d.TRAINABLE_PARAMETERS==2244008


def test_modulation_uses_per_frame_condition_and_backpropagates_to_code():
    core=FeedbackCore();nn.init.normal_(core.out.weight,0,.02);nn.init.normal_(core.modulation[-1].weight,0,.02)
    x=torch.randn(1,4,256).expand(2,-1,-1).clone();code=torch.randn(2,160,requires_grad=True)
    y=core(x,code)
    assert not torch.allclose(y[0],y[1])
    y.square().mean().backward()
    assert code.grad is not None and torch.isfinite(code.grad).all() and code.grad.abs().sum()>0


def test_runtime_F_stays_eval_frozen_and_outside_optimizer():
    reader=TinyReader();model=JointFeedback(ComplexTFUNet(),.1,d.ARMS[1],reader)
    nn.init.normal_(model.correction.out.weight,0,.02)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    model.train();assert not reader.training and all(not p.requires_grad for p in reader.parameters())
    parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=1e-4)
    model(torch.randn(2,2,256)).square().mean().backward()
    assert model.correction.modulation[-1].weight.grad.abs().sum()>0
    optimizer.step()
    assert all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())


def test_semantic_runtime_requires_reader_and_current_frame_predictions_are_batch_independent(tmp_path):
    with pytest.raises(ValueError):JointFeedback(ComplexTFUNet(),.1,d.ARMS[1])
    reader=TinyReader();model=JointFeedback(ComplexTFUNet(),.1,d.ARMS[1],reader).eval()
    nn.init.normal_(model.correction.out.weight,0,.02);nn.init.normal_(model.correction.modulation[-1].weight,0,.02)
    x=torch.randn(2,2,256)
    with torch.no_grad():
        y=model(x);alone=torch.cat([model(v[None]) for v in x])
    assert torch.allclose(y,alone,atol=1e-6,rtol=1e-5)
    path=tmp_path/'m.pt';save_checkpoint(path,model,None,d.ARMS[1],0,0)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARMS[1],'cpu')
    loaded,item=load_checkpoint(path,d.ARMS[1],'cpu',TinyReader())
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    with torch.no_grad():assert torch.equal(loaded(x),y)
    item['runtime_F_seed']=2026092702;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARMS[1],'cpu',TinyReader())


def test_registered_feedback_config_changes_only_code_and_records_extra_runtime_cost():
    left,right=(d.configuration(a) for a in d.ARMS)
    assert left['epochs']==right['epochs']==10 and left['updates']==right['updates']==2470
    assert left['trainable_parameters']==right['trainable_parameters']==2244008
    assert left['runtime_F_seed'] is None and right['runtime_F_seed']==2026092701
    assert left['loss']==right['loss'] and left['posterior_weight']==.3
    with pytest.raises(ValueError):d.configuration('unknown')


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
 ('head_fit_updates',1),('parent_corrector_epochs',0),('coarse_updates',1),('updates',2471),('parent_coarse_epochs',50),('modulation_updates',1)])
def test_checkpoint_permissions_and_budget(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointFeedback(ComplexTFUNet(),.1,d.ARM),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')




def summary(gain=0,nmse=.028,drop=0):
    records=[]
    for view in VIEWS:
        for name in ('identity','refined'):
            accuracy=98.5 if view=='clean' else 80.+(gain if name=='refined' else 0)
            if view=='clean' and name=='refined':accuracy-=drop
            records.append(dict(view=view,readout=name,baseline='identity',dimension='overall',accuracy=accuracy))
    return dict(source_only=True,count=1800,target_reads=0,records=records,waveform=[dict(view=v,NMSE=dict(refined=nmse)) for v in VIEWS])


def test_prospective_low_first_rule_preserves_mid_clean_waveform_and_other_views():
    result=d.source_rule(dict(zip(d.ARMS,(summary(1),summary(2)))),summary())
    assert result['selected_arm']==d.ARMS[1]
    result=d.source_rule(dict(zip(d.ARMS,(summary(.2),summary(1,.034)))),summary())
    assert not result['qualified']
    for bad in ('mid','other','clean','truth'):
        item=summary(1)
        for row in item['records']:
            if row['readout']=='refined' and ((bad=='mid' and row['view']=='practical_mid_urban') or (bad=='other' and row['view']=='practical_high')):row['accuracy']=79.
            if bad=='clean' and row['readout']=='refined' and row['view']=='clean':row['accuracy']=98.
        if bad=='truth':item['target_reads']=1
        if bad=='truth':
            with pytest.raises(ValueError):d.source_rule({a:item for a in d.ARMS},summary())
        else:assert not d.source_rule({a:item for a in d.ARMS},summary())['qualified']
    good=summary(1)
    for row in good['records']:
        if row['readout']=='refined' and row['view']=='practical_mid_urban':row['accuracy']=80.
    assert d.source_rule({a:good for a in d.ARMS},summary())['qualified']
    with pytest.raises(ValueError):d.source_rule({d.ARM:good},summary())


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


def test_unselected_second_arm_cannot_access_target(tmp_path):
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='x')))
    (root/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=True,selected_arm=d.ARMS[0],target_reads=0)))
    row=root/d.ARMS[1];row.mkdir();(row/'training_complete.json').write_text('{}')
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[1]))
