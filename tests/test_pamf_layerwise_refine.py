import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_layerwise_refine import design as d
from experiments.pamf_layerwise_refine.models import JointResidual,save_checkpoint,load_checkpoint
from experiments.pamf_layerwise_refine.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS
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
 ('head_fit_updates',1),('parent_corrector_epochs',20),('corrector_initialization','R03'),('coarse_updates',1),('updates',7411),('parent_coarse_epochs',50)])
def test_joint_checkpoint_permissions_and_budget(tmp_path,field,value):
    path=tmp_path/'m.pt';model=JointResidual(ComplexTFUNet(),.1);save_checkpoint(path,model,None,d.ARM,0,0)
    v=torch.load(path,weights_only=False);v[field]=value;torch.save(v,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def summary(gain=0,nmse=.028,drop=0):
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
    assert result['selected_arm']==d.ARMS[1]


def test_source_selector_keeps_waveform_and_clean_limits():
    result=d.source_rule(dict(zip(d.ARMS,(summary(1,.034),summary(1,.03,.2)))),summary())
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


def test_other_environment_regression_cannot_qualify():
    current=summary(1,.028)
    for row in current['records']:
        if row['view']=='practical_high' and row['readout']=='refined':row['accuracy']=79.
    assert not d.source_rule({a:current for a in d.ARMS},summary(nmse=.028))['qualified']


def test_waveform_limit_does_not_compound_with_current_reference():
    control=summary(nmse=.02964)
    result=d.source_rule({a:summary(1,nmse=.034) for a in d.ARMS},control)
    assert not result['qualified']


@pytest.mark.parametrize('field,value',[('teacher_seeds',[2026092701]),('teacher_aggregation','unknown')])
def test_teacher_configuration_cannot_change_after_freeze(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item['config'][field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


class ProbeTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.core=torch.nn.Module();self.core.id_backbone=torch.nn.Module()
        self.core.id_backbone.t1=torch.nn.Conv1d(2,4,1);self.core.id_backbone.t2=torch.nn.Conv1d(4,4,1)
        self.calls=0;self.eval().requires_grad_(False)
    def features(self,x):
        self.calls+=1
        h=self.core.id_backbone.t2(torch.tanh(self.core.id_backbone.t1(x)))
        return h.mean(-1).repeat(1,40)
    def classify_features(self,x):return x[:,:6]


def test_layer_loss_preserves_input_gradient_detaches_clean_and_removes_hooks():
    from experiments.pamf_layerwise_refine.guidance import guidance
    teachers={s:ProbeTeacher() for s in d.TEACHER_SEEDS}
    clean=torch.randn(2,2,256,requires_grad=True)
    estimate=(clean.detach().repeat(2,1,1)+.2*torch.randn(4,2,256)).requires_grad_(True)
    with torch.no_grad():references=torch.stack([m.features(clean) for m in teachers.values()],1).repeat(2,1,1)
    for model in teachers.values():model.calls=0
    combined,feature,posterior,middle=guidance(teachers,estimate,references,d.ARM,clean)
    assert float(middle.detach().sum())>0
    assert torch.allclose(combined,feature+.3*posterior+.1*middle)
    middle.mean().backward()
    assert torch.isfinite(estimate.grad).all() and float(estimate.grad.abs().sum())>0 and clean.grad is None
    for model in teachers.values():
        assert model.calls==2 and all(p.grad is None for p in model.parameters())
        assert not model.core.id_backbone.t1._forward_hooks and not model.core.id_backbone.t2._forward_hooks


def test_control_skips_middle_clean_forward_and_missing_source_clean_is_rejected():
    from experiments.pamf_layerwise_refine.guidance import guidance
    teachers={s:ProbeTeacher() for s in d.TEACHER_SEEDS};x=torch.randn(2,2,256)
    with torch.no_grad():references=torch.stack([m.features(x) for m in teachers.values()],1)
    for model in teachers.values():model.calls=0
    _,_,_,middle=guidance(teachers,x,references,d.ARMS[0])
    assert torch.equal(middle,torch.zeros(2)) and all(m.calls==1 for m in teachers.values())
    with pytest.raises(ValueError):guidance(teachers,x,references,d.ARM)


def test_hooks_are_removed_when_the_native_feature_path_fails():
    from experiments.pamf_layerwise_refine.guidance import capture
    model=ProbeTeacher()
    def fail(x):raise RuntimeError('probe')
    model.features=fail
    with pytest.raises(RuntimeError):capture(model,torch.randn(2,2,256))
    assert not model.core.id_backbone.t1._forward_hooks and not model.core.id_backbone.t2._forward_hooks


def test_fresh_correction_is_an_exact_coarse_preserving_start():
    torch.set_num_threads(2);coarse=ComplexTFUNet().eval();model=JointResidual(coarse,.1169).eval()
    x=torch.randn(2,2,256)
    with torch.no_grad():assert torch.equal(model(x),coarse(x))


def test_fixed30_epoch_schedule_reuses_only_existing20_cache_epochs():
    from experiments.pamf_layerwise_refine.runner import learning_rate
    assert [d.cache_epoch(i) for i in (1,20,21,30)]==[1,20,1,10]
    for invalid in (0,31):
        with pytest.raises(ValueError):d.cache_epoch(invalid)
    assert learning_rate(246)==pytest.approx(3e-5)
    assert learning_rate(d.UPDATES-1)==pytest.approx(3e-6)
    assert learning_rate(d.UPDATES-1)*(2e-4/3e-5)==pytest.approx(2e-5)
    with pytest.raises(ValueError):learning_rate(7410)


@pytest.mark.parametrize('field,value',[('source_V_reads',1),('target_reads',1),('optimizer_updates',1),('count',100),('scale',float('nan'))])
def test_calibration_rejects_role_or_scale_mismatch(tmp_path,field,value):
    from experiments.pamf_layerwise_refine.models import source_scale
    root=tmp_path/d.CALIBRATION_RUN;root.mkdir();path=root/'residual_scale.json';parent={'run':d.PARENT_RUN}
    item=dict(status='SOURCE_SCALE_FROZEN',count=8192,source_training_only=True,parent=parent,
        source_V_reads=0,TX_labels_read=0,target_reads=0,optimizer_updates=0,scale=.1169)
    path.write_text(json.dumps(item));assert source_scale(path,parent)['scale']==.1169
    item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_scale(path,parent)


def test_final_budget_and_fresh_parent_cannot_be_replaced(tmp_path):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1169),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item['updates']=item['coarse_updates']=7411;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')
