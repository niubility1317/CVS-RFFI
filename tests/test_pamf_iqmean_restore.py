import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_iqmean_restore import design as d
from experiments.pamf_iqmean_restore.models import FixedIQMean,validate_members
from experiments.pamf_iqmean_restore.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS


class Factor(torch.nn.Module):
    def __init__(self,scale):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(float(scale)))
    def forward(self,x):return x*self.scale


def test_fixed_iq_mean_is_frozen_and_independent_per_record():
    model=FixedIQMean(Factor(2),Factor(-1));x=torch.randn(3,2,256)
    assert torch.equal(model(x),.5*(x*2)+.5*(-x))
    assert torch.equal(model(x),torch.cat([model(v[None]) for v in x]))
    assert not model.training and not model.base.training and not model.partner.training
    assert not model(x).requires_grad and all(not p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):model.train()


def test_equal_iq_mean_cannot_exceed_mean_of_same_clean_squared_errors():
    clean=torch.randn(4,2,256);a=torch.randn_like(clean);b=torch.randn_like(clean)
    mean=.5*a+.5*b
    lhs=(mean-clean).square().mean((1,2));rhs=.5*((a-clean).square()+(b-clean).square()).mean((1,2))
    assert torch.all(lhs<=rhs+1e-6)
    # This waveform inequality deliberately makes no claim about classifier accuracy.


def members(root):
    for run,commit,arms,epoch,updates,l,u,coarse,corr in (
        (d.BASE_RUN,d.BASE_COMMIT,('posterior03',),10,2470,63000,567000,30,30),
        (d.PARTNER_RUN,d.PARTNER_COMMIT,('final_guidance','layerwise_guidance'),30,7410,189000,1701000,50,30)):
        r=root/run;r.mkdir();(r/'launch.json').write_text(json.dumps(dict(commit=commit)))
        (r/'complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True)))
        (r/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,target_reads=0)))
        for arm in arms:
            row=r/arm;row.mkdir();value=dict(epochs=epoch,updates=updates,coarse_updates=updates,
                L_physical_exposures=l,U_physical_exposures=u,IQ_examples=2*(l+u),coarse_epochs=coarse,
                total_correction_epochs=corr,F_seeds=list(d.F_SEEDS),F_optimizer_updates=0,head_fit_updates=0,
                TX_labels_read=0,target_reads=0)
            (row/'training_complete.json').write_text(json.dumps(value))


def test_completed_source_members_and_historical_cost_accounting(tmp_path):
    members(tmp_path);assert len(validate_members(tmp_path))==3
    for arm in d.ARMS:
        c=d.configuration(arm);assert c['weights']==[.5,.5] and c['new_training_epochs']==0
        assert c['inherited_member_component_epochs']==[60,80] and c['unique_inherited_component_epochs']==120
        assert c['new_trainable_parameters']==0 and c['inference_member_count']==2
    with pytest.raises(ValueError):d.configuration('unregistered')


@pytest.mark.parametrize('field,value',[('epochs',20),('updates',7410),('TX_labels_read',1),('target_reads',1),
    ('head_fit_updates',1),('F_optimizer_updates',1),('L_physical_exposures',630000),('F_seeds',[2026092701])])
def test_member_permissions_and_fixed_budget_rejected(tmp_path,field,value):
    members(tmp_path);path=tmp_path/d.BASE_RUN/'posterior03/training_complete.json'
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):validate_members(tmp_path)


def test_wrong_member_code_or_target_side_origin_rejected(tmp_path):
    members(tmp_path);path=tmp_path/d.PARTNER_RUN/'launch.json';path.write_text(json.dumps(dict(commit='wrong')))
    with pytest.raises(ValueError):validate_members(tmp_path)
    path.write_text(json.dumps(dict(commit=d.PARTNER_COMMIT)))
    path=tmp_path/d.PARTNER_RUN/'source_complete.json';path.write_text(json.dumps(dict(status='ANALYZED',source_only=False,target_reads=1)))
    with pytest.raises(ValueError):validate_members(tmp_path)


def summary(gain=0,nmse=.028,drop=0):
    records=[]
    for view in VIEWS:
        for name in ('identity','refined'):
            acc=98.5 if view=='clean' else 80.+(gain if name=='refined' else 0)
            if view=='clean' and name=='refined':acc-=drop
            records.append(dict(view=view,readout=name,baseline='identity',dimension='overall',accuracy=acc))
    return dict(source_only=True,count=1800,target_reads=0,records=records,
                waveform=[dict(view=v,NMSE=dict(refined=nmse)) for v in VIEWS])


def test_source_selection_and_no_requalification_of_single_members():
    result=d.source_rule(dict(zip(d.ARMS,(summary(2),summary(1)))),summary())
    assert result['qualified'] and result['selected_arm']==d.ARMS[0]
    result=d.source_rule(dict(zip(d.ARMS,(summary(.2),summary(1)))),summary())
    assert result['selected_arm']==d.ARMS[1]
    result=d.source_rule(dict(zip(d.ARMS,(summary(1,.034),summary(1,.03,.2)))),summary())
    assert not result['qualified'] and result['selected_arm'] is None
    with pytest.raises(ValueError):d.source_rule({'posterior03':summary(2)},summary())


def test_other_environment_regression_and_compounded_nmse_cannot_qualify():
    bad=summary(1)
    for r in bad['records']:
        if r['view']=='practical_high' and r['readout']=='refined':r['accuracy']=79
    assert not d.source_rule({a:bad for a in d.ARMS},summary())['qualified']
    assert not d.source_rule({a:summary(1,.034) for a in d.ARMS},summary(nmse=.03122))['qualified']


def packets(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done))
    values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_all_seven_target_views_and_three_native_heads_before_truth(tmp_path):
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ'])
def test_invalid_target_packets_rejected_before_truth(tmp_path,bad):
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


def source_stage(tmp_path):
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='x')))
    (root/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=True,selected_arm=d.ARMS[0],target_reads=0)))
    for arm in d.ARMS:
        row=root/arm;row.mkdir();(row/'composition.json').write_text(json.dumps(d.configuration(arm)))
        value=dict(status='SOURCE_INFERENCE_COMPLETE',count=1800,new_training_epochs=0,parameter_updates=0,
            F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0,state_updates=0)
        (row/'inference_complete.json').write_text(json.dumps(value))
    return root


def test_only_frozen_selected_composition_may_access_target(tmp_path,monkeypatch):
    import experiments.pamf_iqmean_restore.target as target
    root=source_stage(tmp_path);monkeypatch.setattr(target,'validate_members',lambda root:[])
    source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[0]))
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[1]))
    path=root/d.ARMS[0]/'composition.json';value=json.loads(path.read_text());value['weights']=[.9,.1];path.write_text(json.dumps(value))
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[0]))


@pytest.mark.parametrize('field',['parameter_updates','TX_labels_read','target_reads','head_fit_updates'])
def test_source_state_updates_or_truth_access_forbid_target(tmp_path,monkeypatch,field):
    import experiments.pamf_iqmean_restore.target as target
    root=source_stage(tmp_path);monkeypatch.setattr(target,'validate_members',lambda root:[])
    path=root/d.ARMS[0]/'inference_complete.json';value=json.loads(path.read_text());value[field]=1;path.write_text(json.dumps(value))
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[0]))
