import copy
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .core import paired,decisions,waveform


def check(recipe_args):
    from SSDG import train_ssdg as native
    from .run import provenance
    from experiments.cvs_ordinary_fold20 import design as old,model as api
    a=SimpleNamespace(**recipe_args);base=native.build_aug_base_cfg(a)
    aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,13,a)
    expected=.03508404675089638
    assert abs(aug.p_pa-expected)<1e-14 and not aug.enable_class_signature and aug.p_dac==0
    x=torch.randn(16,2,256);y=torch.arange(16)%6;before=x.clone();aug.p_pa=1.
    state=torch.get_rng_state().clone();full,no_pa,s=paired(aug,x,y,d.DRAW_SEED,1)
    assert torch.equal(state,torch.get_rng_state()) and torch.equal(before,x) and bool((s>0).all())
    other=paired(aug,x,y.flip(0),d.DRAW_SEED,1)
    assert all(torch.equal(v,w) for v,w in zip((full,no_pa,s),other))
    negatives=0
    for key,value in [('p_dac',.1),('enable_class_signature',True),('p_pa',0.)]:
        saved=getattr(aug,key);setattr(aug,key,value)
        try:paired(aug,x,y,d.DRAW_SEED,1);raise AssertionError('Invalid intervention accepted')
        except ValueError:negatives+=1
        finally:setattr(aug,key,saved)
    saved_general=aug._apply_general_channel;counter=[0]
    def divergent(t):counter[0]+=1;return saved_general(t)+counter[0]
    aug._apply_general_channel=divergent
    try:paired(aug,x,y,d.DRAW_SEED,1);raise AssertionError('Different upstream accepted')
    except ValueError:negatives+=1
    finally:aug._apply_general_channel=saved_general
    m=api.build(old.config(old.rows()[0]),torch.device('cpu')).eval().requires_grad_(False)
    snapshot={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():
        for value in (x,full,no_pa):
            pred,margin=decisions(m,value,y);assert len(pred)==16 and len(margin)==16
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in snapshot.items())
    assert all(len(v)==16 for v in waveform(full).values())
    fixture=dict(completion=dict(config=old.config(old.rows()[0]),epoch=20,optimizer_steps=4440,initialization='scratch',checkpoint_sources=[],ancestors=[],target_access=False,head_fit_updates=0),
        initialization=dict(scratch_only=True,checkpoint_sources=[],target_contact=False),
        source_contract=dict(held_RX=1,source_receivers=[3,4,6,8],role_ids=dict(L_s=list(range(5040)),U_s=list(range(45360)),V=list(range(21600)))),
        source_frozen=dict(epoch=20,selection='final_only',target_access=False),
        result=dict(target_access=False,U_truth_read=False,default_head_joint_training=True))
    original=d.read
    try:
        d.read=lambda p:fixture[Path(p).stem];provenance(1)
        for section,key,value in [('completion','epoch',50),('completion','ancestors',['other']),('completion','head_fit_updates',1),('initialization','target_contact',True),('result','U_truth_read',True)]:
            saved=fixture[section][key];fixture[section][key]=value
            try:provenance(1);raise AssertionError('Illegal source checkpoint accepted')
            except ValueError:negatives+=1
            finally:fixture[section][key]=saved
    finally:d.read=original
    c=d.config(d.rows()[0]);c['target_access']=True
    try:d.validate(c);raise AssertionError('Target access accepted')
    except ValueError:negatives+=1
    good=[dict(held_RX=r,retention_gain_pp=1.) for r in d.RX];assert d.qualify(good)['status']=='PASS'
    for values in ([.1]*5,[2.,2.,2.,0.,0.],[-.6,3.,3.,3.,3.]):
        assert d.qualify([dict(held_RX=r,retention_gain_pp=v) for r,v in zip(d.RX,values)])['status']=='FAIL';negatives+=1
    return dict(status='PASS',synthetic_network_forwards=48,synthetic_augmented_input_packets=96,negative_cases=negatives,
        actual_native_PA_pair_identical_upstream=True,labels_do_not_change_views=True,frozen_model_unchanged=True,
        real_IQ_reads=0,checkpoint_loads=0,optimizer_updates=0,target_access=False)
