import json
import numpy as np
import pytest
from tests.test_milkr_semantic_residual import example
from experiments.pamf_phase2_practical.milkr import calibrated_k1 as k1,nested_fisher as nested
from experiments.pamf_phase2_practical.milkr.paired_cca import PairedCCA,DIMS
from experiments.pamf_phase2_practical.milkr.source_class_metric import within_class_covariances,fuse_metric
from experiments.pamf_phase2_practical.milkr_identity_source import source_train_indices,assess


def test_K1_mean_loss_scale_uses_only_allocated_support_class_count():
    rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75],ridge_reference_classes=6)
    for k in (1,3):
        x,_,y,ids=example(k=k);x['sig_views']=np.random.default_rng(91).normal(size=(len(y),336))
        ridge=.1*(int(y.max())+1)/6 if k==1 else .1
        state=k1.fit(x,y,ids,fixed=rule);reference=nested.fit(x,y,ids,fixed=dict(rule,ridge=ridge))
        assert np.array_equal(state.alpha,reference.alpha) and np.array_equal(state.score(x),reference.score(x))
        json.dumps(state.selection,allow_nan=False)
    with pytest.raises(ValueError,match='reference'):k1.fit(x,y,ids,fixed=dict(rule,ridge_reference_classes=0))


def test_subset_class_covariance_handles_noncontiguous_labels_and_rank32():
    rng=np.random.default_rng(181);labels=np.repeat([0,2,3,5],24);features={name:rng.normal(size=(len(labels),d)) for name,d in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160),('sig_views',336)]}
    cca=PairedCCA([(np.zeros(d),np.zeros(d),np.eye(d,32),np.eye(d,32)) for d in DIMS],32)
    with pytest.raises(ValueError,match='Six legal'):within_class_covariances(cca,features,features,labels)
    cov=within_class_covariances(cca,features,features,labels,allow_subset=True)
    assert all(c.shape==(32,32) and np.isfinite(c).all() for c in cov)
    fused=fuse_metric(cca,cov,.5);assert fused.rank==32
    for value in fused.transform(features,restored=False).values():assert np.isfinite(value).all()


def test_encoder_source_fit_excludes_held_TX_on_every_receiver_and_both_epochs():
    rx=np.tile(np.repeat([1,3,4,6,8],6),2);labels=np.tile(np.arange(6),10)
    train=source_train_indices(rx,labels,4,[1,2])
    assert len(train)==32 and not np.isin(labels[train],[1,2]).any() and (rx[train]!=4).all()
    assert set(rx[train])=={1,3,6,8} and set(labels[train])=={0,3,4,5}


def test_K1_only_source_gain_cannot_pass_the_all_K_source_gate():
    def row(k,sp,qp,old,new):return dict(k=k,S=sp,Q=qp,H=2*old*new/(old+new),old_accuracy=old,new_accuracy=new)
    control=[row(k,sp,qp,.7,.5) for k in (1,5,20) for sp in (0,1) for qp in (0,1)]
    improved=[row(k,sp,qp,.72,.56 if qp else .53) for k in (1,5,20) for sp in (0,1) for qp in (0,1)]
    assert assess(improved,control)['eligible'] is True
    partial=[r if r['k']!=20 else row(20,r['S'],r['Q'],.7,.5) for r in improved]
    assert assess(partial,control)['eligible'] is False
