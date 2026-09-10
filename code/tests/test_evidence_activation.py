import copy
from dataclasses import asdict
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.evidence_head import EvidenceHead
from cvsrffi.evidence_head_training import assert_epoch_activation,mechanism_manifest
from cvsrffi.evidence_diagnostics import source_condition_probe,score_change_diagnostics,FixedReadoutBaseline


def ready_head():
    torch.manual_seed(7)
    h=EvidenceHead(3,6,dict(variant='H5',covariance_rank=2))
    h.error.calibrated.fill_(True);h.error.coefficients.fill_(.01)
    h.response.fit_state_domain(torch.tensor([[0.,0.],[3.,3.]]),source_training=True)
    with torch.no_grad():
        h.response.mean.normal_();h.response.class_slopes.normal_(std=.2)
        h.pair.network[-1].weight.normal_(std=.1)
    h.eval()
    x=torch.randn(9,2,32);aux={'feat_joint':torch.randn(9,6)}
    return h,x,aux


def test_active_configuration_parameters_change_consumed_values():
    h,x,aux=ready_head();base=h(x,aux)
    for field,value,key in [('pair_strength',.9,'scores'),('pair_anchor',8.,'scores'),
                            ('observation_floor',.2,'observation_variance'),('observation_ceiling',.005,'observation_variance')]:
        conf=asdict(h.config);conf[field]=value
        other=EvidenceHead(3,6,conf);other.load_state_dict(h.state_dict());other.eval()
        assert not torch.allclose(base[key],other(x,aux)[key]),field
    other=copy.deepcopy(h);other.state_error.mul_(4)
    assert not torch.allclose(base['state_covariance'],other(x,aux)['state_covariance'])
    other=copy.deepcopy(h)
    with torch.no_grad():other.factor.zero_()
    assert not torch.allclose(base['covariance'],other(x,aux)['covariance'])
    labels=torch.arange(9)%3;ids=[f'p{i}' for i in range(9)]
    base_loss,stats=h.extra_loss(base,labels,ids)
    for field in ('nll_weight','response_regularization','support_weight','prior_precision'):
        conf=asdict(h.config);conf[field]*=3
        other=EvidenceHead(3,6,conf);other.load_state_dict(h.state_dict());other.eval()
        loss,_=other.extra_loss(other(x,aux),labels,ids)
        assert not torch.allclose(base_loss,loss),field
    h.train();partial=h(x,aux)
    assert 0<partial['observed'].sum()<partial['observed'].numel()
    manifest=mechanism_manifest(h)
    assert manifest['inactive_by_ablation']==[]
    assert_epoch_activation(h,{'evidence/observation_active':1,'evidence/correlation_energy':1,'evidence/support_queries':3})
    with pytest.raises(RuntimeError,match='support_queries'):
        assert_epoch_activation(h,{'evidence/observation_active':1,'evidence/correlation_energy':1})


def test_source_probes_and_same_feature_strong_baselines():
    x=torch.tensor([[1.,0.],[2.,0.],[0.,1.],[0.,2.],[3.,0.],[0.,3.]])
    y=torch.tensor([0,0,1,1,0,1])
    probe=source_condition_probe(x,y,x,y)
    assert probe['accuracy']==1
    with pytest.raises(ValueError):source_condition_probe(x,y,x,y,val_role='target')
    for kind in ('cosine','diagonal_gaussian','linear_ridge'):
        baseline=FixedReadoutBaseline(x,y,role='L_s',kind=kind)
        assert torch.equal(baseline.predict(x).argmax(-1),y)
        torch.testing.assert_close(baseline.predict(x),torch.cat([baseline.predict(r[None]) for r in x]))
    stats=score_change_diagnostics(torch.tensor([[2.,0.],[2.,0.]]),torch.tensor([[0.,2.],[0.,2.]]),torch.tensor([1,0]),role='V')
    assert stats['rescue'].tolist()==[True,False] and stats['harm'].tolist()==[False,True]
