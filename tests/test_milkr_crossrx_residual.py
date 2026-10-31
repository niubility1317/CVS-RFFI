import numpy as np
import pytest
import torch
from experiments.pamf_phase2_practical.milkr import crossrx_residual as model
from experiments.pamf_phase2_practical.milkr.correlation_cca import CorrelationCCA
from experiments.pamf_phase2_practical import teacher_features
from tests.test_milkr_residual_skip import payload


def test_cross_RX_positive_mask_and_loss_exclude_same_RX_same_TX():
    y=torch.tensor([0,0,0,1,1,1]);rx=torch.tensor([0,0,1,0,0,1]);positive,allowed=model.masks(y,rx)
    assert positive[0,2] and not positive[0,1] and not allowed[0,1] and allowed[0,3] and not positive[0,3] and not allowed.diag().any()
    embedding=torch.nn.functional.normalize(torch.randn(6,12),dim=1).requires_grad_();loss=model.contrastive_loss(embedding,y,rx);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(embedding.grad).all()
    with pytest.raises(ValueError,match='cross-RX positive'):model.masks(y,torch.zeros_like(rx))


def test_zero_initial_residual_and_bound_are_explicit():
    a=teacher_features.ordinary(payload(811,170));b=teacher_features.ordinary(payload(812,170));cca=CorrelationCCA.fit(a,b)
    nets=model.networks('cpu');encoder=model.CrossRXResidualCCA(cca.parameters,model.export(nets))
    for path in (False,True):
        result=encoder.transform(a,restored=path);reference=cca.transform(a,restored=path)
        assert all(np.allclose(v,reference[k],atol=1e-14) for k,v in result.items())
    changed=[[list(p) for p in path] for path in model.export(nets)]
    for path in changed:
        for p in path:p[3]=np.full(64,100,np.float32)
    saturated=model.CrossRXResidualCCA(cca.parameters,changed)
    assert all(np.linalg.norm(v,axis=1).max()<=.25+1e-12 for v in saturated.residuals(a,restored=False))


def test_source_only_training_metadata_checkpoint_reload_and_query_independence(tmp_path):
    def dual(seed):
        a=payload(seed,160);b=payload(seed+1,160);a.update({'C4_'+k:v for k,v in b.items() if k!='teacher_phase_logits'})
        for prefix in ('','C4_'):a[prefix+'fft'][:,96:]=0;a[prefix+'f_emb'][:,160:]=0
        return a
    raw=dual(813);restored=dual(815);labels=np.repeat(np.arange(4),40);rx=np.tile(np.repeat([0,1],20),4);ids=np.array([str(i) for i in range(160)])
    bridge=teacher_features.CrossRXResidualBridge.fit(raw,restored,labels=labels,receivers=rx,physical_ids=ids,seed=817,steps=3,device='cpu')
    assert sum(a.size for e in bridge.encoders for p in e.parameters for a in p)==557440
    assert sum(a.size for e in bridge.encoders for path in e.neural for p in path for a in p)==684032
    assert [e.training_summary['seed'] for e in bridge.encoders]==[817+4*100003,817+8*100003]
    for e in bridge.encoders:
        assert e.training_summary['steps']==3 and e.training_summary['paired_exposures']==200 and e.training_summary['query_reads']==e.training_summary['F_updates']==e.training_summary['G_updates']==0
        assert set(e.training_summary['source_TX'])=={0,1,2,3} and e.training_summary['network_parameters']==342016
        assert any(np.any(p[2]) for path in e.neural for p in path)
    bridge.save(tmp_path/'model.npz');loaded=teacher_features.CrossRXResidualBridge.load(tmp_path/'model.npz')
    with pytest.raises(ValueError,match='checkpoint'):teacher_features.CorrelationEnsembleBridge.load(tmp_path/'model.npz')
    for path in (False,True):
        a=bridge.transform(raw,restored=path,old_columns=[0,2,4,5]);b=loaded.transform(raw,restored=path,old_columns=[0,2,4,5]);one=loaded.transform({k:v[2:3] for k,v in raw.items()},restored=path,old_columns=[0,2,4,5])
        assert all(np.array_equal(v,b[k]) and np.allclose(v[2],one[k][0]) for k,v in a.items())
        assert np.array_equal(a['teacher_probs'],teacher_features.probabilities(raw['teacher_phase_logits'],[0,2,4,5]))
    bad=ids.copy();bad[20]=bad[0]
    with pytest.raises(ValueError,match='changes label/RX'):model.learn(None,{}, {},labels,rx,bad,819,steps=1,device='cpu')


def test_source_contrastive_residual_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen
    from experiments.pamf_phase2_practical.milkr.orbit_ensemble import fit
    assert screen.fit_function({'kernel_method':'crossrx_residual_ensemble'}) is fit
    assert teacher_features.bridge_type('crossrx_residual_ensemble') is teacher_features.CrossRXResidualBridge
    marker=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'crossrx_residual_ensemble'},None,None,'cpu') is marker
