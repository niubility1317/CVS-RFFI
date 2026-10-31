"""Official low-dimensional ViM arithmetic on synthetic classifier coordinates."""
from copy import deepcopy
import importlib
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.vim_readout_core')


def source():
    # A diagonal uncentered second moment with a nonzero empirical mean.
    # Centering would put e0 into NS, whereas the actual cosine-head origin is 0.
    amplitude=np.linspace(.01,.3,159);x=np.zeros((318,160),np.float32);x[:,0]=.8
    for j,a in enumerate(amplitude,1):x[2*(j-1),j]=a;x[2*(j-1)+1,j]=-a
    logits=np.full((len(x),6),-1.,np.float32);logits[:,0]=2.
    return x,logits,amplitude.astype(np.float32)


def test_uncentered_second_moment_low80_basis_alpha_and_exact_flat_state():
    m=api();x,logits,a=source();before=x.copy();state,diag=m.fit(x,logits)
    assert set(state)=={'schema','origin','residual_basis','alpha'}
    assert state['schema'].shape==() and state['schema'].item()==m.SCHEMA
    assert state['origin'].dtype==state['residual_basis'].dtype==state['alpha'].dtype==np.float64
    assert state['residual_basis'].shape==(160,80) and state['alpha'].shape==()
    np.testing.assert_array_equal(state['origin'],np.zeros(160))
    projector=state['residual_basis']@state['residual_basis'].T;expected=np.zeros((160,160));expected[np.arange(1,81),np.arange(1,81)]=1.
    np.testing.assert_allclose(projector,expected,rtol=0,atol=1e-13)
    residual=np.zeros(318);residual[:160]=np.repeat(a[:80].astype(np.float64),2)
    np.testing.assert_allclose(float(state['alpha']),2/residual.mean(),rtol=1e-14,atol=0)
    np.testing.assert_array_equal(x,before)
    assert diag['assume_centered'] is True and not diag['empirical_mean_subtracted'] and diag['source_count']==318
    assert diag['principal_dimension']==diag['residual_dimension']==80 and diag['deployed_numeric_values']==12961 and diag['deployed_numeric_bytes']==103688
    assert m.validate_state(state) is state


def test_fewer_than160_source_rows_allowed_and_score_energy_direction_row_local_no_mutation():
    m=api();rng=np.random.default_rng(122);x=rng.normal(size=(90,160));x=(x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)
    logits=np.tile(np.asarray([2.,1.,0.,-.5,-1.,-2.],np.float32),(90,1));state,diag=m.fit(x,logits);snapshot=deepcopy(state)
    assert diag['source_count']==90
    q=np.vstack((x[:4],np.zeros((1,160),np.float32)));raw=np.vstack((logits[:4],np.zeros((1,6),np.float32)))
    scores=m.raw_scores(state,q,raw);residual=np.linalg.norm(q.astype(np.float64)@state['residual_basis'],axis=1)
    l=raw.astype(np.float64);peak=l.max(1);energy=peak+np.log(np.exp(l-peak[:,None]).sum(1))
    np.testing.assert_allclose(scores[:,0],-residual,rtol=0,atol=1e-14)
    np.testing.assert_allclose(scores[:,1],energy-float(state['alpha'])*residual,rtol=0,atol=1e-13)
    assert scores[-1,0]==0 and scores[-1,1]==pytest.approx(np.log(6.))
    shifted=m.raw_scores(state,q,raw+3.);np.testing.assert_array_equal(shifted[:,0],scores[:,0]);np.testing.assert_allclose(shifted[:,1]-scores[:,1],3.,rtol=0,atol=2e-13)
    order=np.asarray([4,0,3,1,2]);np.testing.assert_allclose(m.raw_scores(state,q[order],raw[order]),scores[order],rtol=0,atol=2e-13)
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,q[i:i+1],raw[i:i+1]) for i in range(len(q))]),scores,rtol=0,atol=2e-13)
    for key in state:np.testing.assert_array_equal(state[key],snapshot[key])


def test_basis_sign_and_orthogonal_rotation_leave_both_scores_unchanged():
    m=api();x,logits,_=source();state,_=m.fit(x,logits);original=m.raw_scores(state,x,logits)
    rng=np.random.default_rng(3);rotation=np.linalg.qr(rng.normal(size=(80,80)))[0]
    for transform in (np.diag(np.tile([-1.,1.],40)),rotation):
        changed=dict(state,residual_basis=state['residual_basis']@transform)
        m.validate_state(changed);np.testing.assert_allclose(m.raw_scores(changed,x,logits),original,rtol=0,atol=2e-13)


def test_exact_native_coordinate_oracle_normal_small_zero_and_gradients():
    from test_pamf_phase3_star_signal_open_model import return_aux_source
    m=api();native,_=return_aux_source();reader=native.id_backbone.encoder;head=reader.core.id_backbone.cls_head
    torch.manual_seed(14);raw=torch.randn(5,160);raw[0]*=1e-8;raw[1].zero_();raw.requires_grad_()
    x=m.classifier_coordinates(reader,raw);expected=F.normalize(raw,dim=1,eps=1e-4)
    assert torch.equal(x,expected) and x.dtype==torch.float32 and x[0].norm()<1 and x[1].abs().sum()==0
    oracle=head.scale*F.linear(x,F.normalize(head.weight,dim=1,eps=1e-4))
    torch.testing.assert_close(reader.classify_features(raw),oracle,rtol=0,atol=0)
    oracle.sum().backward();assert raw.grad is not None and bool(torch.isfinite(raw.grad).all())
    assert not isinstance(head.scale,torch.Tensor) and 'scale' not in dict(head.named_parameters()) and 'scale' not in dict(head.named_buffers())


def test_nonfinite_nonpositive_alpha_degenerate_source_and_input_boundaries(subtests):
    m=api();x,logits,_=source()
    for defect in ('short','zero','constant','negative_alpha','nan','dtype','norm','extra_label'):
        with subtests.test(defect=defect),pytest.raises((ValueError,TypeError)):
            if defect=='short':m.fit(x[:80],logits[:80])
            elif defect=='zero':m.fit(np.zeros_like(x),logits)
            elif defect=='constant':m.fit(np.tile(x[0],(len(x),1)),logits)
            elif defect=='negative_alpha':m.fit(x,np.full_like(logits,-2))
            elif defect=='nan':bad=x.copy();bad[0,0]=np.nan;m.fit(bad,logits)
            elif defect=='dtype':m.fit(x.astype(np.float64),logits)
            elif defect=='norm':m.fit(x*2,logits)
            else:m.fit(x,logits,y=np.zeros(len(x),np.int64))


def test_invalid_state_and_non_native_head_contract_rejected(subtests):
    from test_pamf_phase3_star_signal_open_model import return_aux_source
    m=api();x,logits,_=source();state,_=m.fit(x,logits)
    for defect in ('extra','origin','basis','basis_nan','alpha','alpha_nan'):
        with subtests.test(defect=defect),pytest.raises(ValueError):
            bad={k:v.copy() for k,v in state.items()}
            if defect=='extra':bad['y']=np.zeros(6)
            elif defect=='origin':bad['origin'][0]=.1
            elif defect=='basis':bad['residual_basis'][:,0]*=2
            elif defect=='basis_nan':bad['residual_basis'][0,0]=np.nan
            elif defect=='alpha':bad['alpha']=np.asarray(0.)
            else:bad['alpha']=np.asarray(np.nan)
            m.validate_state(bad)
    for defect in ('scale','bias','weight','dtype','raw_nan'):
        with subtests.test(head_defect=defect),pytest.raises(ValueError):
            native,_=return_aux_source();reader=native.id_backbone.encoder;h=reader.core.id_backbone.cls_head;raw=torch.zeros(2,160)
            if defect=='scale':h.scale=20.
            elif defect=='bias':h.bias=torch.nn.Parameter(torch.zeros(6))
            elif defect=='weight':h.weight=torch.nn.Parameter(torch.zeros(5,160))
            elif defect=='dtype':raw=raw.double()
            else:raw[0,0]=float('nan')
            m.classifier_coordinates(reader,raw)
