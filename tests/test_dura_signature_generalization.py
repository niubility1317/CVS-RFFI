import copy
import numpy as np
import pytest
from experiments.pamf_dura_efficiency.signature_generalization import errors,summarize,decision,SCENES,gain


def test_relative_error_decomposes_in_frozen_head_basis_and_is_per_record():
    r=np.random.default_rng(17);basis=np.linalg.qr(r.normal(size=(160,5)))[0];target=r.normal(size=(3,160));pred=r.normal(size=(2,3,160));iq=r.normal(size=(3,2,256))
    values=errors(pred,target,target,iq,iq,basis)
    np.testing.assert_allclose(values['SIG_relative_MSE'],values['SIG_head_visible_MSE']+values['SIG_head_orthogonal_MSE'],rtol=0,atol=2e-14)
    one=errors(pred[:,:1],target[:1],target[:1],iq[:1],iq[:1],basis)
    np.testing.assert_allclose(one['SIG_relative_MSE'],values['SIG_relative_MSE'][:,:1])
    assert np.all(values['restored_SIG_relative_MSE']==0) and np.all(values['iq_nmse']==0)
    np.testing.assert_allclose(summarize(values,np.ones(3,bool))['SIG_relative_MSE'],values['SIG_relative_MSE'].mean(),rtol=0,atol=1e-14)
    with pytest.raises(ValueError):errors(pred,target[:1],target,iq,iq,basis)


def fixture_models():
    models={}
    for key in ['context','linear','mlp']:
        train={v:{'scene:'+s:{'SIG_relative_MSE':1 if key=='context' else .8} for s in SCENES} for v in ['1','20']}
        validation={s:{'all':{'SIG_relative_MSE':1 if key=='context' else 1.1,'SIG_head_visible_MSE':.1}} for s in SCENES}
        models[key]={'train':train,'V':validation}
    return models


def test_generalization_hypothesis_requires_both_models_and_both_cached_views():
    models=fixture_models();result=decision(models)
    assert result['generalization_failure']=='PASS' and result['head_direction_tradeoff']=='FAIL'
    for s in SCENES[:3]:models['mlp']['train']['20']['scene:'+s]['SIG_relative_MSE']=1
    assert decision(models)['generalization_failure']=='FAIL'
    assert gain(0,0) is None


def test_head_direction_hypothesis_is_separate_and_zero_reference_is_finite():
    models=fixture_models()
    for key in ['linear','mlp']:
        for s in SCENES:models[key]['V'][s]['all']['SIG_relative_MSE']=.8
    result=decision(models);assert result['generalization_failure']=='FAIL' and result['head_direction_tradeoff']=='PASS'
    b=np.eye(160)[:,:5];zero=np.zeros((2,160));iq=np.zeros((2,2,256));v=errors(np.ones((2,2,160)),zero,zero,iq,iq,b)
    assert all(np.isfinite(x).all() for x in v.values())
