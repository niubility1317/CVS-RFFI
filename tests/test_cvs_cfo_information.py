import json
from pathlib import Path
import numpy as np
import pytest
from experiments.cvs_cfo_information.audit import packet_features,fit_probe,predict_probe,factor_association,validate_config


def wave(hz,theta=.0):
    n=np.arange(256);z=np.exp(2j*np.pi*n/20)*np.exp(1j*(theta+2*np.pi*hz*n/25000000))
    return np.stack((z.real,z.imag))[None]


def test_packet_units_phase_gain_and_modulo_alias():
    f,t=packet_features(wave(130000));g,u=packet_features(3*wave(130000,1.2))
    assert np.allclose(t['hz'],130000,atol=1e-7)
    assert np.allclose(t['coherence'],1,atol=1e-12)
    assert np.allclose(f['three_window'],g['three_window'],atol=1e-12)
    g,u=packet_features(wave(1380000));assert np.allclose(f['cfo80'],g['cfo80'],atol=1e-12)
    assert np.allclose(u['hz'],130000,atol=1e-7)


def test_zero_input_is_finite_and_fallback():
    f,t=packet_features(np.zeros((2,2,256)))
    assert not t['valid'].any() and not t['coherence'].any() and not t['hz'].any()
    assert all(np.isfinite(v).all() for v in f.values())
    with pytest.raises(ValueError):packet_features(np.full((1,2,256),np.nan))


def test_probe_class_permutation_and_readonly_validation():
    y=np.repeat(np.arange(6),10);x=np.eye(6)[y]
    state=fit_probe(x,y);before=json.dumps(state,sort_keys=True)
    assert np.array_equal(predict_probe(state,x),y)
    predict_probe(state,x*100-20)
    assert json.dumps(state,sort_keys=True)==before and not state['V_fit']
    perm=np.array([4,2,5,1,0,3]);other=fit_probe(x,perm[y])
    assert np.array_equal(predict_probe(other,x),perm[y])
    with pytest.raises(ValueError):fit_probe(x[:10],y[:10])


def test_crossed_factor_association_separates_rx_from_tx():
    triples=np.array([(t,r,d) for t in range(6) for r in [1,3,4,6,8] for d in [1,2,3]])
    y,r,d=triples.T;x=np.column_stack((np.cos(r),np.sin(r)))
    a=factor_association(x,y,r,d)['association_fraction']
    assert a['RX']==pytest.approx(1) and a['TX']==pytest.approx(0,abs=1e-14)
    x=np.column_stack((np.cos(y),np.sin(y)));a=factor_association(x,y,r,d)['association_fraction']
    assert a['TX']==pytest.approx(1) and a['RX']==pytest.approx(0,abs=1e-14)


def test_remote_release_script_compiles_before_mutation():
    from experiments.cvs_cfo_information.publish import REMOTE,INSPECT
    compile(REMOTE.replace('CONFIG',repr(dict(project='/p',release='r',run='run'))),'<remote>','exec')
    compile(INSPECT.replace('CONFIG',repr(dict(project='/p',release='r',run='run'))),'<inspect>','exec')


@pytest.mark.parametrize('key',['checkpoint','target_truth','p1_capsule','augmentation'])
def test_refuses_target_checkpoint_and_augmentation(key):
    root=Path(__file__).resolve().parents[1];c=json.loads((root/'experiments/cvs_cfo_information/configs/source.json').read_text(encoding='utf-8'))
    validate_config(c);c[key]='forbidden'
    with pytest.raises(ValueError):validate_config(c)
