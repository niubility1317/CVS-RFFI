import numpy as np
import pytest
from experiments.cvs_rff_physics.source_preamble import packet_statistics,validate_config


def iq(z):return np.stack([z.real,z.imag],axis=1)


def test_known_period_offset_and_closure_rotation():
    n=np.arange(256);base=np.exp(2j*np.pi*n/20)+.3*np.exp(-4j*np.pi*n/20)
    x=base[None]*np.exp(2j*np.pi*18_000*n/25_000_000)
    stats,scan,p=packet_statistics(iq(x))
    assert scan[0,19]>.999999
    assert stats['w0_80_lag20_wrapped_offset_hz'][0]==pytest.approx(18_000,abs=1e-8)
    changed=packet_statistics(iq(x*np.exp(1j*(.4+.019*n))))[0]
    for name,v in stats.items():
        if 'closure' in name or 'amplitude' in name or 'peak_power' in name:
            np.testing.assert_allclose(changed[name],v,atol=1e-12)


def test_zero_and_per_packet_independence():
    z=np.zeros((1,256),dtype=complex)
    stats,scan,p=packet_statistics(iq(z))
    assert np.isfinite(scan).all() and all(np.isfinite(v).all() for v in stats.values())
    n=np.arange(256);x=np.stack([np.exp(.03j*n),2*np.exp(.07j*n)])
    both=packet_statistics(iq(x))
    for j in range(2):
        one=packet_statistics(iq(x[j:j+1]))
        for name,v in one[0].items():np.testing.assert_allclose(v,both[0][name][j:j+1],atol=1e-12)


def test_source_config_rejects_target_and_role_changes():
    c=dict(method='rff_source_preamble',split_seed=392005,roles=['L_s','V'],source_receivers=[1,3,4,6,8],
           source_days=[1,2,3],equalized=1,out_len=256,rms_normalize=True,augmentation=False,model=False)
    validate_config(c)
    for key,value in [('target_truth','truth.json'),('checkpoint','old.pt'),('roles',['L_s','V','U_s']),
                      ('source_receivers',[0,1]),('augmentation',True),('equalized',0)]:
        with pytest.raises(ValueError):validate_config(dict(c,**{key:value}))


def test_input_schema_rejects_nonfinite_or_wrong_length():
    with pytest.raises(ValueError):packet_statistics(np.zeros((2,2,255)))
    x=np.zeros((2,2,256));x[0,0,2]=np.nan
    with pytest.raises(ValueError):packet_statistics(x)
