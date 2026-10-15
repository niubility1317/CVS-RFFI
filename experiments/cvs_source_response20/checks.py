"""Synthetic operator checks only: no source arrays, checkpoints or IQ files."""
import json
import numpy as np
from . import response as r


def main():
    rng=np.random.default_rng(24);p=rng.normal(size=(18,5,8));p-=p.mean(1,keepdims=True)
    b=r.bank(p);a=b['empirical'];c=b['rotated']
    assert np.allclose(np.linalg.norm(a,axis=1),np.linalg.norm(c,axis=1),rtol=1e-12)
    assert max(np.abs(a).sum(1).max(),np.abs(c).sum(1).max())<=.06+1e-12
    for v in (a,c):
        assert np.allclose(v,np.conj(v[:,::-1]),atol=1e-14)
        offsets=np.arange(-16,17);omega=np.linspace(-np.pi,np.pi,8193)
        spectrum=1+v@np.exp(-1j*offsets[:,None]*omega[None,:])
        assert np.max(np.abs(spectrum.imag))<1e-12
        assert spectrum.real.min()>=.94-1e-12 and spectrum.real.max()<=1.06+1e-12
    keys=[(1,rx,1,0,0) for rx in (1,3,4,6,8)]
    np.random.seed(71);state=np.random.get_state();h=r.kernels(b,'empirical',keys)
    after=np.random.get_state();assert all(np.array_equal(x,y) for x,y in zip(state,after))
    assert r.session(*keys[0])==r.session(*keys[0])
    assert r.session(*keys[0])!=r.session(2,1,1,0,0)
    assert r.session(*keys[0])!=r.session(1,1,1,0,1)
    x=rng.normal(size=(5,2,256));y=r.apply_numpy(x,h)
    assert np.allclose((x*x).sum((1,2)),(y*y).sum((1,2)),rtol=1e-12)
    assert np.max(np.abs(x-y))>1e-5
    reflect=r.apply_numpy(x,h,restore_rms=False);edge=r.apply_numpy(x,h,'edge',False)
    assert np.array_equal(reflect[:,:,16:240],edge[:,:,16:240])
    identity=np.zeros((5,33),complex);identity[:,16]=1
    assert np.allclose(r.apply_numpy(x,identity),x)
    assert np.count_nonzero(r.apply_numpy(np.zeros_like(x),h))==0
    negatives=[lambda:r.bank(np.zeros((18,5,8))),lambda:r.bank(p[:17]),
        lambda:r.session(0,1,1,0),lambda:r.session(1,2,1,0),lambda:r.session(1,1,1,0,tx=0),
        lambda:r.kernels(b,'chosen',keys),lambda:r.kernels(b,'empirical',[(1,1,1,0,0,0)]),
        lambda:r.apply_numpy(x[:,:,:255],h),lambda:r.apply_numpy(x,h,'wrap')]
    for fn in negatives:
        try:fn()
        except (ValueError,TypeError):pass
        else:raise AssertionError('Invalid operator configuration accepted')
    import torch
    torch.set_num_threads(2)
    t=torch.tensor(x.tolist(),dtype=torch.float32)
    actual=r.apply_torch(t,h).tolist()
    assert np.allclose(np.array(actual),y,atol=2e-6,rtol=2e-6)
    interior_a=r.apply_torch(t,h,'reflect',False)[:,:,16:240]
    interior_b=r.apply_torch(t,h,'edge',False)[:,:,16:240]
    assert torch.equal(interior_a,interior_b)
    from . import probe_design as d
    d.validate(d.config())
    for field,bad in [('source_role','U_s'),('forward_packets',1),('automatic_training',True)]:
        candidate=d.config();candidate[field]=bad
        try:d.validate(candidate)
        except ValueError:pass
        else:raise AssertionError('Protocol change accepted')
    fixture={k:dict(overall=dict(accuracy=.95),strata=[dict(dimension='RX',value=1,accuracy=.95)]) for k in d.CONDITIONS}
    assert d.gate(fixture)['status']=='PASS'
    fixture['EMP_E20_REFLECT']['overall']['accuracy']=.93
    assert d.gate(fixture)['status']=='FAIL'
    fixture['EMP_E20_REFLECT']['overall']['accuracy']=.95
    fixture['EMP_E20_REFLECT']['strata'][0]['accuracy']=.92
    assert d.gate(fixture)['status']=='FAIL'
    print(json.dumps(dict(status='PASS',bound_and_pair_norm=True,zero_phase_interior=True,
        session_and_RNG=True,finite_IQ_RMS_and_boundary=True,torch_numpy_parity=True,
        negative_cases=len(negatives),protocol_negative_cases=3,gate_cases=3,real_IQ_reads=0,model_forwards=0,optimizer_updates=0)))


if __name__=='__main__':main()
