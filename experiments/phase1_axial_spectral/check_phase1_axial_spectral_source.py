"""Focused alignment and fixed-selection checks for the source-only experiment."""
import numpy as np
import torch
from phase1_axial_spectral_source import align, choose
def align_to_reference(wave, reference, width=64):
    pad=np.pad(wave,((0,0),(width,width)),mode='reflect');ref=reference[:,width:-width]
    best=np.full(len(wave),np.inf);chosen=np.zeros_like(wave)
    for shift in range(-width,width+1):
        full=pad[:,width+shift:width+shift+256];part=full[:,width:-width]
        gain=np.sum(np.conj(part)*ref,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20)
        error=np.mean(abs(gain[:,None]*part-ref)**2,axis=1);use=error<best
        best[use]=error[use];chosen[use]=gain[use,None]*full[use]
    return chosen,best

def main():
    rng=np.random.default_rng(33071)
    wave=rng.normal(size=(5,2,256)).astype(np.float32)
    ref=rng.normal(size=(5,2,256)).astype(np.float32)
    actual,err=align(torch.from_numpy(wave),torch.from_numpy(ref))
    w=wave[:,0].astype(np.float64)+1j*wave[:,1];r=ref[:,0].astype(np.float64)+1j*ref[:,1]
    expected,e=align_to_reference(w,r)
    expected_iq=np.stack((expected.real,expected.imag),1)
    np.testing.assert_allclose(actual.numpy(),expected_iq,rtol=1e-4,atol=1e-4)
    np.testing.assert_allclose(err.numpy(),e,rtol=1e-5,atol=1e-5)
    # A heldout gain must not rescue validation regression; and improvement must pass every validation cell.
    cand={2:{'1:4:a':{'baseline':1.,'candidate':[.99,1.01,1.01]},'1:6:a':{'baseline':1.,'candidate':[.99,.99,1.01]},'2:8:a':{'baseline':1.,'candidate':[10.,0.,0.]}}}
    result=choose(cand)
    assert result['epoch']==2 and result['alpha']==.25
    cand[2]['1:4:a']['candidate']=[1.01,1.01,1.01]
    assert choose(cand)['kind']=='raw'
    print('VERIFIED axial source alignment and validation-only guard')
if __name__=='__main__':main()
