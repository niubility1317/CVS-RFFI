"""Independent synthetic checks of the fixed null-tone observation primitive."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import torch

from .null_observation import null_evidence


def _fixture(dtype=torch.float64, null_amplitudes=(0., 0., 0., 0.), slope=0.):
    # Independently synthesize the public occupied subspace with arbitrary
    # complex transfer at every tone, plus a DC component. No dataset is read.
    bins=torch.tensor([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6],dtype=torch.float64)
    signs=torch.tensor([1,-1,1,-1,-1,1,-1,-1,1,1,1,1],dtype=torch.float64)
    transfer=torch.complex(.5+.03*torch.arange(12,dtype=torch.float64),
                           -.2+.025*torch.arange(12,dtype=torch.float64))
    time=torch.arange(20,dtype=torch.float64)
    period=(signs*(1+1j)*transfer*torch.exp(2j*torch.pi*time[:,None]*bins/20)).sum(1)/math.sqrt(24)
    period=period+(.4-.2j)
    cycles=torch.stack([period+(amplitude*(.3+.1j))*torch.exp(2j*torch.pi*7*time/20)
                        for amplitude in null_amplitudes])
    cycles=cycles*torch.exp(1j*slope*torch.arange(80,dtype=torch.float64).reshape(4,20))
    x=torch.zeros((1,2,256),dtype=dtype)
    x[0,0,80:160]=cycles.real.flatten().to(dtype)
    x[0,1,80:160]=cycles.imag.flatten().to(dtype)
    return x,cycles


def _raises(callback):
    try:callback()
    except ValueError:return
    raise AssertionError("expected malformed IQ rejection")


def run_checks():
    py_state=random.getstate();np_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    pure,_=_fixture()
    reference=null_evidence(pure)
    assert reference["valid"].tolist()==[True]
    assert reference["cycles"].shape==(1,3,14) and reference["target"].shape==(1,14)
    assert float(reference["cycles"].abs().max()) < 2e-14
    assert float(reference["target"].abs().max()) < 2e-14
    x,cycles=_fixture(null_amplitudes=(1.,2.,3.,4.),slope=.004)
    x.requires_grad_();before=x.detach().clone()
    result=null_evidence(x)
    visible=cycles[[0,1,3]]
    rms=visible.abs().square().mean().sqrt()
    local=torch.arange(20,dtype=torch.float64)
    corrected=visible*torch.exp(-1j*.004*(torch.tensor([0,20,60])[:,None]+local))
    expected_complex=torch.fft.fft(corrected/rms,dim=1)[:,7:14]/math.sqrt(20)
    expected=torch.cat((expected_complex.real,expected_complex.imag),dim=1)[None]
    assert torch.allclose(result["cycles"],expected,atol=2e-14,rtol=2e-14)
    assert torch.equal(result["mean"],result["cycles"].mean(1))
    assert torch.equal(result["copy_cycle1"],result["cycles"][:,1])
    assert torch.count_nonzero(result["zero_reference"])==0
    variance=(expected-expected.mean(1,keepdim=True)).square().mean(2).sum(1)/2
    assert torch.allclose(result["variance"],variance,atol=2e-14,rtol=2e-14)
    assert torch.equal(result["budget"],(4/3)*result["variance"])
    assert torch.equal(result["signal"],result["mean"].square().mean(1))
    assert float(result["signal"].detach()) > 0 and float(result["budget"].detach()) > 0
    expected_target_complex=torch.fft.fft(cycles[2]/rms*torch.exp(-1j*.004*(40+local)))[7:14]/math.sqrt(20)
    expected_target=torch.cat((expected_target_complex.real,expected_target_complex.imag))[None]
    assert torch.allclose(result["target"],expected_target,atol=2e-14,rtol=2e-14)
    assert not result["target"].requires_grad
    altered=x.detach().clone();altered[:,:,120:140]=17.;altered[:,:,:80]=-31.;altered[:,:,160:]=29.
    replacement=null_evidence(altered)
    for key in ("cycles","mean","copy_cycle1","zero_reference","signal","variance","budget","valid","masked_input"):
        assert torch.equal(result[key].detach(),replacement[key])
    for key in result["calibration"]:
        assert torch.equal(result["calibration"][key].detach(),replacement["calibration"][key])
    assert not torch.equal(result["target"],replacement["target"])
    visible_mask=torch.zeros(256,dtype=torch.bool);visible_mask[80:120]=True;visible_mask[140:160]=True
    assert torch.count_nonzero(result["masked_input"][:,:,~visible_mask])==0
    for probe in (result["cycles"],result["mean"],result["copy_cycle1"],result["signal"],result["budget"],
                  result["masked_input"],*result["calibration"].values()):
        gradient=torch.autograd.grad(probe.abs().square().sum(),x,retain_graph=True)[0]
        assert torch.count_nonzero(gradient[:,:,~visible_mask])==0
        assert bool(torch.isfinite(gradient).all())
    max_scale_error=0.
    for scale in (1e-150,.03,17.,1e150):
        scaled=null_evidence(x.detach()*scale)
        for key in ("cycles","mean","copy_cycle1","target","signal","variance","budget","masked_input"):
            difference=float((scaled[key]-result[key].detach()).abs().max())
            max_scale_error=max(max_scale_error,difference)
            assert torch.allclose(scaled[key],result[key].detach(),atol=3e-14,rtol=3e-14)
        assert torch.equal(scaled["valid"],result["valid"])
    zero=null_evidence(torch.zeros_like(x))
    assert zero["valid"].tolist()==[False]
    for key in ("cycles","mean","copy_cycle1","zero_reference","target","signal","variance","budget","masked_input"):
        assert torch.count_nonzero(zero[key])==0
    mixed=null_evidence(torch.cat((pure,x.detach(),torch.zeros_like(x))))
    assert mixed["valid"].tolist()==[True,True,False]
    assert torch.equal(mixed["cycles"][1],result["cycles"][0].detach())
    # Vanishing phase cross-correlation does not invalidate nonzero visible
    # evidence. Its defined zero slope must have a finite derivative.
    no_cross=pure.clone();no_cross[:,:,80:100]=0.;no_cross.requires_grad_()
    no_cross_result=null_evidence(no_cross)
    assert no_cross_result["valid"].tolist()==[True]
    assert no_cross_result["calibration"]["phase_slope"].item()==0.
    no_cross_gradient=torch.autograd.grad(no_cross_result["cycles"].square().sum(),no_cross)[0]
    assert bool(torch.isfinite(no_cross_gradient).all())
    assert torch.count_nonzero(no_cross_gradient[:,:,~visible_mask])==0
    float32,_=_fixture(torch.float32,null_amplitudes=(1.,2.,3.,4.),slope=.004)
    finite32=null_evidence(float32)
    assert torch.allclose(finite32["cycles"],expected,atol=1e-7,rtol=1e-7)
    _raises(lambda:null_evidence(torch.zeros(1,2,255)))
    _raises(lambda:null_evidence(torch.zeros(1,256,2)))
    _raises(lambda:null_evidence(torch.zeros(0,2,256)))
    _raises(lambda:null_evidence(torch.zeros(1,2,256,dtype=torch.int64)))
    nan=x.detach().clone();nan[0,0,125]=float("nan")
    _raises(lambda:null_evidence(nan))
    assert torch.equal(x.detach(),before) and x.grad is None
    assert random.getstate()==py_state and torch.equal(torch.get_rng_state(),torch_state)
    current_np=np.random.get_state()
    assert current_np[0]==np_state[0] and np.array_equal(current_np[1],np_state[1]) and current_np[2:]==np_state[2:]
    json.dumps(result["summary"],allow_nan=False)
    return {"status":"VERIFIED","torch":torch.__version__,"device":"cpu","dtypes":["float64","float32"],
            "checks":{"occupied_transfer_plus_DC_null_zero":True,"fixed_null_harmonic_observable":True,
                      "baseline_variance_budget_closed_forms":True,"hidden_changes_only_target":True,
                      "hidden_outside_derivatives_zero":True,"masked_hidden_outside_zero":True,
                      "positive_global_scale_invariant":True,"zero_visible_invalid":True,
                      "phase_cross_zero_visible_still_valid_finite_derivative":True,
                      "invalid_inputs_rejected":True,"inputs_and_RNG_unchanged":True},
            "maximum_scale_error":max_scale_error,
            "boundary":"periodic steady-state linear reference subspace only; upstream equalization unknown"}


if __name__=="__main__":print(json.dumps(run_checks(),ensure_ascii=False,allow_nan=False,indent=2))
