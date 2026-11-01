import pytest
import torch
from experiments.pamf_phase3_restoration.residual_diffusion import scratch_residual_diffusion
from experiments.pamf_phase3_restoration.diffusion_source_path import trajectory
from experiments.pamf_phase3_restoration.diffusion_source_fixes import early_denoised4,mix_sigmas
from experiments.pamf_phase3_restoration import diffusion_sigma_mix_trial as mix

torch.set_num_threads(2)


def test_early_readout_is_exact_source_step4_and_seven_core_calls():
    g=scratch_residual_diffusion(5).eval().requires_grad_(False);x=torch.randn(3,2,256)
    with torch.no_grad():g.core.time_out.weight.normal_(0,.001);g.core.freq_out.weight.normal_(0,.001)
    before={k:v.clone() for k,v in g.state_dict().items()};calls=[]
    with torch.inference_mode():expected=trajectory(g,x)[3]
    hook=g.core.register_forward_hook(lambda *a:calls.append(1))
    with torch.inference_mode():actual=early_denoised4(g,x)
    hook.remove();assert len(calls)==7 and torch.equal(actual,expected)
    with torch.inference_mode():torch.testing.assert_close(actual,torch.cat([early_denoised4(g,z[None]) for z in x]),atol=3e-5,rtol=3e-5)
    assert all(torch.equal(v,g.state_dict()[k]) for k,v in before.items())


def test_sigma_mixture_support_coverage_and_stateless_reproducibility():
    g=scratch_residual_diffusion(5);original=torch.full((63000,),.314159)
    before=torch.get_rng_state().clone();sigma,code=mix_sigmas(original,g.sigma_steps[:8],37)
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(original,torch.full_like(original,.314159))
    assert torch.equal(sigma[code==0],original[code==0])
    for k in range(1,9):assert torch.all(sigma[code==k]==g.sigma_steps[k-1]) and (code==k).sum()>3000
    assert 30000<(code==0).sum()<33000
    again,again_code=mix_sigmas(original,g.sigma_steps[:8],37)
    assert torch.equal(sigma,again) and torch.equal(code,again_code)


def test_mixture_preserves_parent_gaussian_noise_and_global_rng_sequence():
    steps=scratch_residual_diffusion(5).sigma_steps[:8]
    torch.manual_seed(2026110203);sigma=(torch.randn(256)*1.2-1.2).exp();noise=torch.randn(256,2,256);after=torch.get_rng_state()
    torch.manual_seed(2026110203);original=(torch.randn(256)*1.2-1.2).exp();other=torch.randn(256,2,256)
    mixed,code=mix_sigmas(original,steps,0)
    assert torch.equal(noise,other) and torch.equal(after,torch.get_rng_state())
    assert torch.equal(sigma,original) and torch.equal(mixed[code==0],sigma[code==0])


def test_sigma_mix_same_scratch_architecture_and_fixed_legal_budget():
    from experiments.pamf_phase3_restoration import residual_diffusion_trial as parent
    c=mix.configuration(mix.ARM);a=mix.factory(mix.ARM);b=parent.factory(parent.ARM)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert (c['epochs'],c['updates'],c['training_stages'],c['sampling_denoiser_calls'])==(30,7410,1,15)
    assert c['initialization']=='scratch' and c['ancestor_checkpoints']==[] and c['diffusion_sigma_mixture_probability']==.5
    assert all(c[k]==0 for k in ['TX_labels_read','V_reads','OE_reads','target_training_reads'])
    with pytest.raises(ValueError):mix.configuration(parent.ARM)


def test_frozen_early_wrapper_preserves_parent_and_rejects_wrong_iq_boundary():
    from experiments.pamf_phase3_restoration.diffusion_early_eval import FrozenEarly4,STREAMS
    g=scratch_residual_diffusion(5).eval().requires_grad_(False);wrapper=FrozenEarly4(g).eval()
    assert sum(p.numel() for p in wrapper.parameters())==1614608
    assert STREAMS==('raw','DURA_v1','full8','candidate')
    with pytest.raises(ValueError):wrapper(torch.ones(2,2,128))
    with pytest.raises(ValueError):wrapper(torch.ones(2,2,256,dtype=torch.float64))
