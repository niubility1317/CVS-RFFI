"""Measure LoRa receiver estimates against analytic references in isolation."""

import json

import torch

from experiments.pamf_lora_main.local_phase import read_observation, rotation


def trial(chirps, snr_db, colored):
    rng = torch.Generator().manual_seed(2026112431)
    shape = (256, chirps, 256)
    sample = torch.arange(shape[-1])/shape[-1]
    template = rotation(.07*(2*torch.pi*5*sample).sin()+.09*(2*torch.pi*17*sample).cos())
    # Only the first chirp differs from the known leave-out reference.
    rate = .25+2.*torch.rand((shape[0],1),generator=rng)
    initial = 2*torch.pi*torch.rand((shape[0],1),generator=rng)
    receive_phase = .24*(2*torch.pi*rate*sample+initial).sin()
    receive_amplitude = 1.+.17*(2*torch.pi*.6*sample+initial).cos()
    relative_action = receive_amplitude*rotation(receive_phase)
    signal = template.expand(shape).clone()
    signal[:,0] *= relative_action
    noise = torch.randn(shape,generator=rng,dtype=torch.complex64)
    if colored:
        donor = torch.randn((shape[0],1,shape[-1]),generator=rng,dtype=torch.complex64)
        noise = (noise + .9*donor) / (1.+.9**2)**.5
        noise = (noise + .85*noise.roll(1,dims=-1)) / (1.+.85**2)**.5
    noise = noise*10**(-snr_db/20.)
    observed = signal+noise
    power = observed.abs().square().mean(-1,keepdim=True)
    q = observed.mean(-1,keepdim=True).abs().square()/power.clamp_min(1e-6)
    outer = torch.fft.fftfreq(shape[-1]).abs()>.25
    _, action, estimate, disagreement, confidence, *_ = read_observation(observed,q,outer)
    actual = noise.abs().square().mean(-1,keepdim=True)
    ratio = (estimate/actual).flatten()
    response_error = (action[:,0]-relative_action).abs().square().mean(-1).sqrt()
    identity_error = (1.-relative_action).abs().square().mean(-1).sqrt()
    assert torch.isfinite(ratio).all() and torch.isfinite(response_error).all()
    return dict(chirps=chirps,snr_db=snr_db,noise='colored_correlated' if colored else 'white_independent',
                outer_to_actual=torch.quantile(ratio,torch.tensor([.1,.5,.9])).tolist(),
                action_rmse_median=float(response_error.median()),identity_rmse_median=float(identity_error.median()),
                confidence_median=float(confidence[:,0].median()),
                disagreement_median=float(disagreement[:,0].median()))


if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps([trial(c,s,colored) for c in (3,4,6) for s in (0,6,18) for colored in (False,True)],indent=2))
