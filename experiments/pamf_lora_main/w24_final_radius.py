"""Use the native noise estimate to widen only the final W24 MAP projection."""

import torch

from experiments.pamf_lora_main.w24_readout_strength import packet_noise


def final_radius(noise, rms, base):
    """Bound the packet-relative full edit between the original cap and 0.95."""
    return (base.square() + 4 * packet_noise(noise, rms)).sqrt().clamp_max(.95)


def projected_stages(model, old, y, quality, rms, *, condition=(), expand_radius=True, adapter=None, final_solver=None):
    # Keep the 7615e8e W24 stage order. The archive exposes the learned-prior
    # projection but has no override for the MAP output projection.
    spec = old.fft(y)
    noise = (spec[..., model.noise_band].abs().square().median(-1, keepdim=True).values
             / .69314718056).clamp(1e-5, 4.)
    log_noise = noise.log().clamp(-11., 2.) / 5.
    even, odd = (y[:, offset::2].mean(1, keepdim=True) for offset in (0, 1))
    average = .5 * (even + odd)
    opposite = torch.cat((odd, even), 1).repeat(1, y.shape[1] // 2, 1)
    n = y.shape[-1]
    flip = (-torch.arange(n, device=y.device)) % n
    fr, fi = old.read_core(model.frequency, (*old.iq_field(spec, spec[..., flip].conj()), log_noise))
    estimate = old.zero_carrier(old.ifft(torch.complex(fr.tanh(), fi.tanh()) * noise.sqrt()))
    pack = (*old.iq_field(y, opposite, estimate, y - average), quality, *condition)
    cr, ci = old.read_core(model.coarse, pack)
    base = y.real.new_tensor(old.MAX_EDIT)
    coarse = y + old.limit(old.zero_carrier(estimate + torch.complex(cr.tanh(), ci.tanh())), y, rms, base)
    nominal, _, mismatch = old.ReceivedUpdate.estimate(y, noise.mean(1, keepdim=True))
    inputs = (*old.iq_field(y, coarse, opposite, nominal - 1.), log_noise, *condition)
    gr, gi, ir, ii, er, ei, excess = (
        x.tanh().unflatten(-1, (-1, 4)).mean(-1).repeat_interleave(4, dim=-1)
        for x in old.read_core(model.risk, inputs)
    )
    action = old.ObservedAction(nominal + .24 * torch.complex(gr, gi),
                                .12 * torch.complex(ir, ii), .15 * torch.complex(er, ei))
    variance = .001 + noise * torch.exp(1.2 * excess)
    current = coarse
    stages = []
    for i, network in enumerate(model.priors):
        remaining = y - action(current)
        innovation = action.adjoint(remaining)
        normalized = innovation / (.001 + innovation.abs().square().mean(-1, keepdim=True)).sqrt()
        fields = (*old.iq_field(y, coarse, current, normalized, opposite, action.main - 1.),
                  variance.log().clamp(-11., 2.) / 5., *condition)
        real, imag = old.read_core(network, fields)
        if i == 1 and adapter is not None:
            learned = torch.complex(real.tanh(), imag.tanh())
            extra = adapter(y, current, normalized, learned, noise)
            extra = old.limit(extra, y, rms, y.real.new_tensor(.15))
            prior = coarse + old.limit(old.zero_carrier(learned + extra), y, rms, base)
        else:
            prior = coarse + model.update(real, imag, y, rms)
        if i == 1 and final_solver is not None:
            current = final_solver(action, y, current, prior, variance, remaining, noise)
        else:
            current = old.map_two(action, y, current, prior, variance, action.adjoint(remaining / variance))
        radius = final_radius(noise, rms, base) if i == 1 and expand_radius else base
        current = y + old.limit(old.zero_carrier(current - y), y, rms, radius)
        stages.append(current)
    return (coarse, *stages), action, variance, noise, mismatch
