"""Isolate signal shrinkage and noise shaping in the LoRa analytic readout."""

from functools import partial

import numpy as np
import torch

from .channel_blind import corruption_score
from .dose_direction_diagnostic import RUN as BASE_RUN, MODES as BASE_MODES, evaluate, main, smoke
from .dura_repair import complex_iq
from .joint_unroll_source import coordinates
from .spectral_mixture_diagnostic import lock
from .variance_transport import physical_norm, posterior_gain


RUN = '20261124-diagnostic-lora-linear-bias-source-m2-r01'
STRENGTHS = ('one','two')
VARIANTS = ('analytic','no_common_signal','no_private_signal','filtered_noise','exact')
MODES = ('direct','reference', *(f'{kind}_{strength}' for strength in STRENGTHS for kind in VARIANTS))
COMPARISONS = (
    *((f'{variant}_{scale}',f'analytic_{scale}') for scale in STRENGTHS for variant in VARIANTS[1:]),
    *((f'exact_{scale}',f'filtered_noise_{scale}') for scale in STRENGTHS),
    *((f'filtered_noise_{scale}',f'{variant}_{scale}') for scale in STRENGTHS
      for variant in ('no_common_signal','no_private_signal')),
    *((name,'reference') for name in MODES[2:]),
)
FIELDS = ('active','analytic_rms_fraction','filtered_noise_rms_fraction','common_signal_rms_fraction',
          'private_signal_rms_fraction','physical_split_rms_error','max_rms_lock_error',
          'analytic_exact_cosine','filtered_exact_cosine')
REPLAY = {name:name for name in ('direct','reference','analytic_one','analytic_two','exact_one','exact_two')}


def decomposed_directions(model,y,added_noise):
    spectrum = model.spectrum(y)
    observed_coeffs = model.coefficients(spectrum)
    # The prior is affine. Taking a difference removes its centering without
    # changing the operator estimated from the original received packet.
    noise_coeffs = observed_coeffs - model.coefficients(model.spectrum(y-added_noise))
    signal_coeffs = observed_coeffs - noise_coeffs
    component = model.component
    variance = model.noise(spectrum)[:,component,None]
    shrink = posterior_gain(model.joint_variance[component],variance).sqrt()-1.
    clean_part = signal_coeffs*shrink
    common = torch.cat((clean_part[:,:1],torch.zeros_like(clean_part[:,1:])),1)
    private = clean_part-common
    return model.synthesize(noise_coeffs*shrink), model.synthesize(common), model.synthesize(private)


def recover(iq,noise_iq,tracker,model,threshold):
    y, phase, _, rms = coordinates(tracker,iq)
    active = corruption_score(y)>threshold
    count = int(active.sum())
    output = iq.expand(len(MODES)-2,*iq.shape).clone()
    measurements = iq.new_zeros((len(iq),len(FIELDS)))
    if not count:
        return output,count,measurements
    y, phase, rms = y[active],phase[active],rms[active]
    frame = phase*rms
    epsilon = complex_iq(noise_iq[active])/frame
    exact = -(epsilon-epsilon.mean(-1,keepdim=True))*frame
    analytic = model.changes(y,rms)[0][1]*frame
    noise,common,private = (part*frame for part in decomposed_directions(model,y,epsilon))
    base_size = physical_norm(analytic,1.)
    raw_size = physical_norm(y,rms).clamp_min(1e-8)
    total = noise+common+private
    split_error = physical_norm(lock(total,base_size)-analytic,1.).flatten()
    if (split_error>5e-6*(1.+raw_size.flatten())).any():
        raise ValueError('The affine observed-domain edit did not decompose')
    offered = (analytic, lock(noise+private,base_size), lock(noise+common,base_size), lock(noise,base_size),
               lock(exact,base_size))
    normalized = torch.stack(offered)
    norm_error = (physical_norm(normalized,1.)-base_size).abs().amax(0).flatten()
    exact_energy = physical_norm(exact,1.).flatten()
    good = exact_energy > 1e-8
    if (norm_error[good]>5e-6*(1.+raw_size.flatten()[good])).any() or not torch.isfinite(normalized).all():
        raise ValueError('The matched physical edit dose changed')
    increments = torch.cat((normalized,2.*normalized))
    output[:,active] = iq[active]+torch.stack((increments.real,increments.imag),dim=-2)

    def cosine(value):
        dot = (value*exact.conj()).real.mean((1,2))
        return dot/(physical_norm(value,1.).flatten()*exact_energy).clamp_min(1e-12)

    fractions = [(physical_norm(shift,1.)/raw_size).flatten() for shift in (analytic,noise,common,private)]
    measurements[active] = torch.stack((torch.ones_like(exact_energy),*fractions,split_error,norm_error,
                                         cosine(analytic),cosine(noise)), -1)
    return output,count,measurements


def render(iq,attached,baseline,tracker,models,*,threshold,measures=None):
    reference,_ = baseline(iq)
    output,count,measurements = recover(iq,attached,tracker,models[0],threshold)
    if measures is not None:
        measures.append(np.asarray(measurements.tolist()))
    return torch.cat((torch.stack((iq,reference)),output)),count


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,modes=MODES,comparisons=COMPARISONS,
         smoke_fn=partial(smoke,modes=MODES,renderer=render),
         evaluate_fn=partial(evaluate,modes=MODES,fields=FIELDS,renderer=render,
                             previous_run=BASE_RUN,previous_modes=BASE_MODES,replay_columns=REPLAY))
