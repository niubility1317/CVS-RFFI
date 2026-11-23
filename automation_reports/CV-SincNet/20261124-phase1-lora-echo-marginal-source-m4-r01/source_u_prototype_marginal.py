import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.echo_marginal_source import RUN
from experiments.pamf_lora_main.crossfit_pilot import LAGS, apply_transfer, generic_taps
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
project = Path('/home/szu2070436088/2510044040/CV-SincNet')
rt = read(project/'releases'/f'{RUN}-release'/'runtime.json')
(fit, _, held), _ = setup(rt, 'cpu')
tracker = PacketRestorer('consensus').eval()
freq = (256 * torch.fft.fftfreq(256)).round().to(torch.long)
kept = (freq.abs() >= 1) & (freq.abs() <= 32)
lower = freq[kept].abs() <= 8
rng = torch.Generator().manual_seed(2026112453)
modes = ('average', 'global_gaussian', 'mixture', 'marginalized_echo', 'echo_uncertainty_average', 'prototype_echo_average')
n_per_rx = 100

def vector_common(waves, weight):
    common = (waves * weight).sum(1)
    spectrum = torch.fft.fft(common, norm='ortho')[..., kept]
    return torch.cat((spectrum.real, spectrum.imag), dim=-1)

def signal_stats(iq):
    observed, rotation, quality, rms = coordinates(tracker, iq)
    weight = quality.square().clamp_min(1e-6)
    weight = weight / weight.sum(1, keepdim=True)
    variance = .5 * (weight.square() * observed_noise_rms(observed).square()).sum(1)
    return observed, weight, variance, rotation * rms, rotation

def relative_echo_columns(observed, template, weights):
    carrier = observed.mean(-1, keepdim=True)
    cols = torch.stack([template.roll(lag,-1) for lag in LAGS], dim=-1)
    native_action = carrier[...,None] * cols * template.conj()[...,None]
    common_action = (native_action * weights[...,None]).sum(1)
    spectrum = torch.fft.fft(common_action, dim=1, norm='ortho')[:, kept]
    real_basis = torch.cat((spectrum.real,spectrum.imag), dim=1)
    imag_basis = torch.cat((-spectrum.imag,spectrum.real), dim=1)
    taper = spectrum.real.new_tensor([.06/(1+abs(lag)) for lag in LAGS]*2)
    return torch.cat((real_basis,imag_basis), dim=-1) * taper

def bayes_echo_mean(measurement, variances, source_eigenvalues, factor):
    diagonal = source_eigenvalues + variances
    inverse_rhs = measurement / diagonal
    scaled = factor / diagonal[...,None]
    dim = factor.shape[-1]
    small = torch.eye(dim,dtype=factor.dtype,device=factor.device) + factor.mT @ scaled
    rhs = factor.mT @ inverse_rhs[...,None]
    posterior = torch.linalg.solve(small,rhs)
    estimate = inverse_rhs - (scaled @ posterior).squeeze(-1)
    score = .5 * ((rhs.mT @ posterior).flatten() - torch.linalg.slogdet(small).logabsdet)
    return source_eigenvalues * estimate, (.25 * score).sigmoid()[:,None]

def posterior_prototype_echo(measurement, variance, local_variance, factor, library):
    """Integrate the unknown clean population component and unknown short taps."""
    difference = measurement[:, None, :] - library
    diagonal = variance + local_variance
    scaled = factor / diagonal[..., None]
    small = torch.eye(factor.shape[-1], dtype=factor.dtype) + factor.mT @ scaled
    rhs = scaled.mT @ difference.mT
    coefficients = torch.linalg.solve(small, rhs)
    square_sum = (difference.square() / diagonal[:,None]).sum(-1)
    h0 = -.5 * square_sum
    h1 = h0 + .5 * ((rhs * coefficients).sum(1) - torch.linalg.slogdet(small).logabsdet[:, None])
    center0 = h0.softmax(-1) @ library
    center1 = h1.softmax(-1) @ library
    prior_trust = local_variance / diagonal
    unused_echo = center0 + prior_trust * (measurement - center0)
    correction = torch.linalg.solve(small, scaled.mT @ (measurement-center1)[...,None])
    integrated_echo = center1 + prior_trust * (measurement-center1) - local_variance * (scaled @ correction).squeeze(-1)
    posterior_echo = (.25 * (h1.logsumexp(-1) - h0.logsumexp(-1))).sigmoid()[:, None]
    return torch.lerp(unused_echo, integrated_echo, posterior_echo), posterior_echo

def statistic(prediction, actual):
    stats = {}
    for name, selected in (('1_to_8', lower), ('9_to_32', ~lower)):
        selection = torch.cat((selected, selected))
        a, p = actual[:, selection], prediction[:, selection]
        error = (a - p).square().sum(-1)
        energy = a.square().sum(-1)
        gain = (a*p).sum(-1) / energy.clamp_min(1e-8)
        stats[name] = torch.stack((error, energy, gain), dim=-1)
    return stats

with torch.no_grad():
    prototypes = [[] for _ in SOURCE]
    for i, source in enumerate(fit.chunk(len(SOURCE))):
        for iq in source[:n_per_rx].split(20):
            y, weights, _, _, _ = signal_stats(iq)
            prototypes[i].append(vector_common(y, weights))
    outputs = {}
    evidences = {}
    for receiver, group in enumerate(held.chunk(len(SOURCE))):
        banks = torch.cat([s for i,rx in enumerate(prototypes) if i != receiver for s in rx])
        mean = banks.mean(0)
        center = banks - mean
        covariance = center.T @ center / (len(center)-1)
        eigenvalues, basis = torch.linalg.eigh(covariance)
        eig = eigenvalues.clamp_min(1e-8)
        library = center @ basis
        for iq in group.split(20):
            clean = complex_iq(iq)
            echoed = apply_transfer(clean, generic_taps(clean,rng))
            noise = colored_observation(echoed,rng) - echoed
            noise_only = colored_observation(clean,rng) - clean
            for label, received in (
                ('clean', clean), ('light_echo', echoed + .25 * noise),
                ('broad_echo', echoed + .85 * noise), ('broad_noise', clean + .85 * noise_only),
                ('light_noise', clean + .25 * noise_only),
            ):
                y,w,v,rotation,template = signal_stats(real_iq(received))
                x = vector_common(y, w)
                actual = vector_common(clean / rotation, w)
                coordinates_real = (x - mean) @ basis
                global_posterior = coordinates_real * eig / (eig + v)
                mixture_variance = .25 * eig
                likelihood_variance = (mixture_variance + v).clamp_min(1e-8)
                diff = coordinates_real[:,None] - library
                weights = (-.5 * (diff.square()/likelihood_variance[:,None]).sum(-1)).softmax(-1)
                posterior_center = weights @ library
                mixture_posterior = posterior_center + (coordinates_real-posterior_center)*mixture_variance/likelihood_variance
                cols = basis.T @ relative_echo_columns(y,template,w)
                marginalized, fitted = bayes_echo_mean(coordinates_real, v, eig, cols)
                effic = torch.lerp(global_posterior, marginalized, fitted)
                prototypes_echo, fitted_mixture = posterior_prototype_echo(coordinates_real,v,mixture_variance,cols,library)
                for family, weight in (('gaussian',fitted),('prototypes',fitted_mixture)):
                    evidences.setdefault(label, {}).setdefault(family,[]).append(weight.squeeze(1))
                for mode, result in zip(modes, (x, mean + global_posterior @ basis.T,
                                                mean + mixture_posterior @ basis.T,
                                                mean + marginalized @ basis.T,
                                                mean + effic @ basis.T,
                                                mean + prototypes_echo @ basis.T),strict=True):
                    for band,value in statistic(result,actual).items():
                        outputs.setdefault((label, mode, band), []).append(value)

records={}
for (label,mode,band), values in outputs.items():
    error, energy, gain = torch.cat(values).T
    assert len(error)==500
    overall = {
        'relative_error': float(error.sum()/energy.sum()),
        'packet_relative_error_p50': float((error/energy.clamp_min(1e-8)).median()),
        'gain_p50': float(gain.median()),
        'per_rx_relative_error': [(e.sum()/p.sum()).item() for e,p in zip(error.split(100),energy.split(100))],
    }
    records.setdefault(label, {}).setdefault(band,{})[mode] = overall
print(json.dumps(dict(packets=500, fit_per_fold=400, held_per_rx=100, receiver_order=SOURCE,
                      target_reads=0,V_reads=0,F_reads=0,TX_reads=0, measurements=records,
                      echo_probability={name:{family:torch.cat(values).quantile(torch.tensor([.1,.5,.9])).tolist()
                                             for family,values in models.items()} for name,models in evidences.items()})),flush=True)
