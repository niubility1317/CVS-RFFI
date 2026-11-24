"""Compare guard power and repetition-shape evidence across held source receivers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import chirp_windows


DOSES = (.0075, .015, .03, .06)
FAMILIES = ('white', 'smooth_colored', 'in_band')
VIEWS = ('ground', *tuple(f'{family}_{dose}' for family in FAMILIES for dose in DOSES))
ARMS = {'guard': slice(0, 8), 'repeat': slice(8, None), 'combined': slice(None)}
SEED = 2026112452


class Evidence:
    def __init__(self, device):
        self.tracker = PacketRestorer('consensus').to(device).eval()
        self.frequencies = (256 * torch.fft.fftfreq(256, device=device)).round().abs()
        window = torch.hann_window(256, device=device)
        self.window = window / window.square().mean().sqrt()

    def guard(self, original):
        rms = original.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        power = torch.fft.fft((original / rms) * self.window, norm='ortho').abs().square()
        features = []
        for lo, hi in ((70, 91), (92, 119)):
            band = power[..., (self.frequencies >= lo) & (self.frequencies <= hi)]
            floor = band.median(-1).values.clamp_min(1e-12)
            mean = band.mean(-1).clamp_min(1e-12)
            for summary in (floor.log(), (floor/mean).log()):
                order = summary.sort(1).values
                features.extend((order[:, 2], order[:, 4]))
        return features

    def repeat(self, original):
        aligned, _, _, _ = coordinates(self.tracker, real_iq(original))
        first, second = aligned.unflatten(1, (3, 2)).unbind(2)
        difference = first - second
        centered = difference - difference.mean(-1, keepdim=True)
        mean_power = centered.abs().square().mean(-1).clamp_min(1e-12)
        summaries = [(centered.square().mean(-1).abs() / mean_power).median(1).values]
        for lag in (1, 2, 8):
            left, right = centered[..., lag:], centered[..., :-lag]
            inner = (left * right.conj()).mean(-1)
            scale = (left.abs().square().mean(-1) * right.abs().square().mean(-1)).sqrt().clamp_min(1e-12)
            for value in ((inner/scale).real, (inner/scale).imag):
                summaries.append(value.median(1).values)
        power = torch.fft.fft(centered, norm='ortho').abs().square()
        for low, high in ((1, 8), (9, 32), (33, 64), (65, 128)):
            band = power[..., (self.frequencies >= low) & (self.frequencies <= high)]
            per_pair = band.mean(-1) / mean_power
            summaries.append(per_pair.clamp_min(1e-8).log().median(1).values)
        return summaries

    def __call__(self, value):
        return torch.stack((*self.guard(value), *self.repeat(value)), 1)


def perturb(value, family, generator):
    noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
    frequency = torch.fft.fftfreq(value.shape[-1], device=value.device)
    if family == 'smooth_colored':
        center = torch.rand((len(value), 1, 1), generator=generator, device=value.device)
        density = 1. + .8*(2*torch.pi*(frequency-center)).cos()
        noise = torch.fft.ifft(torch.fft.fft(noise)*density.sqrt())
    elif family == 'in_band':
        gain = ((.26-frequency.abs())/.04).clamp(0., 1.)
        noise = torch.fft.ifft(torch.fft.fft(noise)*gain)
    return noise / noise.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()


def features(iq, rows, evaluator, device, *, training, random_seed):
    rng = torch.Generator(device=device).manual_seed(random_seed)
    buffers = {'ground': []}
    for batch in np.array_split(rows, (len(rows)+63)//64):
        ground = complex_iq(torch.from_numpy(np.stack([chirp_windows(iq[k]) for k in batch])).to(device))
        buffers['ground'].append(evaluator(ground).tolist())
        for family in FAMILIES if training is not None else ():
            noise = perturb(ground, family, rng)
            if training:
                strength = .0075 * 8. ** torch.rand((len(ground), 1, 1), device=device, generator=rng)
                buffers.setdefault('generic', []).append(evaluator(ground + strength*noise).tolist())
            else:
                for dose in DOSES:
                    buffers.setdefault(f'{family}_{dose}', []).append(evaluator(ground + dose*noise).tolist())
    return {view: np.concatenate(blocks, axis=0) for view, blocks in buffers.items()}


def fit(negatives, positives):
    center = np.median(negatives, axis=0)
    q25, q75 = np.quantile(negatives, (.25, .75), axis=0)
    scale = np.maximum(q75-q25, 1e-3)
    x = np.clip((np.concatenate((negatives, positives)) - center)/scale, -10., 10.)
    labels = np.repeat((0., 1.), (len(negatives), len(positives)))
    importance = np.repeat((.5/len(negatives), .5/len(positives)), (len(negatives), len(positives)))

    def objective(params):
        weights, offset = params[:-1], params[-1]
        prediction = x@weights + offset
        error = importance * (expit(prediction)-labels)
        loss = importance @ (np.logaddexp(0., prediction)-labels*prediction) + .025*(weights@weights)
        gradient = np.r_[x.T@error + .05*weights, error.sum()]
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B')
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Receiver-evidence calibration failed')
    return center, scale, result.x[:-1], result.x[-1]


def score(params, feature):
    center, scale, weights, offset = params
    return np.clip((feature-center)/scale, -10., 10.)@weights + offset


@torch.inference_mode()
def run(source, device):
    started = perf_counter()
    contract = source_contract(source)
    iq, receivers, _, _ = source_role(source, contract, 'U_s')
    fit_rows, cal_rows, held_rows = partition(receivers, fit=1000)
    evaluator = Evidence(device)
    fitting, calibration, held = {}, {}, {}
    for i, receiver in enumerate(SOURCE):
        fitting[receiver] = features(iq, fit_rows[receivers[fit_rows] == receiver], evaluator, device,
                                    training=True, random_seed=SEED+i)
        calibration[receiver] = features(iq, cal_rows[receivers[cal_rows] == receiver], evaluator, device,
                                        training=None, random_seed=SEED+20+i)['ground']
        held[receiver] = features(iq, held_rows[receivers[held_rows] == receiver], evaluator, device,
                                 training=False, random_seed=SEED+40+i)
    result = {}
    for leave in SOURCE:
        others = tuple(name for name in SOURCE if name != leave)
        negatives = np.concatenate([fitting[r]['ground'] for r in others])
        positives = np.concatenate([fitting[r]['generic'] for r in others])
        controls = np.concatenate([calibration[r] for r in others])
        result[leave] = {}
        for name, features_slice in ARMS.items():
            parameters = fit(negatives[:, features_slice], positives[:, features_slice])
            threshold = float(score(parameters, controls[:, features_slice]).max())+.01
            result[leave][name] = dict(threshold=threshold, counts={
                view: int((score(parameters, matrices[:, features_slice]) > threshold).sum())
                for view, matrices in held[leave].items()})
    return dict(status='SOURCE_U_ONLY', seed=SEED, fit_per_receiver=1000, calibration_per_receiver=500,
                held_per_receiver=100, target_reads=0, F_reads=0, TX_reads=0, V_reads=0, draws=VIEWS,
                by_receiver=result, worker_seconds=perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    options = parser.parse_args()
    data = run(options.source, options.device)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
    print(options.output)
