"""Measure expansion DURA's complex transfer on intact unlabeled source packet pairs."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_capacity_source import RECIPES, RUN, recover
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.natural_detail import detail_bands
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.spectral_block_source import load_models


ORIGIN_CODE = '3de50180aaaa1179587a753d4f6dde201fc989c6'
ARMS = ('direct', 'coarse', 'final', 'half')
PAIRS = ('actual_shared', 'actual_independent', 'injected_shared')
COMPONENTS = ('common_1_8', 'private_1_8', 'common_9_32', 'private_9_32')
FIELDS = ('normalized_error', 'real_gain', 'quadrature_gain', 'orthogonal_error', 'reference_power')
CONDITIONS = (('clean', 0.), ('white', .14), ('white', .35), ('burst', .14), ('burst', .35))


def disturbance(value, style, rng):
    noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=rng)
    if style == 'burst':
        pulse = torch.rand((*value.shape[:-1], 8), device=value.device, generator=rng) < .24
        noise = noise*(.24+2.76*pulse.repeat_interleave(32, dim=-1))
    energy = noise.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-10)
    return noise/energy.sqrt()


def spectra(value):
    transform = torch.fft.fft(value, norm='ortho')
    common = transform.mean(dim=-2, keepdim=True)
    private = transform-common
    return [z[..., band] for band in detail_bands(value) for z in (common, private)]


def transfer(predicted, desired):
    parts = []
    for out, ref in zip(spectra(predicted), spectra(desired), strict=True):
        power = ref.abs().square().sum((-2, -1)).clamp_min(1e-12)
        gain = (out*ref.conj()).sum((-2, -1))/power
        error = (out-ref).abs().square().sum((-2, -1))/power
        orthogonal = (out-gain[..., None, None]*ref).abs().square().sum((-2, -1))/power
        parts.append(torch.stack((error, gain.real, gain.imag, orthogonal, power.expand_as(error)), -1))
    return torch.stack(parts, -2)


def run_batch(anchor, peer, clean_anchor, clean_peer, frame_anchor, frame_peer, tracker, models, threshold,
              style, dose, rng):
    bands = detail_bands(clean_anchor)
    actual_delta = clean_peer-clean_anchor
    inserted = .5*torch.fft.ifft(torch.fft.fft(actual_delta, norm='ortho')*(bands[0] | bands[1]), norm='ortho')
    shared = dose*disturbance(clean_anchor, style, rng)
    independent = dose*disturbance(clean_peer, style, rng)
    views = torch.cat((anchor+shared*frame_anchor, peer+shared*frame_peer, peer+independent*frame_peer,
                       anchor+(shared+inserted)*frame_anchor), dim=0)
    received = real_iq(views)
    restored, active, _ = recover(received, tracker, models, threshold)
    samples = torch.stack([complex_iq(x) for x in (received, restored[0], restored[2], restored[3])])
    samples = samples.unflatten(1, (4, len(anchor)))
    a = samples[:, 0]/frame_anchor
    comparisons = (samples[:, 1]/frame_peer-a, samples[:, 2]/frame_peer-a, samples[:, 3]/frame_anchor-a)
    metrics = torch.stack([transfer(pred, ref) for pred, ref in
                           zip(comparisons, (actual_delta, actual_delta, inserted), strict=True)])
    # Canonical tensor order: packet, pair, readout, component, metric.
    metrics = metrics.permute(2, 0, 1, 3, 4)
    if (not torch.isfinite(metrics).all()
            or (metrics[:, (0, 2), 0, :, 0].amax() > 1e-5)):
        raise ValueError('Shared receive disturbance failed to cancel in source pair coordinates')
    return metrics, active


def summarize(rows):
    return dict(n=len(rows), median=np.median(rows, axis=0).tolist(),
                p10=np.quantile(rows, .1, axis=0).tolist(), p90=np.quantile(rows, .9, axis=0).tolist())


def loss_sensitivity(reference):
    """Measure the original transfer loss's scale at a half-strength delivered difference."""
    fields = {}
    for name, band in zip(COMPONENTS, spectra(reference), strict=True):
        power = band.abs().square().mean(-1)
        # For squared error (gain - 1)^2 * power, this is the magnitude of
        # the original loss derivative w.r.t. gain, before outer averaging.
        slope = power / (.002 + 1.25*power)
        fields[name] = {key: np.quantile(value.flatten().tolist(), (.1, .5, .9)).tolist()
                        for key, value in (('mean_bin_power', power), ('half_gain_slope', slope))}
    return fields


@torch.no_grad()
def measure(rt, device):
    begun = perf_counter()
    if (rt['run_id'], rt['code_commit']) != (RUN, ORIGIN_CODE):
        raise ValueError('Expected the original 40,000-step expanded LoRa DURA')
    numerical_policy()
    source = Path(rt['source_root'])
    contract = source_contract(source)
    native, rx, _, _ = source_role(source, contract, 'U_s')
    _, _, held = partition(rx, fit=4000)
    tracker = PacketRestorer('consensus').to(device).eval()
    models, threshold = load_models(Path(rt['run_root']), contract, device, {'expanded': RECIPES['expanded']})
    ground = tensor(np.stack([chirp_windows(native[k]) for k in held]), device)
    clean, phase, _, rms = coordinates(tracker, ground)
    source_basis = phase*rms
    original = complex_iq(ground)
    peers = np.arange(len(held)).reshape(len(SOURCE), -1)
    peers = np.roll(peers, 1, axis=-1).flatten()
    if not np.all(rx[held][peers] == rx[held]) or np.any(peers == np.arange(len(held))):
        raise ValueError('Every actual pair must contain two different source packets from one RX')
    peers = torch.tensor(peers.tolist(), device=device)
    difference = clean[peers] - clean
    sensitivity = dict(actual=loss_sensitivity(difference), injected=loss_sensitivity(.5*difference))
    results = {}
    for style, dose in CONDITIONS:
        rng = torch.Generator(device=device).manual_seed(2026112595)
        rows, active = [], 0
        indices = torch.arange(len(ground), device=device)
        for first in indices.split(12):
            second = peers[first]
            readings, count = run_batch(original[first], original[second], clean[first], clean[second],
                                        source_basis[first], source_basis[second], tracker, models, threshold,
                                        style, dose, rng)
            rows.extend(readings.tolist())
            active += count
        values = np.asarray(rows)
        name = style if not dose else f'{style}_{dose:.2f}'
        results[name] = dict(active=active, all=summarize(values),
                             by_receiver={key: summarize(values[rx[held] == key]) for key in SOURCE})
    return dict(status='VERIFIED', source_commit=rt['code_commit'], role='U_s', held=len(held),
                pairs=PAIRS, components=COMPONENTS, fields=FIELDS, arms=ARMS, seconds=perf_counter()-begun,
                conditions=results, loss_sensitivity=sensitivity, F_reads=0, TX_reads=0, V_reads=0,
                target_reads=0, trace_reads=0,
                optimizer_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(measure(read(args.runtime), args.device), allow_nan=False))
