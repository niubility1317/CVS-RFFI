"""Audit source-U IQ preservation after the empirical source bank freezes."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.empirical_packet_prior import detail_bins, observation_bins, weights
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms
from experiments.pamf_lora_main.empirical_packet_prior_source import MODES, bind_models, readouts
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import fft, private_projection
from experiments.pamf_lora_main.private_polar_source import wave_quantities
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


def components(value):
    return {'common': value.mean(1, keepdim=True).expand_as(value),
            'private': private_projection(value)}



def likelihood_stats(spectrum, expected, bank, variance):
    bins = detail_bins(spectrum.device)
    keys = bins[observation_bins(bins)]
    templates = bank.mean(1)[:, observation_bins(bins)]
    common_noise = variance.sum(1, keepdim=True) / spectrum.shape[1]**2
    statistics = []
    for observed, clean, noise in zip(spectrum.mean(1)[:, keys].split(20),
                                       expected.mean(1)[:, keys].split(20),
                                       common_noise.split(20), strict=True):
        received_weight = weights(observed, templates, noise)
        noiseless_weight = weights(clean, templates, noise)
        statistics.append(torch.stack((received_weight.square().sum(-1).reciprocal(),
                                       noiseless_weight.square().sum(-1).reciprocal(),
                                       (received_weight - noiseless_weight).abs().sum(-1) * .5), dim=-1))
    values = torch.cat(statistics)
    return {key: torch.quantile(column, column.new_tensor((.10, .50, .90))).tolist()
            for key, column in zip(('ESS_received', 'ESS_no_added_noise_same_variance', 'total_variation'),
                                   values.T, strict=True)}


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    started = perf_counter()
    rt = read(Path('runtime.json'))
    activate(rt)
    if rt['runner'] != 'experiments.pamf_lora_main.empirical_packet_prior_source':
        raise ValueError('Audit expects the empirical source IQ bank release')
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    fitted, _, held = partition(rx)
    held = np.concatenate([part[:30] for part in np.split(held, len(SOURCE))])
    if np.intersect1d(fitted, held).size:
        raise ValueError('Waveform panel must be disjoint from source moments fit')
    tracker = PacketRestorer('consensus').eval()
    (model, prior), threshold = bind_models(root, contract, 'cpu')
    def norm(value):
        return torch.linalg.matrix_norm(value, dim=(-2, -1))
    report = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full', 'residual'):
            report[route] = {}
            for view in (VIEWS[3], VIEWS[-1]):
                jobs = ((route, view, raw[i], rx[i], ids[i]) for i in held)
                captures, noise = (tensor(np.stack(values))
                                   for values in zip(*pool.map(replay, jobs, chunksize=6), strict=True))
                recovered = {name: [] for name in ('direct', *MODES[2:])}
                basis, scales, active = [], [], 0
                dose_mismatch = 0.
                for batch in captures.split(12):
                    waves, count = readouts(batch, tracker, model, threshold, prior=prior)
                    _, rot, _, rms = coordinates(tracker, batch)
                    basis.append(rot * rms)
                    scales.append(rms.square())
                    for key, output in zip(recovered, (batch, *waves), strict=True):
                        recovered[key].append(fft(complex_iq(output) / basis[-1]))
                    mask = named_band_masks(rot)[:2]
                    for band in mask:
                        vectors = [fft(complex_iq(out - batch) / basis[-1])[..., band] * rms
                                   for out in (waves[0], waves[-2], waves[-1])]
                        lengths = torch.stack([norm(v) for v in vectors])
                        dose_mismatch = max(dose_mismatch, float((lengths[0] - lengths[1:]).abs().max()))
                    active += count
                # Use the isolated received-coordinate reference after recovery only.
                expected = fft(complex_iq(captures - noise) / torch.cat(basis))
                masks = named_band_masks(expected).float()
                references = components(expected)
                rms_weights = torch.cat(scales)
                received = torch.cat(recovered['direct'])
                noise_var = observed_noise_rms(complex_iq(captures) / torch.cat(basis)).square().squeeze(-1)
                likelihood = likelihood_stats(received, expected, prior[0], noise_var)
                report[route][view] = dict(n=len(held), active=active,
                    physical_dose_max_mismatch=dose_mismatch, likelihood=likelihood, by_mode={
                        key: {name: wave_quantities(component, references[name], rms_weights, masks)
                              for name, component in components(torch.cat(values)).items()}
                        for key, values in recovered.items()})
                print('LORA_EMPIRICAL_BANK_HELD_U', route, view, active, flush=True)
    print(json.dumps(dict(status='PASS', n=len(held), seconds=perf_counter() - started,
          isolated_reference_after_readouts=True,
          F_reads=0, V_reads=0, TX_reads=0, target_reads=0, rows=report)), flush=True)


if __name__ == '__main__':
    main()
