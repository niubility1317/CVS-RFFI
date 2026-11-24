"""Compare frozen positive/negative noise transfer using held source U only."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.antithetic_detail_source import RECIPES, bind_models
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.gated_slope_source import restore
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import band_inner, fft, private_projection
from experiments.pamf_lora_main.private_polar_source import wave_quantities
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


def unpack(spectrum):
    return dict(common=spectrum.mean(1, keepdim=True).expand_as(spectrum),
                private=private_projection(spectrum))


def readouts(iq, tracker, models, threshold):
    frame = coordinates(tracker, iq)
    eligible = corruption_score(frame[0]) > threshold
    if not eligible.any():
        return [iq] * len(models), eligible
    selected = tuple(v[eligible] for v in frame)
    result = [restore(iq, eligible, model, selected, (.5,))[0] for model in models]
    return result, eligible


def power_ratio(output, raw, weight, masks):
    out = band_inner(output, output, weight, masks).sum(0)
    reference = band_inner(raw, raw, weight, masks).sum(0)
    if (reference <= 0).any():
        raise ValueError('Reference noise energy is zero')
    return (out / reference).tolist()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    started = perf_counter()
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    for arm in RECIPES:
        if read(root/f'{arm}-complete.json')['status'] != 'SOURCE_FROZEN':
            raise ValueError('Waveform audit needs both source freezes')
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    train, cal, held = partition(rx)
    held = np.concatenate([block[:30] for block in np.split(held, len(SOURCE))])
    if np.intersect1d(np.concatenate((train, cal)), held).size:
        raise ValueError('Expected disjoint held source U')
    tracker = PacketRestorer('consensus').eval()
    parent, arms, threshold = bind_models(root, contract, 'cpu')
    names = ('direct', 'rank_parent_half', *(f'{arm}_half' for arm in RECIPES))
    result = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full', 'residual'):
            result[route] = {}
            for view in (VIEWS[3], VIEWS[-1]):
                jobs = ((route, view, raw[i], rx[i], ids[i]) for i in held)
                captured, eps = (tensor(np.stack(values))
                                 for values in zip(*pool.map(replay, jobs, chunksize=6), strict=True))
                first, mirrored = captured, captured - 2*eps
                accepted = [0, 0]
                outputs = {k: [] for k in names}
                opposites = {k: [] for k in names}
                source_basis, power = [], []
                for plus, minus in zip(first.split(12), mirrored.split(12), strict=True):
                    recovered, active = readouts(plus, tracker, (parent, *arms), threshold)
                    opposite, active_other = readouts(minus, tracker, (parent, *arms), threshold)
                    _, rot, _, rms = coordinates(tracker, plus)
                    source_basis.append(rot*rms)
                    power.append(rms.square())
                    accepted[0] += int(active.sum())
                    accepted[1] += int(active_other.sum())
                    for name, a, b in zip(names, (plus, *recovered), (minus, *opposite), strict=True):
                        outputs[name].append(fft(complex_iq(a) / source_basis[-1]))
                        opposites[name].append(fft(complex_iq(b) / source_basis[-1]))
                basis, weight = torch.cat(source_basis), torch.cat(power)
                reference = fft(complex_iq(captured - eps) / basis)
                masks = named_band_masks(reference).float()
                noise = unpack(fft(complex_iq(eps) / basis))
                exact = unpack(reference)
                readout = {}
                for name in names:
                    single = unpack(torch.cat(outputs[name]))
                    odd = unpack(.5 * (torch.cat(outputs[name]) - torch.cat(opposites[name])))
                    even = unpack(.5 * (torch.cat(outputs[name]) + torch.cat(opposites[name])))
                    readout[name] = {
                        polarity: dict(remaining_noise_power_ratio=power_ratio(odd[polarity], noise[polarity], weight, masks),
                                       real_inference=wave_quantities(single[polarity], exact[polarity], weight, masks),
                                       pair_average=wave_quantities(even[polarity], exact[polarity], weight, masks))
                        for polarity in ('common', 'private')}
                result[route][view] = dict(n=len(held), source_positive_active=accepted[0],
                    isolated_negative_active=accepted[1], modes=readout)
                print('LORA_ANTITHETIC_HELD_U', route, view, *accepted, flush=True)
    print(json.dumps(dict(status='PASS', n=len(held), seconds=perf_counter()-started,
                          actual_readout_no_trace=True, reference_after_readouts=True, F_reads=0, V_reads=0,
                          TX_reads=0, target_reads=0, rows=result)), flush=True)


if __name__ == '__main__':
    main()
