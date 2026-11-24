"""Screen cross-chirp product features on native weak LoRa sidebands."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation
from experiments.pamf_lora_main.repeat_product import (
    RIDGE, independent_noise, moments, normalized_spectrum, real_parts, repeat_features, solve,
)


SEED = 2026112435
FIT_PER_RX = 4_000
HELD_PER_RX = 100
REGIMES = ('ground', 'independent_noise', 'generic_burst')
MODES = ('linear', 'cross_product', 'shuffled_product')


def examples(samples, indices, tracker, noise, desired, *, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    for group in np.array_split(indices, (len(indices)+95)//96):
        iq = torch.from_numpy(np.stack([chirp_windows(samples[i]) for i in group])).to(device)
        native = normalized_spectrum(coordinates(tracker, iq)[0])
        if noise is None:
            received = native
        else:
            disturbed = real_iq(noise(complex_iq(iq), rng))
            received = normalized_spectrum(coordinates(tracker, disturbed)[0])
        native = native[..., desired].unflatten(1,(2,3)).permute(0,2,3,1)
        linear, products = repeat_features(received, desired)
        _, displaced = repeat_features(received, desired, shuffled=True)
        joined = (real_parts(linear), real_parts(torch.cat((linear, products), -1)),
                  real_parts(torch.cat((linear, displaced), -1)))
        yield joined, real_parts(native)


def spectral_risk(output, target, signed_bins):
    value = torch.view_as_complex(output.unflatten(-1,(2,2)).contiguous()).transpose(-1,-2)
    truth = torch.view_as_complex(target.unflatten(-1,(2,2)).contiguous()).transpose(-1,-2)
    value = value.transpose(1, 2).flatten(1, 2)
    truth = truth.transpose(1, 2).flatten(1, 2)
    error = value - truth
    common = error.mean(1,keepdim=True)
    components = torch.stack((common.abs().square().sum(1), (error-common).abs().square().mean(1)), 1)
    return torch.stack([components[..., m].mean(-1) for m in (signed_bins.abs()<=8, signed_bins.abs()>8)], -1)


@torch.no_grad()
def probe(source, device, *, fit_per_rx=FIT_PER_RX, held_per_rx=HELD_PER_RX):
    start = perf_counter()
    tracker = PacketRestorer('consensus').to(device).eval()
    contract = source_contract(source)
    samples, receivers, _, _ = source_role(source, contract, 'U_s')
    fit, _, held = partition(receivers, fit=fit_per_rx, held=held_per_rx)
    signed = torch.tensor([*range(-32,0), *range(1,33)], device=device)
    desired = signed.remainder(256)
    results = {}
    for regime_index, (regime, augment) in enumerate(zip(REGIMES, (None, independent_noise, burst_observation), strict=True)):
        fitting, held_batches = [], []
        for r, selected in enumerate(np.split(fit,len(SOURCE))):
            matrix = [0,0,0]
            seen = 0
            for values, target in examples(samples, selected, tracker, augment, desired,
                                            device=device, seed=SEED+100*regime_index+r):
                for i, arm in enumerate(values):
                    product, batch = moments(arm, target)
                    matrix[i] = matrix[i] + product
                seen += batch
            if seen != len(selected):
                raise ValueError('The source fit size changed')
            fitting.append(matrix)
        for r, selected in enumerate(np.split(held,len(SOURCE))):
            held_batches.append(list(examples(samples, selected, tracker, augment, desired,
                                               device=device, seed=SEED+100*regime_index+40+r)))
        per_receiver = {}
        for held_rx, (name, groups) in enumerate(zip(SOURCE,held_batches,strict=True)):
            count = (len(SOURCE)-1) * fit_per_rx
            predictors = [
                solve(sum(matrix[i] for r,matrix in enumerate(fitting) if r!=held_rx), count, inputs.shape[-1])
                for i,inputs in enumerate(groups[0][0])
            ]
            risks = []
            for inputs, reference in groups:
                offered = [torch.einsum('bfki,fkio->bfko', arm-mean, weight) + center
                           for arm,(mean,center,weight) in zip(inputs,predictors,strict=True)]
                risks.append(torch.stack([spectral_risk(x, reference, signed) for x in offered], 1))
            risk = torch.cat(risks).double()
            score = 1 - risk[:,1:].sum(0) / risk[:,:1].sum(0).clamp_min(1e-12)
            per_receiver[name] = dict(n=len(risk), errors=risk.mean(0).tolist(), change_vs_linear=score.tolist())
        results[regime] = per_receiver
    return dict(status='SOURCE_U_ONLY', fit=len(fit), held=len(held), seed=SEED, ridge=RIDGE,
                modes=MODES, groups=['common','private'], bands=['1–8 kHz','9–32 kHz'],
                by_regime=results, F_reads=0, V_reads=0, TX_reads=0, target_reads=0, exact_reads=0,
                worker_seconds=perf_counter()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    out=probe(args.source,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(out,stream,ensure_ascii=False,allow_nan=False,indent=2)
    print(args.output)


if __name__=='__main__':
    main()
