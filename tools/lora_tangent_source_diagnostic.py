"""Test whether empirical receive directions distinguish weak source edits from perturbations."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.empirical_tangent_source import RUN, recipes
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.natural_detail import weak_difference
from experiments.pamf_lora_main.runner import activate


@torch.no_grad()
def probe(rt, projection):
    device = 'cuda:0'
    recipe = recipes(rt)['protected']
    (_, _, hold), _ = setup(rt,device,fit_per_rx=recipe.fit_per_rx,bank_per_rx=recipe.bank_per_rx)
    if len(hold)!=500:
        raise ValueError('Expected the fixed disjoint source U holdout')
    model = recipe.model().to(device).eval()
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026112427)
    tables = {}
    for receiver, rows in zip(SOURCE,hold.chunk(len(SOURCE)),strict=True):
        samples, scores = [], []
        for iq in rows.split(25):
            source = complex_iq(iq)
            z, phase, _, scale = coordinates(tracker,iq)
            signal_delta = weak_difference(z)*phase*scale
            received = recipe.perturb(source,rng)
            observed, rot, _, rms = coordinates(tracker,real_iq(received))
            basis = rot*rms
            waves = [received,received+signal_delta,2*received-source]
            own = projection(model,observed,basis,waves)[...,0]
            swap = projection(model,observed.roll(1,0),basis,waves)[...,0]
            samples.append(np.concatenate((own,swap)))
            scores.extend(corruption_score(observed).tolist())
        signal, nuisance, swapped_signal, swapped_nuisance = np.concatenate(samples,axis=1)
        score = np.asarray(scores)
        values = np.stack((signal,nuisance,swapped_signal,swapped_nuisance))
        chosen = (score >= np.median(score))
        tables[receiver] = dict(n=len(rows),all=np.quantile(values,(.1,.5,.9),axis=1).T.tolist(),
                               higher_disturbance=np.quantile(values[:,chosen],(.1,.5,.9),axis=1).T.tolist())
    return dict(n=len(hold),keys=['measured_weak_signal','generic_disturbance','unrelated_weak_signal',
                                'unrelated_disturbance'],quantiles=[.1,.5,.9],by_receiver=tables,
                F_reads=0,V_reads=0,TX_reads=0,target_reads=0)


if __name__=='__main__':
    runtime = read(Path(sys.argv[1]))
    if runtime.get('run_id')!=RUN:
        raise ValueError('Expected the registered LoRa empirical tangent release')
    activate(runtime)
    from tools.lora_local_phase_fidelity import direction_fraction
    print(json.dumps(probe(runtime,direction_fraction),allow_nan=False))
