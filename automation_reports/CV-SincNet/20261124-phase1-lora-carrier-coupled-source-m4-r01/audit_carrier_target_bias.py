"""Compare current carrier supervision with independently tracked source amplitude."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.carrier_coupled import amplitude_perturb, observed_frame, pair_builder
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.runner import activate


def readout(tracker, clean, captured):
    obs, phase, _, rms = observed_frame(tracker, real_iq(captured))
    clean_frame, _, _, true_rms = observed_frame(tracker, real_iq(clean))
    captured_amplitude = obs.mean(-1).abs()*rms[...,0]
    stage_amplitude = (clean*phase.conj()).mean(-1).abs()
    source_amplitude = clean_frame.mean(-1).abs()*true_rms[...,0]
    ratios = {}
    for name, reference in (('target_in_received_frame',stage_amplitude),
                             ('source_own_frame',source_amplitude)):
        ratio = reference/captured_amplitude.clamp_min(1e-6)
        ratios[name] = dict(median=float(ratio.median()),
                            below_half=float((ratio<.5).float().mean()))
    ratios['target_over_source_amplitude'] = float((stage_amplitude/source_amplitude.clamp_min(1e-6)).median())
    return ratios


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    (_,_,held),_ = setup(rt,'cpu')
    clean = complex_iq(torch.cat([rx[:30] for rx in held.chunk(5)]))
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator(device='cpu').manual_seed(2026113901)
    generic = pair_builder(clean,torch.zeros_like(clean),colored_observation,rng,with_identity=False)[:,0]
    gain, _ = amplitude_perturb(clean,clean,generator=rng)
    adjusted = colored_observation(gain,rng)
    print(json.dumps(dict(status='COMPLETE',n=len(clean),target_reads=0,TX_reads=0,F_reads=0,
        original_generic=readout(tracker,clean,generic),
        amplitude_and_noise_only=readout(tracker,clean,adjusted)), allow_nan=False))


if __name__=='__main__':
    main()
