"""Audit actual alias-probe directional corrections on the held source U role."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.alias_curvature_source import MODES, readouts
from experiments.pamf_lora_main.chirp_band_power_source import model_binding
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import fft, private_projection
from experiments.pamf_lora_main.private_polar_source import wave_quantities
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    started = perf_counter()
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    fit, _, held = partition(rx)
    held = np.concatenate([part[:30] for part in np.split(held, len(SOURCE))])
    if np.intersect1d(fit, held).size:
        raise ValueError('Directional recovery audit must use disjoint held U')
    tracker = PacketRestorer('consensus').eval()
    models = model_binding(rt, contract)
    report = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            report[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs = ((route,view,raw[idx],rx[idx],ids[idx]) for idx in held)
                iq, epsilon = (tensor(np.stack(v)) for v in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                axes, weights, changed = [], [], []
                variants = {name:[] for name in ('direct', *MODES[2:])}
                for batch in iq.split(16):
                    estimates,_ = readouts(batch,tracker,*models)
                    y, rot, _, rms = coordinates(tracker,batch)
                    axes.append(rot*rms)
                    weights.append(rms.square())
                    changed.append(corruption_score(y)>models[1])
                    for mode,out in zip(variants,(batch,*estimates),strict=True):
                        value = fft(complex_iq(out)/axes[-1])
                        variants[mode].append(private_projection(value))
                actual=private_projection(fft(complex_iq(iq-epsilon)/torch.cat(axes)))
                masks=named_band_masks(actual).float()
                active=torch.cat(changed)
                full={name:torch.cat(v) for name,v in variants.items()}
                part=named_band_masks(actual)[0]
                weight = torch.cat(weights)
                anchored = full['antithetic_half'][active][...,part]
                true_residual = actual[active][...,part] - anchored

                def norm(v):
                    return (v.abs().square()*weight[active]).sum((-2,-1)).clamp_min(1e-20).sqrt()

                deltas=[full[name][active][...,part]-anchored for name in ('near_extrapolate','near_reverse')]
                dose_difference=float((norm(deltas[0])-norm(deltas[1])).abs().max())
                added=deltas[0]
                align=(added.conj()*true_residual*weight[active]).sum((-2,-1)).real
                cosine=align/(norm(added)*norm(true_residual))
                report[route][view] = dict(n=len(held),active=int(active.sum()),
                    physical_matched_dose_max_difference=dose_difference,
                    relative_edit_median=float((norm(added)/norm(anchored)).median()),
                    true_direction_cosine_median=float(cosine.median()),
                    true_direction_positive_fraction=float((cosine>0).float().mean()),
                    waves={name:wave_quantities(v,actual,weight,masks) for name,v in full.items()})
                print('LORA_ALIAS_CURVATURE_WAVE_AUDIT',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='PASS',n=len(held),reference_after_observed=True,seconds=perf_counter()-started,
          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,rows=report)),flush=True)


if __name__ == '__main__':
    main()
