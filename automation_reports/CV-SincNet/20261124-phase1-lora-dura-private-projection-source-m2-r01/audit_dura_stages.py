"""Trace each fixed DURA solve on the held source U packets."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.antithetic_detail_source import load_arm
from experiments.pamf_lora_main.carrier_coupled import observed_frame, packet_frame
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_private_projection_source import RUN
from experiments.pamf_lora_main.dura_prior_width_source import RECIPES, RUN as WIDTH_RUN
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.private_band_risk_source import band_inner, fft, private_projection
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.relative_carrier import match_packet_power, mix
from experiments.pamf_lora_main.runner import activate


def fidelity(wave, expected, rms, masks):
    found, desired = (private_projection(fft(x)) for x in (wave, expected))
    weight = rms.square()
    energy = band_inner(desired, desired, weight, masks)
    difference = found - desired
    squared = band_inner(difference, difference, weight, masks).sum(0)
    cross = torch.complex(band_inner(found, desired, weight, masks),
                          band_inner(found, 1j * desired, weight, masks))
    result = []
    for i, summed in enumerate(squared):
        valid = energy[:,i] > 1e-12
        gain = cross[valid,i] / energy[valid,i]
        result.append(dict(error=float(summed/energy[:,i].sum()), gain=float(gain.real.median()),
                           phase=float(gain.angle().abs().median())))
    return result


@torch.no_grad()
def main():
    start = perf_counter()
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = (Path(rt[key]) for key in ('run_root', 'source_root'))
    if root.name != RUN or read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed source projection experiment')
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    fit, calibrate, held = partition(rx)
    chosen = np.concatenate([a[:30] for a in np.split(held, len(SOURCE))])
    if np.intersect1d(chosen,np.concatenate((fit,calibrate))).size:
        raise ValueError('Held diagnostic shares a source fit or calibration packet')
    arms = {name:load_arm(root, recipe, contract, 'cpu', dict.fromkeys(RECIPES, WIDTH_RUN)) for name,recipe in RECIPES.items()}
    threshold = arms['compact'][1]
    if arms['wide'][1] != threshold:
        raise ValueError('Different clean gate')
    tracker = PacketRestorer('consensus').eval()
    clean_iq = tensor(np.stack([chirp_windows(raw[i]) for i in chosen]))
    separate = observed_frame(tracker, clean_iq)[0]
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            results[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs = ((route,view,raw[i],rx[i],ids[i]) for i in chosen)
                parts = zip(*pool.map(replay,jobs,chunksize=6),strict=True)
                iq, exact_noise = (tensor(np.stack(values)) for values in parts)
                local = coordinates(tracker, iq)
                active = corruption_score(local[0]) > threshold
                y, rotation, q, rms = packet_frame(local)
                waves = {'direct':y}
                residuals = {}
                native = real_iq(y * rotation).flatten(0,1)
                observed = torch.complex(native[:,0],native[:,1])
                for name,(model,_) in arms.items():
                    stages, _, action, coarse = model.restorer.stage_outputs(native)
                    residuals[name] = []
                    for stage,pair in zip(('coarse','stage1','stage2'), (coarse,*stages), strict=True):
                        pred = torch.complex(pair[:,0],pair[:,1])
                        residual = (observed-action(pred)).abs().square().unflatten(0,y.shape[:2]).mean((1,2))
                        amplitude = pred.abs().square().unflatten(0,y.shape[:2]).mean((1,2)).sqrt()
                        residuals[name].append(dict(stage=stage,fitted_action_residual=float(residual[active].median()),
                                                       native_rms=float(amplitude[active].median())))
                        frame = complex_iq(pair.unflatten(0,y.shape[:2])) * rotation.conj()
                        physical = match_packet_power(frame,y)
                        waves[name+'_'+stage] = torch.where(active[:,None,None],physical,y)
                    final = waves[name+'_stage2']
                    waves[name+'_half'] = mix(y,final,.5)
                    waves[name+'_quarter'] = mix(y,final,.25)
                noiseless = complex_iq(iq-exact_noise)/(rotation*rms)
                masks = named_band_masks(y).float()
                values = {}
                for k,wave in waves.items():
                    rms_edit = (wave-y).abs().square().mean((1,2)).sqrt()
                    values[k] = dict(independent=fidelity(wave,separate,rms,masks),
                                     same_channel=fidelity(wave,noiseless,rms,masks),
                                     active_relative_edit_rms_median=float(rms_edit[active].median()))
                results[route][view] = dict(n=len(chosen), active=int(active.sum()), readouts=values, stage_diagnostics=residuals)
                print('LORA_DURA_STAGES_U',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='VERIFIED', results=results, F_reads=0, TX_reads=0, target_reads=0,
                          V_reads=0, exact_noise_for_diagnostics_only=True, seconds=perf_counter()-start),allow_nan=False))


if __name__ == '__main__':
    main()
