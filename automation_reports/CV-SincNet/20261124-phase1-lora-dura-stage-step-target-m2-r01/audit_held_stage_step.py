"""Measure DURA stage-step usage on the original held source U panel."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.carrier_coupled import observed_frame, packet_frame
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_stage_step_target import MODES, RUN, bind, stage_reads
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.private_band_risk_source import band_inner, fft, private_projection
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
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
    if root.name != RUN or read(root/'smoke.json')['status'] != 'PASS':
        raise ValueError('Expected the real target-run source checkpoint check')
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    fit, calibrate, held = partition(rx)
    chosen = np.concatenate([a[:30] for a in np.split(held, len(SOURCE))])
    if np.intersect1d(chosen,np.concatenate((fit,calibrate))).size:
        raise ValueError('Held diagnostic shares a source fit or calibration packet')
    model, threshold = bind(root, contract, 'cpu')
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
                proposed, solved = stage_reads(y,rotation,model)
                waves = {'direct':y}
                for k, value in zip(MODES,proposed,strict=True):
                    waves[k] = torch.where(active[:,None,None],value,y)
                power = y.abs().square().sum((1,2))
                noiseless = complex_iq(iq-exact_noise)/(rotation*rms)
                masks = named_band_masks(y).float()
                values = {}
                for k,wave in waves.items():
                    rms_edit = (wave-y).abs().square().mean((1,2)).sqrt()
                    values[k] = dict(independent=fidelity(wave,separate,rms,masks),
                                     same_channel=fidelity(wave,noiseless,rms,masks),
                                     active_relative_edit_rms_median=float(rms_edit[active].median()),
                                     max_packet_power_drift=float((wave.abs().square().sum((1,2))/power-1).abs().max()))
                diagnostics = {}
                for mode, (alpha, before, after) in zip(MODES[-2:], solved, strict=True):
                    t, r0, r1 = (x[active].flatten() for x in (alpha,before,after))
                    if (r1 > r0).any() or not t.isfinite().all():
                        raise ValueError('Emitted source readout did not satisfy the receive constraint')
                    diagnostics[mode] = dict(accepted=int((t>0).sum()), full_step=int((t==1).sum()),
                        median_step=float(t.median()), step_p90=float(t.quantile(.90)),
                        median_fraction_reduction=float(((r0-r1)/r0.clamp_min(1e-10)).median()))
                results[route][view] = dict(n=len(chosen), active=int(active.sum()), readouts=values,
                                            step=diagnostics)
                print('LORA_DURA_STAGE_STEP_U',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='VERIFIED', results=results, F_reads=0, TX_reads=0, target_reads=0,
                          V_reads=0, exact_noise_for_diagnostics_only=True, seconds=perf_counter()-start),allow_nan=False))


if __name__ == '__main__':
    main()
