"""Check fitted DURA receive-side action and variance on held source crops."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.carrier_coupled import observed_frame, packet_frame
from experiments.pamf_lora_main.channel_blind import corruption_score, zero_carrier
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.dura_stage_step_target import DOSE, RUN, bind
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.relative_carrier import match_packet_power, mix
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms


def power(value):
    return value.abs().square().mean((1,2))


def association(left, right):
    a, b = (x - x.mean() for x in (left,right))
    denom = a.norm()*b.norm()
    return float((a*b).sum()/denom) if denom > 0 else None


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = (Path(rt[key]) for key in ('run_root', 'source_root'))
    if root.name != RUN or read(root/'smoke.json')['status'] != 'PASS':
        raise ValueError('Expected the held source checkpoint check')
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    train, cal, held = partition(rx)
    chosen = np.concatenate([a[:30] for a in np.split(held,len(SOURCE))])
    if np.intersect1d(chosen,np.concatenate((train,cal))).size:
        raise ValueError('Diagnostic samples must remain held')
    model, threshold = bind(root,contract,'cpu')
    tracker = PacketRestorer('consensus').eval()
    iq_clean = tensor(np.stack([chirp_windows(raw[i]) for i in chosen]))
    own_clean = observed_frame(tracker,iq_clean)[0]
    clean = complex_iq(iq_clean)
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            results[route] = {}
            for view in (VIEWS[3], VIEWS[-1]):
                jobs = ((route,view,raw[i],rx[i],ids[i]) for i in chosen)
                iq, noise = (tensor(np.stack(x)) for x in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                frame = coordinates(tracker,iq)
                active = corruption_score(frame[0]) > threshold
                y, rotation, _, scale = packet_frame(frame)
                native = real_iq(y*rotation).flatten(0,1)
                stages, variance, action, _ = model.restorer.stage_outputs(native)
                variance = variance.reshape_as(y.real)
                first, second = (mix(y,match_packet_power(complex_iq(s.unflatten(0,y.shape[:2]))*rotation.conj(),y),DOSE)
                                 for s in stages)

                def observe(wave):
                    return action(wave.flatten(0,1)).reshape_as(y)

                weight = variance.amin((1,2),keepdim=True)/variance
                raw_y = y*rotation
                residuals = [zero_carrier((raw_y-observe(w*rotation))*rotation.conj()) for w in (first,second)]
                improvement = ((residuals[0].abs().square()-residuals[1].abs().square())*weight).mean((1,2))

                # Exact replay components are evaluation references below.
                epsilon = complex_iq(noise)/scale
                noiseless = complex_iq(iq-noise)/scale
                exp_noise = 2*variance.mean((1,2))
                measured_noise = power(epsilon)
                offband_noise = observed_noise_rms(y).square().mean((1,2))
                projected_clean = observe(match_packet_power(clean,raw_y))
                channel_error = power(projected_clean-noiseless)
                mask = named_band_masks(y)[0]

                def private_near(wave):
                    residual = wave-wave.mean(1,keepdim=True)
                    return torch.fft.fft(residual,norm='ortho')[...,mask]

                near_a, near_b = (private_near(x) for x in (first,second))
                expected = private_near(own_clean)
                same = private_near(noiseless*rotation.conj())
                clean_improvement = power(near_a-expected)-power(near_b-expected)
                noiseless_improvement = power(near_a-same)-power(near_b-same)
                def agreement(reference):
                    vector = reference[active]
                    estimate = improvement[active]
                    return dict(correlation=association(estimate,vector),
                                agree=int(((estimate>0)==(vector>0)).sum()),
                                favourable=int((vector>0).sum()))
                results[route][view] = dict(active=int(active.sum()),
                    favourable_estimated=int((improvement[active]>0).sum()),
                    estimated_vs_clean_private=agreement(clean_improvement),
                    estimated_vs_noiseless_private=agreement(noiseless_improvement),
                    offband_log_correlation=association(offband_noise[active].log(),measured_noise[active].log()),
                    offband_to_noise_median=float((offband_noise[active]/measured_noise[active]).median()),
                    variance_log_correlation=association(exp_noise[active].log(),measured_noise[active].log()),
                    variance_to_noise_median=float((exp_noise[active]/measured_noise[active]).median()),
                    forward_model_error_to_noise_median=float((channel_error[active]/measured_noise[active]).median()))
                print('LORA_DURA_ACTION_CALIBRATION_U',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='VERIFIED',n=len(chosen),results=results,F_reads=0,V_reads=0,TX_reads=0,
        target_reads=0,exact_trace_for_diagnostics_only=True),allow_nan=False))


if __name__ == '__main__':
    main()
