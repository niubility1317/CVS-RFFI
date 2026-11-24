"""Compare received pilot estimates to an isolated noiseless pilot projection."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.crossfit_pilot import estimate_shared
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.pilot_after_denoise_source import RECIPES
from experiments.pamf_lora_main.private_band_risk_source import band_inner, fft, private_projection
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms


def band_fidelity(actual, expected, weight, masks):
    expected = private_projection(fft(expected))
    actual = private_projection(fft(actual))
    energy = band_inner(expected, expected, weight, masks)
    error = band_inner(actual-expected, actual-expected, weight, masks)
    overlap = torch.complex(band_inner(actual, expected, weight, masks),
                            band_inner(actual, 1j*expected, weight, masks))
    result = []
    for j in range(energy.shape[-1]):
        valid = energy[:, j] > 1e-12
        gain = overlap[valid, j] / energy[valid, j]
        result.append(dict(error=float(error[:, j].sum()/energy[:, j].sum()),
                           gain=float(gain.real.median()), abs_phase=float(gain.angle().abs().median())))
    return result


@torch.no_grad()
def audit():
    started = perf_counter()
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    training, calibration, hold = partition(rx)
    held = np.concatenate([part[:30] for part in np.split(hold, len(SOURCE))])
    if np.intersect1d(np.concatenate((training,calibration)), held).size:
        raise ValueError('Expected an independent source U waveform panel')
    arms = {name: load_model(root, recipe, contract, 'cpu') for name,recipe in RECIPES.items()}
    threshold = arms['raw_pilot'][1]
    if arms['denoised_pilot'][1] != threshold:
        raise ValueError('The source U bypass thresholds differ')
    tracker = PacketRestorer('consensus').eval()
    clean = complex_iq(tensor(np.stack([chirp_windows(raw[idx]) for idx in held])))
    result = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            result[route] = {}
            for scene in (VIEWS[3], VIEWS[-1]):
                jobs = ((route,scene,raw[idx],rx[idx],ids[idx]) for idx in held)
                capture, noise = (tensor(np.stack(v)) for v in zip(*pool.map(replay,jobs,chunksize=6), strict=True))
                measured, rotation, quality, rms = coordinates(tracker, capture)
                active = corruption_score(measured) > threshold
                base = rotation*rms
                rows, pending = {}, {}
                for name, (model, _) in arms.items():
                    coarse, prior, variance = model.pilot_groups(measured,quality)
                    argument = coarse if model.denoised_fit else None
                    transfer, support = estimate_shared(measured,rotation,quality,variance,fit_observed=argument)
                    fit_response = model.calibration(transfer,support)
                    first, fine = model(measured,quality,rotation)
                    _, application = RECIPES[name].transform_stages(measured,(first,fine),rms)
                    coarse = torch.where(active[:,None,None],coarse,measured)
                    half = torch.where(active[:,None,None],torch.lerp(measured,application,.5),measured)
                    pending[name] = (fit_response, support, coarse, half)
                noiseless = complex_iq(capture-noise)/base
                observed_variance = observed_noise_rms(measured).square()
                isolated, isolated_support = estimate_shared(noiseless,rotation,quality,observed_variance)
                desired = arms['raw_pilot'][0].calibration(isolated,isolated_support)
                aligned_clean = clean/base
                masks = named_band_masks(measured).float()
                for name, (response, support, coarse, half) in pending.items():
                    region = torch.fft.fftfreq(measured.shape[-1]).abs() <= .25
                    normalized_difference = (response-desired)[active][...,region].abs().square().mean(-1)
                    rows[name] = dict(
                        mean_support=float(support[active].mean()),
                        response_rmse=float(normalized_difference.mean().sqrt()),
                        response_error_median=float(normalized_difference.sqrt().median()),
                        coarse_error=float(((coarse-noiseless)*base).abs().square().sum()
                                            /(noiseless*base).abs().square().sum()),
                        raw_error=float(((measured-noiseless)*base).abs().square().sum()
                                         /(noiseless*base).abs().square().sum()),
                        private_coarse=band_fidelity(coarse,noiseless,rms.square(),masks),
                        private_half=band_fidelity(half,aligned_clean,rms.square(),masks))
                result[route][scene] = dict(n=len(held), active=int(active.sum()), modes=rows)
                print('LORA_PILOT_ISOLATED_REFERENCE',route,scene,int(active.sum()),flush=True)
    print(json.dumps(dict(status='PASS',n=len(held),F_reads=0,V_reads=0,TX_reads=0,target_reads=0,
          reference_after_frozen_output=True,seconds=perf_counter()-started,rows=result), allow_nan=False),flush=True)


if __name__ == '__main__':
    audit()
