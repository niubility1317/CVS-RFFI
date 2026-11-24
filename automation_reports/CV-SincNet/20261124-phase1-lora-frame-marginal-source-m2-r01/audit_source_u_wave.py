"""Quantify synchronous uncertainty and original-frame recovery on held source U in actual received coordinates."""

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
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.lag_profile import lag_profile_risk
from experiments.pamf_lora_main.frame_marginal import marginal_readouts
from experiments.pamf_lora_main.frame_marginal_source import MODES, RECIPES, bind
from experiments.pamf_lora_main.fission_risk import band_masks
from experiments.pamf_lora_main.private_dose import band_energy
from experiments.pamf_lora_main.gated_slope_source import restore
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import band_inner, fft, private_projection
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate



def analyze_bands(actual, expected, weight, masks):
    labels = ('1_to_8', '9_to_32', '33_to_64', '65_to_128')
    energy = band_inner(expected, expected, weight, masks)
    error = band_inner(actual-expected, actual-expected, weight, masks)
    overlap = torch.complex(band_inner(actual, expected, weight, masks),
                            band_inner(actual, 1j*expected, weight, masks))
    by_band = {}
    for j, band in enumerate(labels):
        valid = energy[:, j]>1e-12
        complex_gain = overlap[valid, j] / energy[valid, j]
        by_band[band] = dict(relative_error=float(error[:, j].sum()/energy[:, j].sum()),
                             median_real_gain=float(complex_gain.real.median()),
                             median_abs_phase_rad=float(complex_gain.angle().abs().median()))
    return by_band


def components(x):
    return dict(common=x.mean(1,keepdim=True).expand_as(x), private=private_projection(x))


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
    models, threshold = bind(root, contract, 'cpu')
    parent, model = models
    recipe = RECIPES['dose_bound']
    report = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            report[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs = ((route,view,raw[idx],rx[idx],ids[idx]) for idx in held)
                iq, epsilon = (tensor(np.stack(v)) for v in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                axes, scales = [], []
                active_count = 0
                names = ('direct', *MODES[2:])
                spreads, frames, active_masks, dose_error = [], [], [], []
                variants = {n: [] for n in names}
                for batch in iq.split(12):
                    frame = coordinates(tracker,batch)
                    y, rot, _, rms = frame
                    eligible = corruption_score(y)>threshold
                    active_count += int(eligible.sum())
                    chosen = tuple(a[eligible] for a in frame)
                    active_masks.append(eligible)
                    frames.append(rot)
                    out = [batch] * (len(names)-1)
                    if eligible.any():
                        out[0] = restore(batch, eligible, parent, chosen, (.5,))[0]
                        observed, rotation, quality, selected_rms = chosen
                        results, uncertainty = marginal_readouts(
                            observed, rotation, quality, selected_rms, model, recipe)
                        spreads.append(torch.cat(uncertainty, dim=-1).abs())
                        for j, delta in enumerate(results, 1):
                            out[j] = batch.clone()
                            out[j][eligible] += .5 * real_iq(delta * rotation * selected_rms)
                        bands = band_masks(observed).float()
                        energy = [band_energy(fft(d), selected_rms, bands) for d in (results[0], results[-1])]
                        dose_error.append((energy[0]-energy[1]).abs()/energy[0].clamp_min(1e-10))
                    axes.append(rot*rms)
                    scales.append(rms.square())
                    for name,values in zip(names,(batch,*out),strict=True):
                        variants[name].append(fft(complex_iq(values)/axes[-1]))
                actual = fft(complex_iq(iq-epsilon)/torch.cat(axes))
                outside = torch.fft.fftfreq(actual.shape[-1]).abs() > .25
                masks = torch.cat((named_band_masks(actual),outside[None])).float()
                reference = components(actual)
                weight = torch.cat(scales)
                clean_reference = torch.fft.ifft(actual, norm='ortho')
                lag_risks = {}
                for name, values in variants.items():
                    estimate = torch.fft.ifft(torch.cat(values), norm='ortho')
                    groups = zip(estimate.split(30), clean_reference.split(30), strict=True)
                    scores = [lag_profile_risk(y[:, None], target) for y, target in groups]
                    lag_risks[name] = torch.stack(scores).mean(0).tolist()
                # Compare the observed proxy to a tracking reference only after outputs exist.
                active = torch.cat(active_masks)
                expected_rotation = torch.cat([coordinates(tracker, batch)[1] for batch in (iq-epsilon).split(12)])
                relative = expected_rotation[active] * torch.cat(frames)[active].conj()
                center = relative[..., relative.shape[-1]//2].angle()
                slope = (relative[...,1:] * relative[...,:-1].conj()).mean(-1).angle() * ((relative.shape[-1]-1)/2)
                reference_error = torch.stack((center, slope), dim=-1).abs().flatten(0,1)
                received_spread = torch.cat(spreads).flatten(0,1)
                coverage = (reference_error<=2**.5*received_spread).float().mean(0)
                correlation = torch.stack([torch.corrcoef(torch.stack((a,b)))[0,1]
                                           for a,b in zip(received_spread.T,reference_error.T,strict=True)])
                report[route][view] = dict(n=len(held), active=active_count, lag_risks=lag_risks,
                    max_same_norm_relative_error=float(torch.cat(dose_error).max()),
                    uncertainty=dict(proxy_median=received_spread.median(0).values.tolist(),
                    isolated_median=reference_error.median(0).values.tolist(),
                    fraction_inside=coverage.tolist(), pearson=correlation.tolist()), waves={
                    name:{k:analyze_bands(v,reference[k],weight,masks)
                          for k,v in components(torch.cat(spectra)).items()}
                    for name,spectra in variants.items()})
                print('LORA_FRAME_MARGINAL_WAVE_AUDIT',route,view,active_count,flush=True)
    print(json.dumps(dict(status='PASS',n=len(held),reference_after_observed=True,seconds=perf_counter()-started,
          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,rows=report)),flush=True)


if __name__ == '__main__':
    main()
