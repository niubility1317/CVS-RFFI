"""Quantify the coupled-bootstrap estimator and its frozen endpoints on held source U in actual received coordinates."""

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
from experiments.pamf_lora_main.fission_risk import band_masks, edit_risk, mix_readouts, optimal_mix
from experiments.pamf_lora_main.fission_risk_source import bind
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


def independent_reference(linear, quadratic, global_dose, band_dose):
    values = []
    for l, q, dose in ((linear.sum(-1), quadratic.sum(-1), global_dose[:, 0]),
                       *((l, q, d) for l, q, d in zip(linear.T, quadratic.T, band_dose.T, strict=True))):
        valid = q > 1e-12
        l, q, dose = l[valid], q[valid], dose[valid]
        oracle = optimal_mix(l, q)
        loss = 2*dose*l + dose.square()*q
        midpoint = l + .25*q
        best = 2*oracle*l + oracle.square()*q
        values.append(dict(n=len(q), dose_median=float(dose.median()), ideal_median=float(oracle.median()),
                           dose_mae=float((dose-oracle).abs().mean()),
                           direction_agreement=float(((dose>.5)==(2*l+q<0)).float().mean()),
                           excess_over_midpoint=float((loss-midpoint).sum()/q.sum()),
                           excess_of_midpoint_over_best=float((midpoint-best).sum()/q.sum())))
    return dict(zip(('whole','1_to_8','9_to_32','33_to_128'), values, strict=True))


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
    report = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            report[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs = ((route,view,raw[idx],rx[idx],ids[idx]) for idx in held)
                iq, epsilon = (tensor(np.stack(v)) for v in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                axes, scales = [], []
                active_count = 0
                names = ('direct', 'rank_half', 'direction_half', 'fixed_mid', 'split_global', 'split_bands')
                global_weights, band_weights, masks_active = [], [], []
                variants = {n: [] for n in names}
                for batch in iq.split(12):
                    frame = coordinates(tracker,batch)
                    y, rot, _, rms = frame
                    eligible = corruption_score(y)>threshold
                    active_count += int(eligible.sum())
                    chosen = tuple(a[eligible] for a in frame)
                    out = [batch] * 5
                    masks_active.append(eligible)
                    if eligible.any():
                        observed, rotation, quality, selected_rms = chosen
                        edits, (global_dose, band_dose) = mix_readouts(*chosen, models)
                        global_weights.append(global_dose)
                        band_weights.append(band_dose)
                        for j, edit in enumerate(edits):
                            out[j] = batch.clone()
                            out[j][eligible] += real_iq(edit * rotation * selected_rms)
                    axes.append(rot*rms)
                    scales.append(rms.square())
                    for name,values in zip(names,(batch,*out),strict=True):
                        variants[name].append(fft(complex_iq(values)/axes[-1]))
                actual = fft(complex_iq(iq-epsilon)/torch.cat(axes))
                outside = torch.fft.fftfreq(actual.shape[-1]).abs() > .25
                masks = torch.cat((named_band_masks(actual),outside[None])).float()
                reference = components(actual)
                weight = torch.cat(scales)
                which = torch.cat(masks_active)
                original = torch.cat(variants['rank_half'])[which]
                alternative = torch.cat(variants['direction_half'])[which]
                direction_masks = band_masks(original).float()
                linear, quadratic = edit_risk(original[:, None], alternative[:, None], actual[which, None],
                                               weight[which], direction_masks)
                global_dose = torch.cat(global_weights)
                band_dose = torch.cat(band_weights)
                risk = independent_reference(linear, quadratic, global_dose, band_dose)
                report[route][view] = dict(n=len(held), active=active_count, risk=risk, waves={
                    name:{k:analyze_bands(v,reference[k],weight,masks)
                          for k,v in components(torch.cat(spectra)).items()}
                    for name,spectra in variants.items()})
                print('LORA_FISSION_RISK_WAVE_AUDIT',route,view,active_count,flush=True)
    print(json.dumps(dict(status='PASS',n=len(held),reference_after_observed=True,seconds=perf_counter()-started,
          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,rows=report)),flush=True)


if __name__ == '__main__':
    main()
