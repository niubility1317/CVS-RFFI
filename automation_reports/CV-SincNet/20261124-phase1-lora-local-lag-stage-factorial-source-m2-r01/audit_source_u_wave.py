"""Quantify the two frozen coarse and four crossed half-recovery estimates on held source U in actual received coordinates."""

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
from experiments.pamf_lora_main.stage_splice import read_stages
from experiments.pamf_lora_main.stage_splice_source import MODES, bind
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
    parent, *arms = models
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
                edits = {key: [] for key in MODES[-4:]}
                variants = {n: [] for n in names}
                for batch in iq.split(12):
                    frame = coordinates(tracker,batch)
                    y, rot, _, rms = frame
                    eligible = corruption_score(y)>threshold
                    active_count += int(eligible.sum())
                    chosen = tuple(a[eligible] for a in frame)
                    out = [batch] * (len(names)-1)
                    if eligible.any():
                        out[0] = restore(batch, eligible, parent, chosen, (.5,))[0]
                        observed, rotation, quality, selected_rms = chosen
                        stages = read_stages(observed, rotation, quality, arms)
                        for j, wave in enumerate(stages, 1):
                            out[j] = batch.clone()
                            out[j][eligible] += real_iq((wave - observed) * rotation * selected_rms)
                        norm = lambda x: torch.linalg.vector_norm(x * selected_rms, dim=(-2,-1))
                        for key, fine, coarse in zip(MODES[-4:], stages[2:],
                                                     (stages[0], stages[1], stages[0], stages[1]), strict=True):
                            coarse_half = torch.lerp(observed, coarse, .5)
                            edits[key].append(norm(fine-coarse_half) / norm(coarse_half-observed).clamp_min(1e-8))
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
                report[route][view] = dict(n=len(held),active=active_count, lag_risks=lag_risks,
                    fine_edit_ratio={key:float(torch.cat(parts).median()) for key,parts in edits.items()}, waves={
                    name:{k:analyze_bands(v,reference[k],weight,masks)
                          for k,v in components(torch.cat(spectra)).items()}
                    for name,spectra in variants.items()})
                print('LORA_STAGE_SPLICE_WAVE_AUDIT',route,view,active_count,flush=True)
    print(json.dumps(dict(status='PASS',n=len(held),reference_after_observed=True,seconds=perf_counter()-started,
          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,rows=report)),flush=True)


if __name__ == '__main__':
    main()
