"""Audit frozen LoRa recovery against paired source waveforms and traces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from math import log
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main import lapped_posterior_source, local_phase_source, spectral_block_source, split_shrink_source
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.lapped_posterior import analysis as lapped_analysis
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.split_shrink import spectrum as split_spectrum


BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))
METRICS = ('complex_nmse', 'magnitude_nmse', 'normalized_real_correlation')
GEOMETRY_RUN = '20261124-phase1-lora-complex-geometry-source-m4-r01'
TANGENT_RUN = '20261124-phase1-lora-empirical-tangent-source-m4-r01'


def bounded_projection(model, direct, reference, y, frame, threshold):
    """L2 projection onto the candidate's exact noncarrier editing disks."""
    x, _, _, level, _ = model.observe(y)
    target = model.spectral(reference/frame)
    bound = .60*(x.abs()+.15*level.clamp_min(1e-8).sqrt())
    correction = model.bounded(target-x,bound)
    output = direct+model.temporal(correction)*frame
    use = corruption_score(y)>threshold
    return torch.where(use[:,None,None],output,direct)


def spectral_metrics(waves, known, frame):
    spectrum = torch.fft.fft(waves/frame,norm='ortho')
    truth = torch.fft.fft(known/frame,norm='ortho')
    bins = torch.arange(256,device=waves.device)
    bins = torch.minimum(bins,256-bins)
    bands = []
    for low,high in BANDS:
        use = (bins>=low)&(bins<=high)
        received, reference = spectrum[...,use],truth[...,use]
        power = reference.abs().square().sum((-2,-1)).clamp_min(1e-6)
        error = (received-reference).abs().square().sum((-2,-1))/power
        magnitude = (received.abs()-reference.abs()).square().sum((-2,-1))/power
        product = (received*reference.conj()).sum((-2,-1)).real
        coherence = product/(power*received.abs().square().sum((-2,-1)).clamp_min(1e-6)).sqrt()
        bands.append(torch.stack((error,magnitude,coherence),dim=-1))
    return np.asarray(torch.stack(bands,dim=2).tolist(),dtype=np.float32)


def direction_fraction(model, tracked, frame, waves):
    direct, *predictions = waves
    edits = torch.fft.fft((torch.stack(predictions)-direct)/frame,norm='ortho')
    actual = torch.fft.fft(tracked,norm='ortho')
    power = actual[...,model.outer].abs().square().mean(-1,keepdim=True)
    numerator, denominator = [], []
    for name, support, destination in (('even',slice(0,None,2),slice(1,None,2)),
                                        ('odd',slice(1,None,2),slice(0,None,2))):
        _, _, atoms, dual = model.query(actual[:,support],power[:,support],name,destination)
        move = edits[:,:,destination][...,model.weak].flatten(2).unsqueeze(-1)
        projection = atoms@(dual@move)
        numerator.append(projection.abs().square().sum((-1,-2)))
        denominator.append(move.abs().square().sum((-1,-2)))
    ratio = (numerator[0]+numerator[1])/(denominator[0]+denominator[1]).clamp_min(1e-10)
    dose = edits.abs().square().mean((-1,-2)).sqrt()
    return np.asarray(torch.stack((ratio,dose),dim=-1).tolist(),dtype=np.float32)


def outer_noise_error(outer, tracked, frame, epsilon, clean):
    observed = split_spectrum(tracked)[..., outer]
    actual = split_spectrum(complex_iq(epsilon) / frame)[..., outer]
    estimate = observed.abs().square().flatten(1).median(dim=1).values / log(2.)
    reference = actual.abs().square().mean(dim=(-1,-2))
    if clean:
        return estimate.cpu().numpy()[:, None]
    error_db = 10. * (estimate.clamp_min(1e-12) / reference.clamp_min(1e-12)).log10()
    return torch.stack((estimate, reference, error_db, error_db.abs()), dim=1).cpu().numpy()


def split_shrink_doses(waves, tracked, frame):
    received = split_spectrum(tracked)
    mean = received.mean(dim=1, keepdim=True)
    common = mean.expand_as(received)
    unique = received - common
    edits = split_spectrum((torch.stack(waves[1:]) - waves[0]) / frame)
    bin_number = torch.fft.fftshift(torch.fft.fftfreq(tracked.shape[-1], device=tracked.device)).abs() * tracked.shape[-1]
    bands = []
    for low, high in BANDS:
        selected = (bin_number >= low) & (bin_number <= high)
        component_gains = []
        for component in (common, unique):
            basis = component[..., selected]
            energy = basis.abs().square().sum((-1,-2)).clamp_min(1e-10)
            overlap = (basis.conj() * edits[..., selected]).sum((-1,-2)).real
            component_gains.append(-overlap / energy)
        bands.append(torch.stack(component_gains, dim=-1))
    return torch.stack(bands, dim=2).cpu().numpy()


LAPPED_VARIANCE_FIELDS = ('local_power', 'pooled_power', 'prior_variance', 'prior_normalized_error',
                          'local_log_bias_db', 'pooled_log_bias_db', 'local_abs_log_error_db',
                          'pooled_abs_log_error_db', 'local_temporal_rmse_db', 'pooled_temporal_rmse_db')


def lapped_variance(model, tracked, quality, frame, source, epsilon, *, clean):
    coefficients = lapped_analysis(tracked, model.window)
    measured = model.estimate_power(coefficients)
    pooled = measured.mean(dim=-2, keepdim=True).expand_as(measured)
    local = .75 * measured + .25 * pooled
    prior, variance = model.restore(tracked, quality)[2:]
    error = (lapped_analysis(source / frame, model.window) - prior).abs().square()
    noncarrier = model.noncarrier
    normalize = (error / variance)[..., noncarrier].flatten(1).mean(-1)
    results = [local.flatten(1).mean(-1), pooled.flatten(1).mean(-1),
               variance[..., noncarrier].flatten(1).mean(-1), normalize]
    if not clean:
        actual_spec = lapped_analysis(complex_iq(epsilon) / frame, model.window)
        actual = actual_spec[..., model.outer].abs().square().mean(-1, keepdim=True)
        errors = 10. * (torch.stack((local, pooled)).clamp_min(1e-10).log10() - actual.clamp_min(1e-10).log10())
        results.extend(errors.flatten(2).mean(-1).unbind(0))
        results.extend(errors.abs().flatten(2).mean(-1).unbind(0))
        variation = errors - errors.mean(dim=-2, keepdim=True)
        results.extend(variation.square().flatten(2).mean(-1).sqrt().unbind(0))
    return torch.stack(results, dim=-1).cpu().numpy()


def paired_source(data, cache, contract, root, route, modes):
    raw, receivers, ids, _ = source_role(data,contract,'V')
    selected = panel(receivers)
    frozen_ids, _, marks = read_frozen(root,route,modes)
    if ids[selected].tolist()!=frozen_ids:
        raise ValueError('The diagnostic must follow the same frozen source V physical IDs')
    rows = list(zip(raw[selected],receivers[selected],ids[selected],strict=True))
    registered = np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    clean = registered[0,selected]
    return rows, registered, clean, selected, marks


@torch.no_grad()
def diagnose(rt, route, device):
    is_geometry = rt['run_id']==GEOMETRY_RUN
    is_tangent = rt['run_id']==TANGENT_RUN
    is_split = rt['run_id']==split_shrink_source.RUN
    is_lapped = rt['run_id']==lapped_posterior_source.RUN
    if is_geometry or is_tangent or is_split or is_lapped:
        if is_tangent:
            from experiments.pamf_lora_main import empirical_tangent_source as study
            recipes = study.recipes(rt)
        elif is_split:
            study = split_shrink_source
            recipes = study.RECIPES
        elif is_lapped:
            study = lapped_posterior_source
            recipes = study.RECIPES
        else:
            from experiments.pamf_lora_main import complex_geometry_source as study
            recipes = study.RECIPES
        recover = spectral_block_source.recover
        readouts = ('direct',*study.MODES[2:])
        indices = range(len(study.MODES)-2)
    else:
        study = local_phase_source
        recipes = study.RECIPES
        recover = study.recover
        readouts = ('direct','middle_full','edges_full','edges_half')
        indices = (1,4,5)
    root = Path(rt['run_root'])
    data = Path(rt['source_root'])
    cache = root.parent/DATA_RUN/'source_views'
    contract = source_contract(data)
    rows, registered, clean, selected, marks = paired_source(data,cache,contract,root,route,study.MODES)
    models, threshold = spectral_block_source.load_models(root,contract,device,recipes)
    if is_split:
        models = (*models, split_shrink_source.ScalarReadout(models[-1]).eval())
    elif is_lapped:
        models = (*models, lapped_posterior_source.PooledReadout(models[-1]).eval())
    tracker = PacketRestorer('consensus').to(device).eval()
    truth = tensor(clean,device)
    output = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, view in enumerate(VIEWS):
            capture, noise = (np.stack(v) for v in zip(*pool.map(partial(replay,route,view),rows,chunksize=8),strict=True))
            if route=='full' and not np.array_equal(capture,registered[i,selected]):
                raise ValueError('Full source trace replay changed registered IQ')
            if route=='residual':
                for j in range(0,len(rows),len(rows)//len(SOURCE)):
                    if not np.array_equal(capture[j],receive_only(route,view,rows[j])[0]):
                        raise ValueError('Residual source trace replay changed sampled registered IQ')
            active, bands_from_ground, bands_from_noiseless = 0, [], []
            direction = []
            noise_stats = []
            attenuation = []
            for received, epsilon, source in zip(tensor(capture,device).split(18),tensor(noise,device).split(18),truth.split(18),strict=True):
                recoveries, count = recover(received,tracker,models,threshold)[:2]
                active += count
                y = complex_iq(received)
                waves = [y,*(complex_iq(recoveries[k]) for k in indices)]
                tracked, phase, quality, rms = coordinates(tracker,received)
                frame = phase*rms
                if is_tangent:
                    direction.append(direction_fraction(models[-1],tracked,frame,waves))
                if is_split:
                    noise_stats.append(outer_noise_error(models[1].outer,tracked,frame,epsilon,view=='clean'))
                    attenuation.append(split_shrink_doses(waves,tracked,frame))
                # One fit against the direct readout is shared by all diagnostic comparisons.
                ground = complex_iq(source)
                gain = (y*ground.conj()).mean(-1,keepdim=True)/ground.abs().square().mean(-1,keepdim=True).clamp_min(1e-6)
                target = ground*gain
                noiseless = y-complex_iq(epsilon)
                if is_lapped:
                    noise_stats.append(lapped_variance(models[1],tracked,quality,frame,ground,epsilon,clean=view=='clean'))
                for reference, group in ((target,bands_from_ground),(noiseless,bands_from_noiseless)):
                    compared = waves
                    if is_geometry:
                        projected = bounded_projection(models[-1],y,reference,tracked,frame,threshold)
                        compared = [*waves,projected]
                    group.append(spectral_metrics(torch.stack(compared),reference,frame))
            if active!=marks[view]['active']:
                raise ValueError('Diagnostic gate triggers differ from frozen predictions')
            ground, noiseless = [np.concatenate(parts,axis=1) for parts in (bands_from_ground,bands_from_noiseless)]
            output[view] = {'active':active}
            for key, values in (('aligned_source',ground),('noiseless_reception',noiseless)):
                output[view][key] = dict(median=np.median(values,axis=1).tolist(),
                    by_receiver={name:np.median(group,axis=1).tolist()
                                 for name,group in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
            if is_tangent:
                values = np.concatenate(direction,axis=1)
                output[view]['direction'] = dict(readouts=readouts[1:],fields=['projected_weak_fraction','rms_dose'],
                    median=np.median(values,axis=1).tolist(),
                    by_receiver={name:np.median(group,axis=1).tolist()
                                 for name,group in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
            if is_split:
                values = np.concatenate(noise_stats)
                fields = ['estimate'] if view=='clean' else ['estimate','actual','estimate_minus_actual_db','absolute_error_db']
                output[view]['outer_noise'] = dict(fields=fields, median=np.median(values,axis=0).tolist(),
                    by_receiver={name:np.median(part,axis=0).tolist()
                                 for name,part in zip(SOURCE,np.split(values,len(SOURCE)),strict=True)})
                gains = np.concatenate(attenuation,axis=1)
                output[view]['attenuation'] = dict(readouts=readouts[1:],fields=['common_gain','unique_gain'],
                    median=np.median(gains,axis=1).tolist(),
                    by_receiver={name:np.median(part,axis=1).tolist()
                                 for name,part in zip(SOURCE,np.split(gains,len(SOURCE),axis=1),strict=True)})
            if is_lapped:
                values = np.concatenate(noise_stats)
                output[view]['lapped_variance'] = dict(fields=LAPPED_VARIANCE_FIELDS[:values.shape[1]],
                    median=np.median(values,axis=0).tolist(),
                    by_receiver={name:np.median(part,axis=0).tolist()
                                 for name,part in zip(SOURCE,np.split(values,len(SOURCE)),strict=True)})
    if is_geometry:
        readouts = (*readouts,'bounded_pair_projection')
    return dict(route=route,n=len(rows),bands=BANDS,metrics=METRICS,readouts=readouts,scenes=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('route',choices=ROUTES)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    rt = read(args.runtime)
    root = Path(rt['run_root'])
    analyzed_runs = (local_phase_source.RUN, GEOMETRY_RUN, TANGENT_RUN, split_shrink_source.RUN,
                     lapped_posterior_source.RUN)
    if rt.get('run_id') not in analyzed_runs or read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('The full dual-route truth-last scoring must already be complete')
    activate(rt)
    print(json.dumps(diagnose(rt,args.route,args.device),allow_nan=False))


if __name__=='__main__':
    main()
