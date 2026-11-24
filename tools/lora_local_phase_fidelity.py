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
from experiments.pamf_lora_main import held_chirp_precision_source, lapped_posterior_source, local_phase_source
from experiments.pamf_lora_main import coherent_interval_source, joint_spectrum_source
from experiments.pamf_lora_main import packet_relation_source, reference_cancel_source, spectral_block_source, split_shrink_source
from experiments.pamf_lora_main import received_update_source
from experiments.pamf_lora_main import phase_dose_source
from experiments.pamf_lora_main import variance_transport_source
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.held_chirp_precision import held_eigenvalues
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.received_update import update_coordinates
from experiments.pamf_lora_main.joint_spectrum import split as spectrum_parts
from experiments.pamf_lora_main.lapped_posterior import other_mean
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen
from experiments.pamf_lora_main.packet_relation import fingerprint
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.split_shrink import spectrum as split_spectrum


BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))
METRICS = ('complex_nmse', 'magnitude_nmse', 'normalized_real_correlation')
GEOMETRY_RUN = '20261124-phase1-lora-complex-geometry-source-m4-r01'
TANGENT_RUN = '20261124-phase1-lora-empirical-tangent-source-m4-r01'
ANCHOR_RUN = '20261124-phase1-lora-common-anchor-source-m4-r01'
BURST_DOSE_RUN = '20261124-phase1-lora-phase-dose-bursts-source-m4-r01'
NEIGHBOR_PHASE_RUN = '20261124-phase1-lora-neighbor-phase-source-m4-r01'


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
    coefficients = model.transformed(tracked)
    measured = model.estimate_power(coefficients)
    pooled = measured.mean(dim=-2, keepdim=True).expand_as(measured)
    local = .75 * measured + .25 * pooled
    prior, variance = model.restore(tracked, quality)[2:]
    error = (model.transformed(source / frame) - prior).abs().square()
    noncarrier = model.noncarrier
    normalize = (error / variance)[..., noncarrier].flatten(1).mean(-1)
    results = [local.flatten(1).mean(-1), pooled.flatten(1).mean(-1),
               variance[..., noncarrier].flatten(1).mean(-1), normalize]
    if not clean:
        actual_spec = model.transformed(complex_iq(epsilon) / frame)
        actual = actual_spec[..., model.outer].abs().square().mean(-1, keepdim=True)
        errors = 10. * (torch.stack((local, pooled)).clamp_min(1e-10).log10() - actual.clamp_min(1e-10).log10())
        results.extend(errors.flatten(2).mean(-1).unbind(0))
        results.extend(errors.abs().flatten(2).mean(-1).unbind(0))
        variation = errors - errors.mean(dim=-2, keepdim=True)
        results.extend(variation.square().flatten(2).mean(-1).sqrt().unbind(0))
    return torch.stack(results, dim=-1).cpu().numpy()


HELD_RISK_FIELDS = ('learned_risk_ratio', 'isotropic_risk_ratio', 'directional_large_risk_ratio',
                    'directional_small_risk_ratio', 'weak_band_gain_contrast')


def held_risk(model, tracked, quality, frame, source):
    observed = model.transformed(tracked)
    support = other_mean(observed)
    dispersion = (other_mean(observed.abs().square()) - support.abs().square()).clamp_min(0.)
    measurement = model.estimate_power(observed)
    noise = .75 * measurement + .25 * measurement.mean(dim=-2, keepdim=True)
    support_noise = other_mean(noise).clamp_min(0.)
    large, small, axis = held_eigenvalues(observed, support, dispersion, support_noise)
    prior, variance = model.restore(tracked, quality)[2:]
    error = model.transformed(source / frame) - prior
    axis = torch.where(axis.abs() > .5, axis, 1.)
    major = .5 * (error + axis * error.conj())
    minor = error - major
    energy = error.abs().square()
    fields = (energy / variance, energy / (variance + .5 * (large + small)),
              2. * major.abs().square() / (variance + large),
              2. * minor.abs().square() / (variance + small))
    ratios = [v[...,model.noncarrier].flatten(1).mean(-1) for v in fields]
    weak = (model.frequency.abs() <= .125) & model.noncarrier
    contrast = (large - small) / (2 * (.02 + noise + variance) + large + small)
    ratios.append(contrast[...,weak].flatten(1).mean(-1))
    return torch.stack(ratios, dim=-1).cpu().numpy()


CANCEL_FIELDS = ('near_common_dose', 'middle_common_dose', 'near_private_dose', 'middle_private_dose',
                 'near_common_spectral_gain', 'middle_common_spectral_gain', 'near_private_spectral_gain',
                 'middle_private_spectral_gain', 'near_common_source_budget_coverage',
                 'middle_common_source_budget_coverage', 'near_private_source_budget_coverage',
                 'middle_private_source_budget_coverage', 'outer_power_bias_db')


def cancellation_evidence(model, tracked, frame, epsilon, clean):
    from experiments.pamf_lora_main.reference_cancel import pair_spectrum

    edit, doses = model.cancellation(tracked)
    correction = pair_spectrum(edit)
    known_noise = complex_iq(epsilon) / frame
    actual = pair_spectrum(known_noise)
    noiseless = pair_spectrum(tracked - known_noise)
    gains, coverage = [], []
    for name in ('near', 'middle'):
        bins = getattr(model, f'{name}_bins') % 256
        noise = actual[..., bins]
        cancel = correction[..., bins]
        power = noise.abs().square().sum((1, 3)).clamp_min(1e-8)
        excess = (noise + cancel).abs().square().sum((1, 3))
        gains.append((1 - excess / power).masked_fill(power < 1e-7, 0.))
        signal, _ = model.signed_residual(noiseless, name)
        within = signal.abs().square().mean(-1) <= getattr(model, f'{name}_budget')
        coverage.append(within.float().mean(1))
    values = [doses.mean(1).flatten(1), torch.stack(gains, dim=-1).flatten(1),
              torch.stack(coverage, dim=-1).flatten(1)]
    if not clean:
        estimate = model.estimate_power(model.transformed(tracked)).mean((1, 2, 3))
        true = model.transformed(known_noise)
        power = true[..., model.outer].abs().square().mean((1, 2, 3))
        values.append((10 * (estimate.clamp_min(1e-10)/power.clamp_min(1e-10)).log10()).unsqueeze(-1))
    return np.asarray(torch.cat(values, dim=1).tolist(), dtype=np.float32)


TRANSPORT_FIELDS = ('centered_power_ratio', 'signed_source_gain', 'centered_complex_correlation')


def transport_evidence(model, waves, source, frame):
    known = torch.view_as_complex(model.center)
    restored = model.analysis(torch.stack(waves).flatten(0, 1) / frame.repeat(len(waves), 1, 1))
    restored = restored[..., model.weak].unflatten(0, (len(waves), len(frame))) - known
    reference = model.analysis(source / frame)[..., model.weak] - known
    bin_number = torch.arange(256, device=frame.device)[model.weak]
    distance = torch.minimum(bin_number, 256 - bin_number)
    results = []
    for selected in (distance <= 8, distance > 8):
        value, truth = restored[..., selected], reference[..., selected]
        for mode in (slice(0, 1), slice(1, None)):
            a, b = value[..., mode, :], truth[..., mode, :]
            p = b.abs().square().sum((-1, -2)).clamp_min(1e-8)
            q = a.abs().square().sum((-1, -2)).clamp_min(1e-8)
            cross = (a * b.conj()).sum((-1, -2)).real
            results.append(torch.stack((q / p, cross / p, cross / (p * q).sqrt()), dim=-1))
    return np.asarray(torch.stack(results, dim=-2).tolist(), dtype=np.float32)


def transport_noise_bias(model, tracked, frame, epsilon):
    estimated = model.observation_noise(model.analysis(tracked))[:, :2]
    power = .5 * model.analysis(complex_iq(epsilon) / frame)[..., model.outer].abs().square()
    measured = torch.stack((power[:, 0].mean(-1), power[:, 1:].mean((1, 2))), dim=1)
    bias = 10 * (estimated.clamp_min(1e-10) / measured.clamp_min(1e-10)).log10()
    return np.asarray(bias.tolist(), dtype=np.float32)


def joint_noise_bias(model, tracked, frame, epsilon):
    received = spectrum_parts(split_spectrum(tracked))[...,model.outer]
    truth = spectrum_parts(split_spectrum(complex_iq(epsilon)/frame))[...,model.outer]
    estimate = received.abs().square().median(-1).values / log(2.)
    actual = truth.abs().square().mean(-1)
    pairs = [torch.stack((value[:,:1].mean(1),value[:,1:].mean(1)),dim=1) for value in (estimate,actual)]
    bias = 10.*(pairs[0].clamp_min(1e-10)/pairs[1].clamp_min(1e-10)).log10()
    return np.asarray(bias.tolist(),dtype=np.float32)


def joint_part_metrics(waves, reference, frame):
    normalized = torch.stack(waves) / frame
    parts = spectrum_parts(normalized.flatten(0,1)).unflatten(0,normalized.shape[:2])
    desired = spectrum_parts(reference/frame)
    return np.stack([spectral_metrics(parts[:,:,select],desired[:,select],1.)
                     for select in (slice(0,1),slice(1,None))],axis=-2)


def anchored_action_metrics(model, observed, quality, rms, reference, noiseless, frame):
    _,_,action,variance = model.states(observed,quality,rms)
    normalized = reference/frame
    baseline = (normalized-observed).abs().square().mean((1,2))
    predicted = (action*normalized-observed).abs().square().mean((1,2))
    expected = (observed-noiseless/frame).abs().square().mean((1,2))
    return np.asarray(torch.stack((baseline,predicted,variance.mean((1,2)),expected),dim=1).tolist(),dtype=np.float32)


def received_update_evidence(model, y, q, rms, aligned_source, noiseless, frame):
    coarse, final, stages, estimates = model.states(y, q, rms, reference=aligned_source / frame)
    action, outer, hold = estimates
    aligned = aligned_source / frame
    noise = y - noiseless / frame
    values = [(y - aligned).abs().square().mean((1, 2)),
              (y - action * aligned).abs().square().mean((1, 2)),
              outer.mean((1, 2)), hold.mean((1, 2)), noise.abs().square().mean((1, 2))]
    for stage_id, (restored, stats) in enumerate(zip((coarse, final), stages, strict=True)):
        estimate, sigma, projection, valid, gain = stats
        lower = estimate - model.kappa[stage_id] * sigma
        supported = (lower <= projection) | ~valid
        values.extend((supported.flatten(1).all(1), (gain > 0).float().flatten(1).mean(1),
                       ((y - aligned).abs().square() - (restored - aligned).abs().square()).mean((1, 2))))
    return np.asarray(torch.stack(values, -1).tolist(), dtype=np.float32)


def phase_placement(waves, epsilon, frame):
    raw = waves[0]
    noise = complex_iq(epsilon) / frame
    reference = raw - (noise - noise.mean(-1, keepdim=True)) * frame
    present = noise.abs().amax((-2, -1), keepdim=True) > 0
    reference = torch.where(present, torch.polar(raw.abs(), reference.angle()), raw)
    edit = torch.stack(waves) - raw
    ideal = reference - raw
    values = []
    intervals = (slice(None), *(slice(start, start + 64) for start in range(0, 256, 64)))
    for region in intervals:
        selected = edit[..., region]
        wanted = ideal[..., region]
        norm = selected.abs().square().sum((-2, -1))
        reference_norm = wanted.abs().square().sum((-2, -1))
        raw_norm = raw[..., region].abs().square().sum((-2, -1)).clamp_min(1e-8)
        inner = (selected * wanted.conj()).real.sum((-2, -1))
        values.append(torch.stack(((norm / raw_norm).sqrt(),
                                   inner / (norm * reference_norm).sqrt().clamp_min(1e-8),
                                   (2 * inner - norm) / raw_norm), dim=-1))
    return np.asarray(torch.stack(values, dim=-2).tolist(), dtype=np.float32)


def neighbor_explained_phase(model, raw, tracked, q, rms, phase):
    from experiments.pamf_lora_main.local_phase import project
    from experiments.pamf_lora_main.neighbor_phase import neighbor_angle, remove_parallel
    from experiments.pamf_lora_main.phase_observation_diagnostic import repeat_angle

    power = raw.abs().square()
    shared = tracked * rms / rms.square().mean(1, keepdim=True).sqrt()
    within, agreement = neighbor_angle(tracked)
    window = torch.ones_like(model.time)

    def projected(z, mask):
        return project(z, power, mask, model.time, angle_cap=.65)

    repeat = projected(repeat_angle(shared, q), window)
    repeat_inside = projected(repeat, model.valid_neighbors)
    neighbor = remove_parallel(projected(within, model.valid_neighbors), repeat_inside, power)
    target = projected(phase, window)
    energy = (power * target.square()).sum((1, 2)).clamp_min(1e-10)
    explained = [((power * target * direction).sum(-1).square() /
                  (power * direction.square()).sum(-1).clamp_min(1e-10)).sum(1) / energy
                 for direction in (repeat, neighbor)]
    return torch.stack((*explained, agreement.mean((1, 2))), -1)


@torch.no_grad()
def neighbor_held_support(rt):
    from experiments.pamf_lora_main.channel_blind_source import setup
    from experiments.pamf_lora_main.dura_repair import real_iq
    from experiments.pamf_lora_main.neighbor_phase import NeighborPhaseDose
    from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation

    if rt['run_id'] != NEIGHBOR_PHASE_RUN:
        raise ValueError('This U-only diagnostic uses the registered neighbor-phase source data')
    torch.set_num_threads(3)
    (_, _, held), _ = setup(rt, 'cpu', fit_per_rx=4000)
    model = NeighborPhaseDose(True).eval()
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator().manual_seed(2026112477)
    by_rx = {}
    for rx, batches in zip(SOURCE, held.chunk(len(SOURCE)), strict=True):
        measures = []
        for iq in batches.split(24):
            ground = complex_iq(iq)
            received = burst_observation(ground, rng)
            tracked, _, q, rms = coordinates(tracker, real_iq(received))
            phase = (ground * received.conj()).angle()
            measures.append(neighbor_explained_phase(model, received, tracked, q, rms, phase))
        data = torch.cat(measures, 0)
        by_rx[rx] = dict(n=len(data), median=data.median(dim=0).values.tolist())
    return dict(n=len(held), target_reads=0, source_V_reads=0, classifier_reads=0,
                fields=['repeat_explained_energy', 'additional_explained_energy', 'neighbor_support'],
                by_receiver=by_rx)


def shared_interval_evidence(model, observed, quality, waves, source, frame):
    (_, _, _, confidence), _ = model.spectral_evidence(observed, quality)
    confidence = confidence.squeeze(1)
    estimate = torch.fft.fft((torch.stack(waves) / frame).mean(-2), norm='ortho')
    expected = torch.fft.fft((source / frame).mean(-2), norm='ortho')
    cross = (estimate * expected.conj()).real
    ref_power = expected.abs().square()
    estimate_power = estimate.abs().square()
    f = torch.fft.fftfreq(frame.shape[-1], device=frame.device).abs() * frame.shape[-1]
    outputs = []
    for band in ((f>0.) & (f<=8.), (f>8.) & (f<=32.)):
        for confirmed in (confidence >= .10, confidence < .10):
            selection = confirmed & band
            weight = selection * ref_power
            predicted = (selection * estimate_power).sum(-1)
            true = weight.sum(-1)
            overlap = (selection * cross).sum(-1)
            counts = selection.sum(-1).expand_as(overlap)
            outputs.append(torch.stack((counts, true.expand_as(overlap), predicted, overlap), dim=-1))
    return np.asarray(torch.stack(outputs, dim=-2).tolist(), dtype=np.float64)


def report_interval(values):
    count, reference, estimated, dot = values.sum(axis=1).transpose(2, 0, 1)
    covariance = dot / np.maximum((reference * estimated)**.5, 1e-12)
    ratio = estimated / np.maximum(reference, 1e-12)
    signed = dot / np.maximum(reference, 1e-12)
    result = np.stack((count, ratio, signed, covariance), axis=-1)
    return [[list(row) if row[0] > 0. else [0, None, None, None] for row in mode] for mode in result.tolist()]


RELATION_FIELDS = ('log_distance_rmse', 'neighbor_recall_at_5', 'paired_source_retrieval_at_1')


def distance_matrix(left, right):
    cross = left @ right.transpose(-2,-1) / left.shape[-1]
    return (left.square().mean(-1)[...,None] + right.square().mean(-1)[...,None,:] - 2*cross).clamp_min(0.)


def relation_summary(clean_features, output_features):
    n = clean_features[0].shape[0]
    device = clean_features[0].device
    diagonal = torch.eye(n,dtype=torch.bool,device=device)
    source_pairs, restored_pairs, cross_pairs, log_risks = [], [], [], []
    for clean, output in zip(clean_features,output_features,strict=True):
        center = clean.mean(0)
        variance = clean.var(0,unbiased=False)
        scale = (variance + .05*variance.mean() + 1e-5).sqrt()
        reference, restored = (clean-center)/scale, (output-center)/scale
        true = distance_matrix(reference,reference)
        estimate = distance_matrix(restored,restored)
        error = estimate[...,~diagonal].log1p()-true[~diagonal].log1p()
        log_risks.append(error.square().mean(-1).sqrt())
        source_pairs.append(true.log1p())
        restored_pairs.append(estimate.log1p())
        cross_pairs.append(distance_matrix(restored,reference).log1p())
    true = torch.stack(source_pairs).mean(0).masked_fill_(diagonal,torch.inf)
    estimate = torch.stack(restored_pairs).mean(0).masked_fill_(diagonal,torch.inf)
    expected = true.argsort(dim=-1,stable=True)[...,:5]
    selected = estimate.argsort(dim=-1,stable=True)[...,:5]
    recall = (selected[..., :, None]==expected[:,None,:]).any(-1).float().mean(dim=(-2,-1))
    retrieval = (torch.stack(cross_pairs).mean(0).argmin(-1)==torch.arange(n,device=device)).float().mean(-1)
    return torch.stack((torch.stack(log_risks).mean(0),recall,retrieval),dim=-1)


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
    is_held = rt['run_id']==held_chirp_precision_source.RUN
    is_relation = rt['run_id']==packet_relation_source.RUN
    is_cancel = rt['run_id']==reference_cancel_source.RUN
    is_transport = rt['run_id']==variance_transport_source.RUN
    is_cross_frequency = rt['run_id']=='20261124-phase1-lora-cross-frequency-source-m2-r01'
    is_cross_dose = rt['run_id']=='20261124-phase1-lora-cross-frequency-dose-source-m4-r01'
    is_mixture = rt['run_id']=='20261124-phase1-lora-spectral-mixture-source-m2-r01'
    is_angular = rt['run_id']=='20261124-phase1-lora-angular-transport-source-m4-r01'
    is_band_allocation = rt['run_id']=='20261124-phase1-lora-band-allocation-source-m3-r01'
    is_weak_sideband = rt['run_id']=='20261124-phase1-lora-weak-sideband-source-m4-r01'
    is_interval = rt['run_id']==coherent_interval_source.RUN
    is_joint = rt['run_id']==joint_spectrum_source.RUN
    is_anchor = rt['run_id']==ANCHOR_RUN
    is_received = rt['run_id']==received_update_source.RUN
    is_neighbor = rt['run_id'] == NEIGHBOR_PHASE_RUN
    is_dose = rt['run_id'] in (phase_dose_source.RUN, BURST_DOSE_RUN, NEIGHBOR_PHASE_RUN)
    if is_dose:
        study = phase_dose_source
        if rt['run_id'] == BURST_DOSE_RUN:
            from experiments.pamf_lora_main import phase_dose_bursts_source as study
        if is_neighbor:
            from experiments.pamf_lora_main import neighbor_phase_source as study
        recipes = study.RECIPES
        recover = partial(study.recover, compare=True)
        readouts = ('direct', *study.MODES[2:])
        indices = range(len(study.MODES)-2)
    elif is_received:
        study = received_update_source
        recipes = study.RECIPES
        recover = partial(study.recover, matrix=True)
        coordinate_frame = update_coordinates
        readouts = ('direct', *study.MODES[2:])
        indices = range(len(study.MODES)-2)
    elif is_anchor:
        from experiments.pamf_lora_main import common_anchor_source as study
        from experiments.pamf_lora_main.joint_spectrum_source import recover as joint_recover
        recipes = study.RECIPES
        recover = partial(joint_recover,compare=True,switch='anchor_common')
        readouts = ('direct',*study.MODES[2:])
        indices = range(len(study.MODES)-2)
    elif is_joint:
        study = joint_spectrum_source
        recipes = study.RECIPES
        recover = partial(study.recover,compare=True)
        readouts = ('direct',*study.MODES[2:])
        indices = range(len(study.MODES)-2)
    elif is_interval:
        study = coherent_interval_source
        recipes = study.RECIPES
        recover = partial(study.recovery, ablate=True)
        readouts = ('direct', 'scalar_full', 'interval_full', 'interval_matched',
                    'scalar_matched', 'interval_off')
        indices = tuple(study.MODES.index(name) - 2 for name in readouts[1:])
    elif is_transport or is_cross_frequency or is_mixture:
        if is_cross_frequency:
            from experiments.pamf_lora_main import cross_frequency_source as study
        elif is_mixture:
            from experiments.pamf_lora_main import spectral_mixture_source as study
        else:
            study = variance_transport_source
        recover = lambda iq, tracker, models, threshold: study.recover(iq, tracker, models[0], threshold)
        readouts = ('direct', *study.MODES[2:])
        indices = range(len(study.MODES) - 2)
    elif is_cross_dose or is_angular or is_weak_sideband:
        if is_angular:
            from experiments.pamf_lora_main import angular_transport_source as study
        elif is_weak_sideband:
            from experiments.pamf_lora_main import weak_sideband_source as study
        else:
            from experiments.pamf_lora_main import cross_frequency_dose_source as study
        recipes = study.RECIPES
        recover = partial(study.recover, compare=True)
        readouts = ('direct', *study.MODES[2:])
        indices = range(len(study.MODES) - 2)
    elif is_band_allocation:
        from experiments.pamf_lora_main import band_allocation_source as study
        recipes = study.RECIPES
        recover = partial(study.recover, matrix=True)
        readouts = ('direct', 'anchor', 'observed_allocation', 'learned_coarse',
                    'learned_final', 'unit_action', 'fixed_noise')
        indices = tuple(study.MODES.index(name) - 2 for name in readouts[1:])
    elif is_cancel:
        study = reference_cancel_source
        recipes = study.RECIPES
        recover = partial(study.recovery, controls=True)
        readouts = ('direct', 'fixed_full', 'guarded_full', 'guarded_half', 'guarded_without_cancel')
        indices = tuple(study.MODES.index(name) - 2 for name in readouts[1:])
    elif is_geometry or is_tangent or is_split or is_lapped or is_held or is_relation:
        if is_tangent:
            from experiments.pamf_lora_main import empirical_tangent_source as study
            recipes = study.recipes(rt)
        elif is_split:
            study = split_shrink_source
            recipes = study.RECIPES
        elif is_lapped:
            study = lapped_posterior_source
            recipes = study.RECIPES
        elif is_held:
            study = held_chirp_precision_source
            recipes = study.RECIPES
        elif is_relation:
            study = packet_relation_source
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
    if not is_received:
        coordinate_frame = coordinates
    root = Path(rt['run_root'])
    data = Path(rt['source_root'])
    cache = root.parent/DATA_RUN/'source_views'
    contract = source_contract(data)
    rows, registered, clean, selected, marks = paired_source(data,cache,contract,root,route,study.MODES)
    if is_transport or is_cross_frequency or is_mixture:
        threshold = study.load_gate(root)
        models = (study.load(root, contract, threshold, device),)
    else:
        models, threshold = spectral_block_source.load_models(root,contract,device,recipes)
    if is_split:
        models = (*models, split_shrink_source.ScalarReadout(models[-1]).eval())
    elif is_lapped:
        models = (*models, lapped_posterior_source.PooledReadout(models[-1]).eval())
    elif is_held:
        models = (*models, held_chirp_precision_source.ScalarReadout(models[-1]).eval())
    tracker = PacketRestorer('consensus').to(device).eval()
    truth = tensor(clean,device)
    if is_relation:
        canonical,rotation,_,scale = coordinates(tracker,truth)
        source_frames = (rotation*scale).split(18)
        source_features = fingerprint(canonical)
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
            held_stats = []
            relation_stats = []
            cancellation_stats = []
            transport_stats = []
            interval_stats = []
            joint_stats = []
            action_stats = []
            phase_stats = []
            neighbor_stats = []
            angular_stats = []
            batches = zip(tensor(capture,device).split(18),tensor(noise,device).split(18),truth.split(18),strict=True)
            for batch,(received, epsilon, source) in enumerate(batches):
                recoveries, count = recover(received,tracker,models,threshold)[:2]
                active += count
                y = complex_iq(received)
                waves = [y,*(complex_iq(recoveries[k]) for k in indices)]
                tracked, phase, quality, rms = coordinate_frame(tracker,received)
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
                if is_held:
                    held_stats.append(held_risk(models[1],tracked,quality,frame,ground))
                if is_relation:
                    relation_stats.append(fingerprint(torch.stack(waves)/source_frames[batch]))
                if is_cancel:
                    cancellation_stats.append(cancellation_evidence(models[1],tracked,frame,epsilon,view=='clean'))
                if is_transport:
                    transport_stats.append(transport_evidence(models[0], waves, target, frame))
                    if view != 'clean':
                        noise_stats.append(transport_noise_bias(models[0], tracked, frame, epsilon))
                if is_interval:
                    interval_stats.append(shared_interval_evidence(models[1],tracked,quality,waves,target,frame))
                if is_joint:
                    joint_stats.append(joint_part_metrics(waves,noiseless,frame))
                    if view!='clean':
                        noise_stats.append(joint_noise_bias(models[1],tracked,frame,epsilon))
                if is_cross_frequency or is_cross_dose or is_mixture or is_angular or is_band_allocation or is_weak_sideband:
                    joint_stats.append(joint_part_metrics(waves,noiseless,frame))
                    if view != 'clean':
                        prior = models[-1].prior if is_cross_dose or is_angular or is_band_allocation or is_weak_sideband else models[0]
                        estimated = prior.noise(prior.spectrum(tracked))
                        true_spectrum = prior.spectrum(complex_iq(epsilon) / frame)[..., prior.outer].abs().square()
                        true = .5 * torch.stack((true_spectrum[:, 0].mean(-1),
                                                 true_spectrum[:, 1:].mean((1, 2))), dim=-1)
                        error_db = 10 * (estimated / true.clamp_min(1e-10)).clamp_min(1e-10).log10()
                        noise_stats.append(np.asarray(error_db.tolist(), dtype=np.float32))
                if (is_angular or is_band_allocation or is_weak_sideband) and view != 'clean':
                    wanted = -complex_iq(epsilon) / frame
                    wanted = (wanted - wanted.mean(-1, keepdim=True)) * frame
                    edit = torch.stack(waves) - y
                    dot = (edit * wanted.conj()).real.sum((2, 3))
                    target_power = wanted.abs().square().sum((1, 2)).clamp_min(1e-10)
                    edit_power = edit.abs().square().sum((2, 3))
                    cosine = dot / (edit_power * target_power).clamp_min(1e-12).sqrt()
                    angular_stats.append(np.asarray(torch.stack((cosine, dot / target_power), dim=-1).tolist()))
                if is_anchor:
                    joint_stats.append(joint_part_metrics(waves,noiseless,frame))
                    action_stats.append(anchored_action_metrics(models[1],tracked,quality,rms,target,noiseless,frame))
                if is_received:
                    joint_stats.append(joint_part_metrics(waves,noiseless,frame))
                    action_stats.append(received_update_evidence(models[1],tracked,quality,rms,target,noiseless,frame))
                if is_dose:
                    joint_stats.append(joint_part_metrics(waves,noiseless,frame))
                    phase_stats.append(phase_placement(waves, epsilon, frame))
                if is_neighbor and view != 'clean':
                    noise = complex_iq(epsilon) / frame
                    exact = y - (noise - noise.mean(-1, keepdim=True)) * frame
                    phase_delta = (exact * y.conj()).angle()
                    values = neighbor_explained_phase(models[-1], y, tracked, quality, rms, phase_delta)
                    neighbor_stats.append(np.asarray(values.tolist(), dtype=np.float32))
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
            if is_held:
                values = np.concatenate(held_stats)
                output[view]['held_risk'] = dict(fields=HELD_RISK_FIELDS, median=np.median(values,axis=0).tolist(),
                    by_receiver={name:np.median(part,axis=0).tolist()
                                 for name,part in zip(SOURCE,np.split(values,len(SOURCE)),strict=True)})
            if is_cancel:
                values = np.concatenate(cancellation_stats)
                output[view]['cancellation'] = dict(fields=CANCEL_FIELDS[:values.shape[1]],
                    median=np.median(values,axis=0).tolist(), mean=np.mean(values,axis=0).tolist(),
                    by_receiver={name:np.median(part,axis=0).tolist()
                                 for name,part in zip(SOURCE,np.split(values,len(SOURCE),axis=0),strict=True)})
            if is_transport:
                values = np.concatenate(transport_stats, axis=1)
                output[view]['transport'] = dict(fields=TRANSPORT_FIELDS,
                    groups=['near_shared', 'near_private', 'middle_shared', 'middle_private'],
                    median=np.median(values, axis=1).tolist(),
                    by_receiver={name:np.median(part, axis=1).tolist()
                                 for name,part in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
                if noise_stats:
                    error = np.concatenate(noise_stats)
                    output[view]['outer_noise_bias_db'] = dict(order=['shared', 'private'],
                        median=np.median(error, axis=0).tolist(),
                        by_receiver={name:np.median(part, axis=0).tolist()
                                     for name,part in zip(SOURCE,np.split(error,len(SOURCE)),strict=True)})
            if is_interval:
                values = np.concatenate(interval_stats, axis=1)
                output[view]['interval_support'] = dict(
                    order=['near_supported','near_other','middle_supported','middle_other'],
                    fields=['bins','power_ratio','signed_gain','complex_correlation'],
                    pooled=report_interval(values),
                    by_receiver={name:report_interval(part) for name,part in
                                 zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
            if (is_joint or is_anchor or is_received or is_dose or is_cross_frequency
                    or is_cross_dose or is_mixture or is_angular or is_band_allocation or is_weak_sideband):
                values = np.concatenate(joint_stats,axis=1)
                output[view]['components_noiseless'] = dict(order=['shared','private'],
                    median=np.median(values,axis=1).tolist(),
                    by_receiver={name:np.median(group,axis=1).tolist()
                                 for name,group in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
                if noise_stats:
                    error = np.concatenate(noise_stats)
                    output[view]['outer_noise_bias_db'] = dict(order=['shared','private'],
                        median=np.median(error,axis=0).tolist(),
                        by_receiver={name:np.median(group,axis=0).tolist()
                                     for name,group in zip(SOURCE,np.split(error,len(SOURCE)),strict=True)})
            if angular_stats:
                values = np.concatenate(angular_stats, axis=1)
                output[view]['exact_direction_alignment'] = dict(fields=['real_cosine', 'projected_exact_fraction'],
                    median=np.median(values, axis=1).tolist(),
                    by_receiver={name:np.median(row, axis=1).tolist()
                                 for name,row in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
            if is_anchor:
                actions = np.concatenate(action_stats)
                output[view]['action_error'] = dict(fields=['identity_residual','estimated_action_residual',
                    'estimated_residual_variance','exact_added_noise_power'],median=np.median(actions,axis=0).tolist(),
                    by_receiver={name:np.median(group,axis=0).tolist() for name,group in
                                 zip(SOURCE,np.split(actions,len(SOURCE)),strict=True)})
            if is_received:
                actions = np.concatenate(action_stats)
                fields = ['identity_residual','estimated_action_residual','outer_variance','held_residual',
                          'exact_added_noise_power','coarse_lower_coverage','coarse_positive_fraction',
                          'coarse_actual_waveform_gain','final_lower_coverage','final_positive_fraction',
                          'final_actual_waveform_gain']
                output[view]['direction_risk'] = dict(
                    fields=fields, median=np.median(actions,axis=0).tolist(), mean=actions.mean(0).tolist(),
                    by_receiver={name:part.mean(0).tolist() for name,part in
                                 zip(SOURCE,np.split(actions,len(SOURCE)),strict=True)})
            if is_dose:
                placement = np.concatenate(phase_stats, axis=1)
                output[view]['phase_placement'] = dict(
                    order=['whole', '0–63', '64–127', '128–191', '192–255'],
                    fields=['physical_edit_rms_ratio', 'edit_cosine_to_isolated_phase',
                            'normalized_squared_error_improvement'],
                    mean=placement.mean(1).tolist(), median=np.median(placement, axis=1).tolist(),
                    by_receiver={name:part.mean(1).tolist() for name,part in
                                 zip(SOURCE,np.split(placement,len(SOURCE),axis=1),strict=True)})
            if neighbor_stats:
                identified = np.concatenate(neighbor_stats)
                output[view]['neighbor_geometry'] = dict(
                    fields=['repeat_explained_energy', 'additional_explained_energy', 'neighbor_support'],
                    median=np.median(identified, axis=0).tolist(),
                    by_receiver={name:np.median(part, axis=0).tolist() for name, part in
                                 zip(SOURCE,np.split(identified,len(SOURCE)),strict=True)})
                with_neighbor, without_neighbor = (placement[readouts.index(name)] for name in
                                                   ('neighbor_equal', 'repeat_equal'))
                edited = (with_neighbor[:, 0, 0] > 1e-7) & (without_neighbor[:, 0, 0] > 1e-7)
                change = with_neighbor[..., 2] - without_neighbor[..., 2]
                output[view]['neighbor_equal_phase_gain'] = equal_phase_gain(change, edited)
            if is_relation:
                estimates = [torch.cat(parts,dim=1) for parts in zip(*relation_stats,strict=True)]
                reference_groups = zip(*(value.chunk(len(SOURCE),dim=0) for value in source_features),strict=True)
                estimate_groups = zip(*(value.chunk(len(SOURCE),dim=1) for value in estimates),strict=True)
                by_receiver = {name:relation_summary(reference,estimate)
                               for name,reference,estimate in zip(SOURCE,reference_groups,estimate_groups,strict=True)}
                output[view]['packet_relation'] = dict(fields=RELATION_FIELDS,readouts=readouts,
                    mean=torch.stack(tuple(by_receiver.values())).mean(0).tolist(),
                    by_receiver={name:value.tolist() for name,value in by_receiver.items()})
    if is_geometry:
        readouts = (*readouts,'bounded_pair_projection')
    return dict(route=route,n=len(rows),bands=BANDS,metrics=METRICS,readouts=readouts,scenes=output)


def equal_phase_gain(change, edited):
    def summarize(values, valid):
        values = values[valid]
        if len(values) == 0:
            return dict(n=0, median=None, positive_fraction=None)
        return dict(n=len(values), median=np.median(values, axis=0).tolist(),
                    positive_fraction=(values > 1e-7).mean(axis=0).tolist())

    return dict(pooled=summarize(change, edited),
                by_receiver={name:summarize(values, valid) for name, values, valid in
                             zip(SOURCE, np.split(change, len(SOURCE)), np.split(edited, len(SOURCE)), strict=True)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('route',choices=(*ROUTES, 'neighbor-held'))
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    rt = read(args.runtime)
    if args.route == 'neighbor-held':
        print(json.dumps(neighbor_held_support(rt), allow_nan=False))
        return
    root = Path(rt['run_root'])
    analyzed_runs = (local_phase_source.RUN, GEOMETRY_RUN, TANGENT_RUN, split_shrink_source.RUN,
                     lapped_posterior_source.RUN, held_chirp_precision_source.RUN, packet_relation_source.RUN,
                     reference_cancel_source.RUN, variance_transport_source.RUN, coherent_interval_source.RUN,
                     joint_spectrum_source.RUN, ANCHOR_RUN, received_update_source.RUN,
                     phase_dose_source.RUN, BURST_DOSE_RUN, NEIGHBOR_PHASE_RUN,
                     '20261124-phase1-lora-cross-frequency-source-m2-r01',
                     '20261124-phase1-lora-cross-frequency-dose-source-m4-r01',
                     '20261124-phase1-lora-spectral-mixture-source-m2-r01',
                     '20261124-phase1-lora-angular-transport-source-m4-r01',
                     '20261124-phase1-lora-band-allocation-source-m3-r01',
                     '20261124-phase1-lora-weak-sideband-source-m4-r01')
    if rt.get('run_id') not in analyzed_runs or read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('The full dual-route truth-last scoring must already be complete')
    activate(rt)
    print(json.dumps(diagnose(rt,args.route,args.device),allow_nan=False))


if __name__=='__main__':
    main()
