"""Source-calibrated radius on the complete received-IQ edit."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.w24_native_dura import capture_scale
from experiments.pamf_restoration.models import pair


CALIBRATION_PACKETS = 512
CALIBRATION_SEED = 2026113088


def log_relative_variance(variance, received):
    power = capture_scale(received).square().flatten()
    relative = variance.reshape(-1) / power
    return relative.clamp_min(1e-12).log()


def calibrate_values(clean_logvar, perturbed_logvar, perturbation_rms):
    clean = np.asarray(clean_logvar)
    perturbed = np.asarray(perturbed_logvar)
    disturbance = np.asarray(perturbation_rms)
    if not (clean.shape == perturbed.shape == disturbance.shape == (CALIBRATION_PACKETS,)):
        raise ValueError('Expected the preregistered native source calibration batch')
    if not all(np.isfinite(x).all() for x in (clean, perturbed, disturbance)):
        raise ValueError('Expected finite source-only uncertainty and paired disturbance')
    impaired = disturbance >= .35
    if np.count_nonzero(impaired) < 32:
        raise ValueError('Insufficient generic simulated impairments for fixed source calibration')
    lower = float(np.quantile(clean, .995))
    upper = max(lower + np.log(4.), float(np.quantile(perturbed[impaired], .25)))
    radius = min(.50, float(np.quantile(disturbance[impaired], .75)))
    return dict(lower_logvar=lower, upper_logvar=upper, max_radius=radius,
                clean_median=float(np.median(clean)), strong_median=float(np.median(perturbed[impaired])),
                strong_samples=int(impaired.sum()))


def project_output(received, recovered, normalized_received, variance, curve):
    """Preserve raw IQ exactly when the calibrated radius is zero."""
    logvar = log_relative_variance(variance, normalized_received)
    position = ((logvar - curve['lower_logvar']) /
                (curve['upper_logvar'] - curve['lower_logvar'])).clamp(0., 1.)
    radius = curve['max_radius'] * position.square() * (3. - 2. * position)
    edit = recovered - received
    relative_rms = edit.abs().square().mean(-1).sqrt() / capture_scale(received).flatten()
    amount = (radius / relative_rms.clamp_min(1e-12)).clamp(max=1.)
    return received + amount[:, None] * edit


@torch.no_grad()
def calibrate_source(rt, dataset, device):
    from experiments.pamf_lora_main.gain_dura_stage_fit import source_training_pair, training_components

    out = Path(rt['run_root']) / ('fit-' + dataset)
    out.mkdir()
    training_ids, reserved, sample, dura, _ = training_components(rt, dataset, device)
    if len(np.intersect1d(training_ids, reserved)):
        raise ValueError('Source calibration must exclude reserved smoke records')
    dura.eval().requires_grad_(False)
    rows = np.random.default_rng(CALIBRATION_SEED).choice(training_ids, CALIBRATION_PACKETS, replace=False)
    rng = torch.Generator(device=device).manual_seed(CALIBRATION_SEED + 1)
    clean_noise, disturbed_noise, edits = [], [], []
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for chunk in np.array_split(rows, CALIBRATION_PACKETS // 8):
        raw = sample(chunk)
        scale = capture_scale(raw)
        clean, disturbed = source_training_pair(raw, rng)
        original = raw / scale if dataset == 'lora' else raw
        if dataset != 'lora':
            clean, disturbed = clean * scale, disturbed * scale
        _, _, exact_clean = dura.estimate_context(pair(original))
        _, _, exact_disturbed = dura.estimate_context(pair(disturbed))
        clean_noise.extend(log_relative_variance(exact_clean, original).tolist())
        disturbed_noise.extend(log_relative_variance(exact_disturbed, disturbed).tolist())
        edits.extend(((clean - disturbed).abs().square().mean(-1, keepdim=True).sqrt()
                      / capture_scale(disturbed)).flatten().tolist())
    calibration = calibrate_values(clean_noise, disturbed_noise, edits)
    torch.cuda.synchronize()
    write(out/'calibration.json', dict(status='SOURCE_FROZEN', dataset=dataset, packets=CALIBRATION_PACKETS,
          seed=CALIBRATION_SEED, learned_parameters=0, admissible_U_count=len(training_ids),
          F_reads=0, V_reads=0, TX_reads=0, target_reads=0, calibration=calibration,
          elapsed_seconds=perf_counter()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print(dataset, 'SOURCE_CALIBRATION_FROZEN', calibration, flush=True)


def load_curve(rt, dataset):
    proof = read(Path(rt['run_root']) / ('fit-' + dataset) / 'calibration.json')
    if ((proof['status'], proof['dataset'], proof['packets'], proof['seed'])
            != ('SOURCE_FROZEN', dataset, CALIBRATION_PACKETS, CALIBRATION_SEED)
            or any(proof[k] for k in ('F_reads', 'V_reads', 'TX_reads', 'target_reads'))):
        raise ValueError('Expected fixed lawful source calibration')
    curve = proof['calibration']
    if (not all(np.isfinite(curve[k]) for k in ('lower_logvar', 'upper_logvar', 'max_radius'))
            or not curve['lower_logvar'] < curve['upper_logvar'] or not 0. < curve['max_radius'] <= .50):
        raise ValueError('Invalid frozen variance radius')
    return curve
