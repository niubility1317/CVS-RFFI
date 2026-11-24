"""Measure frozen uncertainty on disjoint raw LoRa source U packets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import DATA_RUN, partition, source_contract, source_role
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation
from experiments.pamf_lora_main.posterior_transport_source import RECIPES, RUN
from experiments.pamf_lora_main.spectral_block_source import load_models


REGIMES = (("gaussian_005", .05), ("gaussian_020", .20), ("gaussian_045", .45), ("mixed_burst", None))
BANDS = ((1, 8), (9, 32), (33, 128))


def projection(edit, direction, rms):
    physical_direction = direction * rms
    energy = physical_direction.abs().square().sum((1, 2)).clamp_min(1e-12)
    return ((edit * rms * physical_direction.conj()).sum((1, 2)).real / energy)


def probe(ground, noise_level, rng):
    if noise_level is None:
        return burst_observation(ground, rng)
    scale = ground.abs().square().mean((1, 2), keepdim=True).sqrt()
    standard = torch.randn(ground.shape, dtype=ground.dtype, device=ground.device, generator=rng)
    return ground + noise_level * scale * standard


@torch.inference_mode()
def diagnostics(models, tracker, ground, delivered, threshold, rx):
    y, phase, _, rms = coordinates(tracker, real_iq(delivered))
    truth = ground / (phase * rms)
    active = corruption_score(y) > threshold
    source_error = truth - y
    ground_power = ground.abs().square().sum((1, 2)).clamp_min(1e-12)
    rows = []
    for name, model in zip(RECIPES, models, strict=True):
        context = model.observe(y)
        mean, variance = model.estimate(y, rms, context)[1]
        squared = (context[0] + mean - model.spectral(truth)).abs().square()
        frequency = (256 * model.frequency).round().abs()
        calibration = {}
        for component, index in (("common", slice(0, 1)), ("private", slice(1, None))):
            for low, high in BANDS:
                band = (frequency >= low) & (frequency <= high)
                error = squared[:, index, band]
                predicted = variance[:, index, band]
                statistics = torch.stack(
                    (error.mean((1, 2)), predicted.mean((1, 2)),
                     (error / predicted).mean((1, 2)), (error <= predicted).float().mean((1, 2))), 1
                )
                calibration[f"{component}_{low}_{high}"] = np.asarray(statistics.tolist())
        direction = model.waveform(mean)
        fixed = model.readout(y, rms, (mean, variance), context[2], adaptive=False)
        adaptive = model.readout(y, rms, (mean, variance), context[2], adaptive=True)
        optimum = projection(source_error, direction, rms).clamp(0., 1.)
        gains = torch.stack((optimum, projection(fixed, direction, rms),
                             projection(adaptive, direction, rms)), dim=1)
        gated = (torch.zeros_like(fixed), fixed * active[:, None, None], adaptive * active[:, None, None])
        risks = torch.stack([((source_error - edit) * rms).abs().square().sum((1, 2)) / ground_power
                             for edit in gated], dim=1)
        rows.append((name, dict(active=np.asarray(active.tolist()), rx=rx, gains=np.asarray(gains.tolist()),
                                risks=np.asarray(risks.tolist()), **calibration)))
    return rows


def summary(batches):
    joined = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    if any(not np.isfinite(values).all() for values in joined.values() if np.issubdtype(values.dtype, np.number)):
        raise FloatingPointError("Nonfinite source holdout reliability")
    active = joined.pop("active")
    rx = joined.pop("rx")
    risks = joined.pop("risks")
    gains = joined.pop("gains")
    by_receiver = {}
    for receiver in SOURCE:
        select = (rx == receiver) & active
        optimum, fixed, adaptive = gains[select].T
        correlation = (float(spearmanr(optimum, adaptive).statistic)
                       if select.sum() >= 3 and np.ptp(optimum) and np.ptp(adaptive) else None)
        by_receiver[str(receiver)] = dict(n=int(select.sum()), adaptive_gain_spearman=correlation,
                                         gain_mae=np.mean(np.abs(gains[select, 1:] -
                                                                 gains[select, :1]), axis=0).tolist()
                                         if select.any() else None)
    calibration = {}
    for subset, select in (("all", np.ones(len(active), dtype=bool)), ("active", active)):
        calibration[subset] = {band: values[select].mean(0).tolist() for band, values in joined.items()} if select.any() else {}
    return dict(n=len(active), active=int(active.sum()), mean_physical_nmse=risks.mean(0).tolist(),
                active_by_receiver=by_receiver, calibration=calibration)


@torch.inference_mode()
def run(runtime, device):
    root = Path(runtime["run_root"])
    if root.name != RUN or read(root / "state.json")["status"] != "SOURCE_SCORED":
        raise ValueError("Need completed same-run source predictions")
    source = Path(runtime["source_root"])
    contract = source_contract(source)
    _, rx, _, _ = source_role(source, contract, "U_s")
    model_fit = {recipe.fit_per_rx for recipe in RECIPES.values()}
    (fit_per_rx,) = model_fit
    _, _, held = partition(rx, fit=fit_per_rx)
    raw = Path(runtime["parent_root"]).parent / DATA_RUN / "source_views" / "U_s.npy"
    iq = np.load(raw, mmap_mode="r", allow_pickle=False)
    if iq.shape != (1, len(rx), 6, 2, 256) or iq.dtype != np.float32:
        raise ValueError("Expected the single clean source U view")
    models, threshold = load_models(root, contract, device, RECIPES)
    tracker = PacketRestorer("consensus").to(device).eval()
    result = {}
    for regime, level in REGIMES:
        rng = torch.Generator(device=device).manual_seed(2026112468)
        arms = {name: [] for name in RECIPES}
        for indices in np.array_split(held, 21):
            ground = complex_iq(tensor(iq[0, indices], device))
            delivered = probe(ground, level, rng)
            for name, values in diagnostics(models, tracker, ground, delivered, threshold, rx[indices]):
                arms[name].append(values)
        result[regime] = {arm: summary(batches) for arm, batches in arms.items()}
    return dict(run=RUN, source_u_held=len(held), target_reads=0,
                calibration_columns=("mean_squared_error", "mean_predicted_variance",
                                     "mean_error_over_variance", "coverage_at_one_variance"),
                risk_columns=("direct", "fixed", "adaptive"), gain_mae_columns=("fixed", "adaptive"),
                regimes=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--device", default="cuda")
    options = parser.parse_args()
    print(json.dumps(run(read(options.runtime), options.device), allow_nan=False))
