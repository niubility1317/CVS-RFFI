"""Check whether the SatIQ stopband distinguishes beams on matched source panels."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import apply_calibration, fit_calibration
from .paper_eval import metrics
from .paper_native_pamf import native_frequency_taper
from .paper_signal_readout import COMMON_SAMPLES, panel_rows, source_panels
from .paper_train import source_inputs
from .train import device_array


PASS_EDGES = tuple(np.rint(np.linspace(1, 80, 13)).astype(int))
STOP_EDGES = tuple(np.rint(np.geomspace(80, COMMON_SAMPLES // 2, 13)).astype(int))


def band_shape(power, edges):
    energy = torch.stack([power[..., lo:hi].sum(-1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)], dim=-1)
    widths = energy.new_tensor(np.diff(edges).tolist())
    total = energy.sum(dim=(1, 2), keepdim=True)
    return ((energy / widths) / total.clamp_min(1e-12)).clamp_min(1e-12).log().flatten(1)


def spectral_features(iq):
    common = iq[:, :, :COMMON_SAMPLES]
    if common.shape[1:] != (2, COMMON_SAMPLES):
        raise ValueError("Spectral readout requires the complete 8000-point common word")
    centered = common - common.mean(dim=-1, keepdim=True)
    power = torch.fft.fft(torch.complex(centered[:, 0], centered[:, 1])).abs().square()
    half = COMMON_SAMPLES // 2
    negative = F.pad(power[:, half + 1:].flip(-1), (1, 0))
    signed = torch.stack((power[:, :half], negative), dim=1)
    bins = torch.arange(80, dtype=power.dtype, device=power.device)
    taper = native_frequency_taper(bins).square()
    passed = band_shape(signed[..., :80] * taper, PASS_EDGES)
    stopped = band_shape(signed, STOP_EDGES)
    fraction = signed[..., 80:].sum(dim=(1, 2)) / signed[..., 1:].sum(dim=(1, 2)).clamp_min(1e-12)
    return passed, stopped, fraction


def extract(raw, rows):
    batches = [spectral_features(device_array(raw[block], torch.device("cpu")))
               for block in np.array_split(rows, (len(rows) + 127) // 128)]
    return tuple(torch.cat(parts) for parts in zip(*batches, strict=True))


@torch.no_grad()
def compare(source, cache, seed):
    classes, _, roles = source_inputs(source, cache)
    raw, labels, _ = roles["V"]
    panels, cutoff = source_panels(source, labels, classes, seed)
    needed, panels = panel_rows(panels, len(raw))
    training_raw = roles["L"][0]
    fit_rows = np.sort(np.random.default_rng(seed).choice(len(training_raw), 8192, replace=False))
    l_pass, l_stop, l_fraction = extract(training_raw, fit_rows)
    v_pass, v_stop, v_fraction = extract(raw, needed)

    passed = F.normalize(apply_calibration(v_pass, fit_calibration(l_pass)), dim=1)
    stopped = F.normalize(apply_calibration(v_stop, fit_calibration(l_stop)), dim=1)
    log_fraction = l_fraction.clamp_min(1e-12).log()
    centered = (v_fraction.clamp_min(1e-12).log() - log_fraction.mean()) / log_fraction.std().clamp_min(1e-6)
    energy = F.normalize(torch.stack((torch.ones_like(centered), centered), dim=1), dim=1)
    joint = torch.cat(((.75 ** .5) * passed, (.25 ** .5) * stopped), dim=1)
    device = torch.device("cpu")
    results = {}
    for name, features in (("passband_shape", passed), ("stopband_shape", stopped),
                           ("stopband_energy_fraction", energy), ("combined_shape", joint)):
        vectors = np.from_dlpack(features)
        results[name] = {tag: metrics(vectors, panel, device) for tag, panel in panels.items()}
        print(name, {tag: row["K16"]["same_satellite"]["auc"] for tag, row in results[name].items()}, flush=True)
    quantiles = torch.tensor([.1, .5, .9])
    return dict(fit_count=len(fit_rows), seed=seed, cutoff=cutoff, feature_records=len(needed),
                stopband_fraction_L=torch.quantile(l_fraction, quantiles).tolist(),
                stopband_fraction_V=torch.quantile(v_fraction, quantiles).tolist(), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.seed))
