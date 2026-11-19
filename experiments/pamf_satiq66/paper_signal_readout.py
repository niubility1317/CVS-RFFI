"""Evaluate SatIQ waveform representations on the fixed source panels."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import SYMBOLS, write_json
from .paper_eval import metrics, normalize_packet, panel
from .paper_train import source_inputs


SAMPLES_PER_SYMBOL = 1000
COMMON_SYMBOLS = 8
COMMON_SAMPLES = COMMON_SYMBOLS * SAMPLES_PER_SYMBOL


def common_template():
    n = torch.arange(COMMON_SAMPLES, dtype=torch.float32)
    symbols = torch.utils.dlpack.from_dlpack(SYMBOLS)
    transitions = torch.arange(COMMON_SYMBOLS - 1) * SAMPLES_PER_SYMBOL + 500
    profile = .5 + .5 * ((n[:, None] - transitions) / 140).tanh()
    return symbols[0] + (profile * symbols.diff()).sum(dim=1)


def align_common(signal, reference):
    common = signal[:, :COMMON_SAMPLES]
    template_mean = reference.mean()
    template_centered = reference - template_mean
    common_mean = common.mean(dim=1, keepdim=True)
    gain = ((common - common_mean) * template_centered.conj()).mean(dim=1, keepdim=True)
    gain = gain / template_centered.abs().square().mean()
    dc = common_mean - gain * template_mean
    aligned = (signal - dc) * gain.conj() / gain.abs().square().clamp_min(1e-8)
    return aligned


def ordered_iq(signal):
    symbols = signal.reshape(len(signal), -1, SAMPLES_PER_SYMBOL)
    iq = torch.view_as_real(symbols).flatten(0, 1).transpose(1, 2)
    return F.adaptive_avg_pool1d(iq, 32).reshape(len(signal), -1)


def waveform_views(iq, reference):
    iq = normalize_packet(iq)
    iq = iq - iq.mean(dim=(1, 2), keepdim=True)
    z = torch.complex(iq[:, 0], iq[:, 1])
    common = z[:, :COMMON_SAMPLES]
    old_window = z[:, 384:640].contiguous()
    aligned = align_common(z, reference)
    residual = aligned[:, :COMMON_SAMPLES] - reference
    delayed = z[:, 25:] * z[:, :-25].conj()
    delayed = F.pad(delayed, (25, 0))
    return {
        "raw_256": torch.view_as_real(old_window).flatten(1),
        "raw_8000": ordered_iq(common),
        "raw_11000": ordered_iq(z),
        "affine_aligned_8000": ordered_iq(aligned[:, :COMMON_SAMPLES]),
        "affine_aligned_11000": ordered_iq(aligned),
        "uw_residual_8000": ordered_iq(residual),
        "phase_difference_8000": ordered_iq(delayed[:, :COMMON_SAMPLES]),
        "phase_difference_11000": ordered_iq(delayed),
    }


def source_panels(source, labels, classes, seed):
    with np.load(source / "L_index.npz", allow_pickle=False) as index:
        train_times = index["timestamp_ns"]
    median = np.partition(train_times, len(train_times) // 2)[len(train_times) // 2]
    cutoff = np.datetime64(int(median), "ns").astype("datetime64[D]").astype("datetime64[ns]").astype(np.int64)
    with np.load(source / "V_index.npz", allow_pickle=False) as index:
        temporary = index["timestamp_ns"] < cutoff
    panels = {"paper": panel(labels, classes, seed),
              "chronological": panel(labels, classes, seed, reference_period=temporary)}
    return panels, str(np.datetime64(int(cutoff), "ns"))


def panel_rows(panels, record_count):
    # Extract only the physical records used by either fixed reference/query panel.
    needed = np.unique(np.concatenate([np.r_[p[0].ravel(), p[1]] for p in panels.values()]))
    data = np.full(record_count, -1, dtype=np.int64)
    data[needed] = np.arange(len(needed))
    remapped = {name: (data[ref], data[query], *mapping)
                for name, (ref, query, *mapping) in panels.items()}
    return needed, remapped


@torch.no_grad()
def diagnose(source, cache, seed):
    classes, _, roles = source_inputs(source, cache)
    raw, labels, _ = roles["V"]
    panels, cutoff = source_panels(source, labels, classes, seed)
    needed, panels = panel_rows(panels, len(raw))
    reference = common_template()
    values = {}
    for start in range(0, len(needed), 128):
        batch = torch.utils.dlpack.from_dlpack(raw[needed[start:start + 128]])
        for name, feature in waveform_views(batch, reference).items():
            values.setdefault(name, []).append(np.from_dlpack(feature))
    outcomes = {}
    for name, parts in values.items():
        feature = np.concatenate(parts)
        outcomes[name] = {"dimensions": feature.shape[1]}
        for label, indices in panels.items():
            outcomes[name][label] = metrics(feature, indices, torch.device("cpu"))
    return {"cutoff_utc": cutoff, "seed": seed,
            "distinct_packets": len(needed), "outcomes": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, diagnose(args.source, args.cache, args.seed))
