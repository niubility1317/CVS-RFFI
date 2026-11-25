"""Cross-fit a packet-shared LoRa channel estimate from public chirp coordinates."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, no_tone, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.relative_echo import shift
from tools.lora_w24_shared_guard_probe import RATES, complex_tensor, extract, generate_noise


ARMS = ('mirror', 'echo', 'joint')
DELAYS_US = (4, 8, 12, 16)
CASES = ('clean', 'white', 'mirror_weak', 'mirror_strong', 'echo_weak', 'echo_strong', 'joint_strong')
SEED = 2026112595


def delay(phase, microseconds):
    samples = phase.shape[-1] * microseconds / 1000
    low = int(samples)
    fraction = samples-low
    return ((1-fraction)*shift(phase, low) + fraction*shift(phase, low+1))*phase.conj()


def public_columns(phase):
    vectors = (phase.conj().square(), *(delay(phase, lag) for lag in DELAYS_US))
    return torch.stack([no_tone(vector) for vector in vectors], dim=-1)


def fit_shared(design, value):
    """Fit on alternate chirps and return the edit of each held-out chirp."""
    n = value.shape[-1]
    taper = torch.hann_window(n, periodic=False, dtype=value.real.dtype, device=value.device).sqrt()
    target = no_tone(value)
    weighted = taper[None, None, :, None]*design
    labels = (taper*target).unsqueeze(-1)
    correction = torch.empty_like(value)
    eye = torch.eye(design.shape[-1], device=value.device, dtype=value.dtype)
    for offset in (0, 1):
        x = weighted[:, offset::2].flatten(1, 2)
        y = labels[:, offset::2].flatten(1, 2)
        gram = x.mH @ x / x.shape[1]
        ridge = .025*gram.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(.01)
        coeff = torch.linalg.solve(gram+ridge[:, None, None]*eye, x.mH @ y / x.shape[1])
        edit = (design[:, 1-offset::2] @ coeff[:, None]).squeeze(-1)
        correction[:, 1-offset::2] = edit
    before = (taper*target).abs().square().mean((1, 2)).clamp_min(1e-12)
    after = (taper*(target-correction)).abs().square().mean((1, 2))
    return correction, (1-after/before).clamp_min(0)


def absolute_error(estimate, clean, rms):
    power = (clean*rms).abs().square().sum((1, 2)).clamp_min(1e-12)
    return ((estimate-clean)*rms).abs().square().sum((1, 2))/power


def private_error(estimate, clean, rms):
    n = clean.shape[-1]
    bins = (torch.arange(n, device=clean.device).add(n//2).remainder(n)-n//2).abs()
    select = (bins >= 9) & (bins <= 32)
    a, b = (torch.fft.fft(y-y.mean(1, keepdim=True), norm='ortho')[..., select] for y in (estimate*rms, clean*rms))
    return (a-b).abs().square().sum((1, 2)) / b.abs().square().sum((1, 2)).clamp_min(1e-12)


def panel(raw, trackers, observed=None):
    rows = []
    for start in range(0, len(raw), 32):
        clean = raw[start:start+32]
        capture = clean if observed is None else observed[start:start+32]
        rates = []
        for rate, tracker in trackers.items():
            iq = real_iq(complex_tensor(extract(capture, rate)))
            value, phase, _, rms = coordinates(tracker, iq)
            columns = public_columns(phase)
            fits = [fit_shared(basis, value) for basis in (columns[..., :1], columns[..., 1:], columns)]
            evidence = torch.stack([score for _, score in fits], dim=1)
            if observed is None:
                rates.append(evidence)
                continue
            # The clean copy is accessed only after the receive-only fits have run.
            ref = complex_tensor(extract(clean, rate)) / (phase*rms)
            methods = (value, *(value-.5*edit for edit, _ in fits))
            metrics = [torch.stack([fn(edit, ref, rms) for edit in methods], dim=1)
                       for fn in (absolute_error, private_error)]
            rates.append(torch.cat((evidence, *metrics), dim=1))
        rows.extend(torch.stack(rates, dim=-1).tolist())
    return np.asarray(rows)


def disturbance(raw, case):
    if case == 'clean':
        return raw
    noise = generate_noise(raw, 'white', seed=SEED)
    if case == 'white':
        return raw + .04*noise
    rng = np.random.default_rng(SEED+1)
    phi = rng.uniform(-np.pi, np.pi, (2, len(raw), 1))
    phase = np.exp(1j*phi).astype(np.complex64)
    large = case.endswith('strong')
    dose = np.float32(.12 if large else .04)
    out = raw + np.float32(.025 if large else .015)*noise
    if case.startswith(('mirror', 'joint')):
        out += dose*phase[0]*raw.conj()
    if case.startswith(('echo', 'joint')):
        offset = 13 if large else 7
        echo = np.pad(raw[:, :-offset], ((0, 0), (offset, 0)))
        out += dose*phase[1]*echo
    return out


@torch.no_grad()
def measure(source):
    numerical_policy()
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    _, calibration, held = partition(rx, fit=4000, held=400)
    held = held.reshape(len(SOURCE), 400)[:, 300:].ravel()
    trackers = {rate: PacketRestorer('consensus', samples=rate).eval() for rate in RATES}
    scores = panel(iq[calibration], trackers)
    pooled = 1.05*scores.max((0, 2))
    leave = {name: 1.05*scores[rx[calibration] != name].max((0, 2)) for name in SOURCE}
    raw = iq[held]

    def summary(matrix, threshold):
        on = matrix[:, :3] > threshold[None, :, None]
        absolute, private = matrix[:, 3:7], matrix[:, 7:11]
        out = dict(n=len(matrix), active=on.sum(0).tolist(),
                   held_explained=np.quantile(matrix[:, :3], (.1,.5,.9), axis=0).tolist())
        for key, block in (('physical', absolute), ('private_9_32', private)):
            base = block[:, :1]
            actual = np.where(on, block[:, 1:], base)
            reduction = 1-actual/np.maximum(base, 1e-12)
            out[key] = dict(direct_median=np.median(base[:, 0], axis=0).tolist(),
                            gated_median=np.median(actual, axis=0).tolist(),
                            paired_ungated_reduction=np.median(1-block[:, 1:]/np.maximum(base, 1e-12), axis=0).tolist(),
                            gated_median_reduction=np.median(reduction, axis=0).tolist(),
                            gated_worse=(actual > base+1e-9).sum(0).tolist(),
                            gated_better=(actual < base-1e-9).sum(0).tolist())
        return out

    results = {}
    for case in CASES:
        rows = panel(raw, trackers, disturbance(raw, case))
        if not np.isfinite(rows).all():
            raise FloatingPointError('Nonfinite public reference attribution')
        results[case] = dict(pooled=summary(rows, pooled),
                             leave_rx={name: summary(rows[rx[held] == name], leave[name]) for name in SOURCE})
    return dict(status='VERIFIED', role='U_s', protocol=contract['protocol'], rates_ksps=RATES, arms=ARMS,
                delays_us=DELAYS_US, case_seed=SEED, calibration=len(calibration), held=len(held),
                held_offset_per_rx=4800, pooled_threshold=pooled.tolist(), results=results,
                source_clean_only_after_fit=True, TX_reads=0, F_reads=0, V_reads=0, target_reads=0, trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, measure(args.source))
