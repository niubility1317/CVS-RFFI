"""Test frozen receive-only LoRa observables on the matched practical source V IQ."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_apertures import shared_guard_coefficients
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from tools.lora_public_reference_probe import ARMS, absolute_error, fit_shared, private_error, public_columns
from tools.lora_receive_frame_practical_evidence import source_panel, view_pairs
from tools.lora_w24_signed_guard_probe import guard_scores


MODES = ('shared_outer', 'balanced_outer', 'signed_covariance', *ARMS)
THRESHOLDS = np.asarray((0.02941850256174803, 0.015004556067287922, 0.013684820942580701,
                         9.628292173147203e-06, 0.009021071204915644, 0.009003243315964938))
QUANTILES = (.1, .5, .9)


@torch.no_grad()
def measure(observation, clean, tracker):
    rows = []
    for wave, ref in zip(tensor(observation).split(32), tensor(clean).split(32), strict=True):
        value, phase, _, rms = coordinates(tracker, wave)
        scores = guard_scores(shared_guard_coefficients(value))[:, (0, 1, 3)]
        columns = public_columns(phase)
        fits = [fit_shared(basis, value, absolute=True) for basis in (columns[..., :1], columns[..., 1:], columns)]
        fitted_scores = torch.stack([evidence for _, evidence in fits], dim=1)

        # Source clean IQ becomes available to the diagnostic only after the fits.
        reference = complex_iq(ref) / (phase*rms)
        recovered = (value, *(value-.5*edit for edit, _ in fits))
        fidelity = [torch.stack([metric(edit, reference, rms) for edit in recovered], dim=1)
                    for metric in (absolute_error, private_error)]
        rows.extend(torch.cat((scores, fitted_scores, *fidelity), dim=1).tolist())
    values = np.asarray(rows)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite LoRa practical observation metric')
    return values


def describe(values):
    scores = values[:, :len(MODES)]
    active = scores > THRESHOLDS
    out = dict(n=len(values), active=dict(zip(MODES, active.sum(0).tolist(), strict=True)),
               score_quantiles=dict(zip(MODES, np.quantile(scores, QUANTILES, axis=0).T.tolist(), strict=True)),
               signed_only=int(np.count_nonzero(active[:, 2] & ~active[:, 0])),
               outer_only=int(np.count_nonzero(active[:, 0] & ~active[:, 2])))
    for key, block in (('physical', values[:, 6:10]), ('private_9_32', values[:, 10:14])):
        raw = block[:, :1]
        delta = raw-block[:, 1:]
        by_arm = {}
        for index, arm in enumerate(ARMS):
            mask = active[:, index+3]
            selected = delta[mask, index]
            by_arm[arm] = dict(
                gated_nmse_median=float(np.median(np.where(mask, block[:, index+1], raw[:, 0]))),
                better=int((selected > 1e-9).sum()), worse=int((selected < -1e-9).sum()),
                median_active_relative_reduction=float(np.median(selected/np.maximum(raw[mask, 0], 1e-12)))
                if len(selected) else None,
                median_active_absolute_reduction=float(np.median(selected)) if len(selected) else None)
        out[key] = dict(direct_nmse_median=float(np.median(raw)), arms=by_arm)
    return out


def audit(rt):
    start = perf_counter()
    numerical_policy()
    ids, selected_rx, captures, cached, chosen = source_panel(rt)
    tracker = PacketRestorer('consensus').eval()
    raw = cached[0, chosen]
    rows = {'full': {}, 'residual': {}}
    for view, full, residual in view_pairs(captures, cached, chosen):
        rows['full'][view] = measure(full, raw, tracker)
        rows['residual'][view] = rows['full'][view] if view == 'clean' else measure(residual, raw, tracker)

    results = {}
    for route, views in rows.items():
        six = np.stack([views[view] for view in VIEWS[1:]])
        results[route] = dict(six=describe(six.reshape(-1, six.shape[-1])), views={
            key: dict(all=describe(w), by_rx={rx: describe(w[selected_rx == rx]) for rx in SOURCE})
            for key, w in views.items()}, receiver_six={
                rx: describe(six[:, selected_rx == rx].reshape(-1, six.shape[-1])) for rx in SOURCE})
    return dict(status='VERIFIED', role='source_V_IQ_only', modes=MODES, quantiles=QUANTILES,
                thresholds=dict(zip(MODES, THRESHOLDS.tolist(), strict=True)), ids=ids.tolist(), routes=results,
                base_release=rt['code_commit'], elapsed_seconds=perf_counter()-start,
                clean_reference_only_after_estimation=True, F_reads=0, TX_reads=0, target_reads=0,
                trace_reads=0, optimizer_updates=0, calibration_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    write(args.output, audit(read(args.runtime)))
