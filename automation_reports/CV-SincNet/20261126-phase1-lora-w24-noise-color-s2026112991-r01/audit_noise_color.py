"""Measure the realized covariance change on fixed held source packets."""

import argparse
import json
import sys
from pathlib import Path

import torch


RUN = '20261126-phase1-lora-w24-noise-color-s2026112991-r01'
SEED = 2026112984


def lag_one(noise):
    numerator = (noise[..., 1:] * noise[..., :-1].conj()).mean((-2, -1)).abs()
    powers = [x.abs().square().mean((-2, -1)) for x in (noise[..., 1:], noise[..., :-1])]
    return numerator / (powers[0] * powers[1]).sqrt().clamp_min(1e-10)


def audit(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    if rt['run_id'] != RUN or rt['third_prior'] != 'noise-color':
        raise ValueError('Expected the registered source-only noise-color run')
    sys.path.insert(0, rt['training_release_root'])
    from experiments import pamf_lora_main
    from experiments.cric_dura_comparison.io import tensor

    pamf_lora_main.__path__.append(str(Path(rt['release_root']) / 'experiments/pamf_lora_main'))
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train
    from experiments.pamf_lora_main.w24_noise_color import noise_views
    from experiments.pamf_lora_main.w24_white_corruption import center_power

    source = Path(rt['source_root'])
    contract = dura_w24_train.source_contract(source)
    iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
    _, _, held = channel_blind_source.partition(rx)
    selected = held.reshape(5, -1)[:, :16]
    generators = [torch.Generator().manual_seed(SEED) for _ in range(2)]
    tracker = dura_w24_train.WideTracker().eval()
    lag_white, lag_color, relative, identical = [], [], [], 0
    with torch.no_grad():
        for ids in selected:
            raw = tensor(iq[ids].view('float32'), 'cpu').view(torch.complex64)
            raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            clean = tracker.aperture(raw)
            white, color = (noise_views(raw, clean, tracker.aperture, dura_w24_model.condition_waveforms, rng,
                                        colored=colored)
                            for colored, rng in zip((False, True), generators, strict=True))
            if not torch.equal(generators[0].get_state(), generators[1].get_state()):
                raise ValueError('The fit data streams diverged')
            for a, b in zip(white, color, strict=True):
                different = (a - b).abs().amax((-2, -1)) > 1e-7
                identical += int((~different).sum())
                energy_a, energy_b = center_power(a - clean), center_power(b - clean)
                relative.append(((energy_a - energy_b).abs() / energy_a.clamp_min(1e-12)).flatten())
                lag_white.append(lag_one((a - clean)[different, :, :384]))
                lag_color.append(lag_one((b - clean)[different, :, :384]))
    max_rel = float(torch.cat(relative).max())
    if max_rel > 5e-6:
        raise ValueError('The actual W24 center disturbance powers did not match')
    return dict(status='MATCHED', seed=SEED, n_packets=int(selected.size), n_views=int(2 * selected.size),
                identical_views=identical, max_relative_center_power_difference=max_rel,
                modified_lag_abs_median_white=float(torch.cat(lag_white).median()),
                modified_lag_abs_median_colored=float(torch.cat(lag_color).median()))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(audit(cli.parse_args().runtime)))
