"""Recount frozen native correction errors on an isolated Full source U replay."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


RUN = '20261127-phase1-lora-w24-native-band-risk-s2026112991-r01'
ARMS = ('native_band_control', 'native_band_risk')
SCENES = ('practical_high', 'practical_mid_urban', 'practical_low_urban')
NAMESPACE = 'source-U-native-band-risk-diagnostic'


def measures(noise, edit, frame):
    def spectrum(value):
        return torch.fft.fft(value.reshape(-1, 8, 1024) / frame, norm='ortho')

    def components(value):
        shared = value.mean(-2, keepdim=True)
        return torch.stack((shared.expand_as(value), value - shared), dim=1)

    removed, injected = (components(spectrum(value)) for value in (-edit, noise))
    fft_bins = (torch.fft.fftfreq(1024).abs() * 1024).to(noise.device)
    results = {}
    for low, high in ((1, 8), (9, 32), (33, 64)):
        mask = (fft_bins >= low) & (fft_bins <= high)
        a, b = removed[..., mask], injected[..., mask]
        a2, b2 = (x.abs().square().sum((-2, -1)) for x in (a, b))
        cosine = (a * b.conj()).real.sum((-2, -1)) / (a2 * b2).sqrt().clamp_min(1e-12)
        left = (b - a).abs().square().sum((-2, -1)) / b2.clamp_min(1e-12)
        for i, kind in enumerate(('common', 'private')):
            results[f'{kind}_{low}_{high}'] = torch.stack((cosine[:, i], left[:, i]), dim=-1)
    numerator = (noise + edit).abs().square().sum(-1)
    results['whole_remaining'] = (numerator / noise.abs().square().sum(-1).clamp_min(1e-12))[:, None]
    return results


@torch.inference_mode()
def audit(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    root = Path(rt['run_root'])
    score = json.loads((root / 'full_native/source-scores.json').read_text(encoding='utf-8'))
    if (rt['run_id'] != RUN or rt['third_prior'] != 'native-band-risk'
            or not score['source_truth_last'] or score['target_reads']):
        raise ValueError('Expected the registered frozen source run')
    sys.path.insert(0, rt['w24_release_root'])
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(rt['release_root']) / 'experiments/pamf_lora_main'))
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_lora_main.full_packet_views import native, source_contract
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.w24_factor_audit import trace_components
    from experiments.pamf_lora_main.w24_native_full import SOURCE, native_full_view
    from experiments.pamf_lora_main.w24_packet_dura import NativePacketDura

    activate(rt)

    models = {}
    fit = {}
    for arm in ARMS:
        model = NativePacketDura().eval()
        payload = torch.load(root / arm / f'w24_{arm}-model.pt', map_location='cpu', weights_only=True)
        if payload['proof']['steps'] != 24000:
            raise ValueError('Expected final frozen native weights')
        model.load_state_dict(payload['weights'], strict=True)
        models[arm] = model
        proof = payload['proof']
        fit[arm] = (proof['seed'], proof['fit_per_rx'], proof['calibration_per_rx'], proof['held_per_rx'])

    source = Path(rt['source_root'])
    contract = source_contract(source)
    physical_ids = contract['role_ids']['U_s']
    iq = native(source, 'U_s', contract['counts']['U_s'])
    groups = {rx: [i for i, name in enumerate(physical_ids) if name.startswith(rx + '/')]
              for rx in SOURCE}
    if len(set(fit.values())) != 1:
        raise ValueError('The two weights do not have a matched source partition')
    seed, n_fit, n_cal, n_held = fit[ARMS[0]]
    rng = np.random.default_rng(seed)
    held = {rx: rng.permutation(indices)[n_fit+n_cal:n_fit+n_cal+n_held]
            for rx, indices in groups.items()}
    rows = [(rx, i) for rx, indices in held.items() for i in indices[:8]]
    if len(rows) != 40:
        raise ValueError('Expected eight fixed U_s traces per source RX')
    results = {}
    for scene in SCENES:
        fields = []
        for rx, i in rows:
            args = (iq[i], scene, rx, i, NAMESPACE)
            y, _, disturbance = trace_components(*args)
            if not fields and not np.array_equal(y, native_full_view(*args)):
                raise ValueError('Trace replay changed the registered Full received waveform')
            fields.append((y, disturbance))
        rx_wave, noise = [tensor(np.asarray(a, np.complex64).view(np.float32)).view(torch.complex64)
                          for a in zip(*fields, strict=True)]
        results[scene] = {}
        for arm, model in models.items():
            stages, frame = model(rx_wave)
            edit = torch.lerp(rx_wave, stages[-1], .5) - rx_wave
            values = measures(noise, edit, frame)
            values['native_relative_edit'] = (
                edit.norm(dim=-1) / rx_wave.norm(dim=-1).clamp_min(1e-12))[:, None]
            results[scene][arm] = {k: torch.quantile(v, .50, dim=0).tolist() for k, v in values.items()}
    return dict(status='SOURCE_DIAGNOSTIC', namespace=NAMESPACE, n_source_packets=len(rows),
                band_metric_order=['direction_cosine', 'remaining_noise_power_ratio'], by_scene=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(audit(parser.parse_args().runtime)))
