"""Verify frozen ancestry and measure the learned common update on held source IQ."""

import argparse
import json
import sys
from pathlib import Path

import torch


RUN = '20261126-phase1-lora-w24-common-refine-s2026112991-r01'
PARENT = '20261126-phase1-lora-w24-packet-unet-fidelity-s2026112991-r01'
ARMS = ('common_refine_control', 'common_refine_bands')


def audit(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    if rt['run_id'] != RUN or rt['third_prior'] != 'common-refine':
        raise ValueError('Expected the frozen common-refinement candidate')
    root = Path(rt['run_root'])
    state = json.loads((root / 'state.json').read_text(encoding='utf-8'))
    if state['status'] not in ('SOURCE_SCORED', 'FORMAL_PREDICTING', 'PREDICTIONS_FROZEN', 'FAILED'):
        raise ValueError('Wait for independent source scoring')
    if not (root / 'full_native/source-scores.json').is_file():
        raise ValueError('Missing independently scored frozen source outputs')
    sys.path.insert(0, rt['training_release_root'])
    from experiments import pamf_lora_main
    from experiments.cric_dura_comparison.io import tensor

    pamf_lora_main.__path__.append(str(Path(rt['release_root']) / 'experiments/pamf_lora_main'))
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train
    from experiments.pamf_lora_main.w24_common_refiner import CommonRefiner

    parent = root.parent / PARENT / 'unet_standard_long/w24_unet_standard_long-model.pt'
    base = torch.load(parent, weights_only=True, map_location='cpu')['weights']
    models = {}
    for name in ARMS:
        weights = torch.load(root / name / f'w24_{name}-model.pt', weights_only=True, map_location='cpu')['weights']
        copied = {key.removeprefix('base.'): value for key, value in weights.items() if key.startswith('base.')}
        if copied.keys() != base.keys() or any(not torch.equal(base[k], copied[k]) for k in base):
            raise ValueError(f'The frozen inherited U-Net changed in {name}')
        model = CommonRefiner().eval()
        model.load_state_dict(weights, strict=True)
        models[name] = model

    source = Path(rt['source_root'])
    contract = dura_w24_train.source_contract(source)
    iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
    _, _, held = channel_blind_source.partition(rx)
    selected = held.reshape(5, -1)[:, :8]
    tracker = dura_w24_train.WideTracker().eval()
    rng = torch.Generator().manual_seed(2026112994)
    values = {a: {scene: [] for scene in ('clean', 'generic')} for a in ARMS}
    with torch.inference_mode():
        freq = (torch.fft.fftfreq(384) * 384).round().abs()
        for ids in selected:
            raw = tensor(iq[ids].view('float32'), 'cpu').view(torch.complex64)
            raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            distorted = dura_w24_model.condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2)
            for scene, wave in (('clean', raw), ('generic', distorted)):
                x = tracker.aperture(wave)
                y, _, quality, rms = dura_w24_model.coordinates(tracker, dura_w24_model.real_iq(x[..., :384]))
                scale = rms.square().mean((-2, -1)).clamp_min(1e-12)
                for name, model in models.items():
                    original = model.base(y, quality, rms, x[..., 384:])[-1]
                    corrected = model(y, quality, rms, x[..., 384:])[-1]
                    update = .5 * (corrected - original) * rms
                    mean = update.mean(-2, keepdim=True)
                    if (update - mean).abs().max() > 2e-6:
                        raise ValueError('A common refiner changed the aligned private contrast')
                    field = torch.fft.fft(mean.squeeze(-2), norm='ortho')
                    near_power = field[..., freq <= 32].abs().square().sum(-1) / (384 * scale)
                    far_power = field[..., freq > 32].abs().square().sum(-1) / (384 * scale)
                    total = (near_power + far_power).sqrt()
                    if total.max() > .220001:
                        raise ValueError('Common refinement exceeded the registered cap')
                    values[name][scene].append(torch.stack((total, near_power.sqrt(), far_power.sqrt()), dim=-1))
    return dict(status='VERIFIED', backbone_identical=True, n_source_packets=int(selected.size),
                rms_fields=['total', 'near', 'far'],
                median_relative_edit={a: {scene: torch.cat(rows).median(0).values.tolist()
                                          for scene, rows in by_scene.items()} for a, by_scene in values.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(audit(parser.parse_args().runtime)))
