"""Verify actual posterior edits on an independent held source U_s subset."""

import argparse
import json
import sys
from pathlib import Path

import torch


RUN = '20261126-phase1-lora-w24-spectral-posterior-s2026112991-r01'
PARENT = '20261126-phase1-lora-w24-packet-unet-fidelity-s2026112991-r01'
ARMS = ('posterior_all_bins', 'posterior_exclude_bin')


def power(x):
    return x.abs().square().mean((-2, -1))


def audit(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    root = Path(rt['run_root'])
    if (rt['run_id'] != RUN or rt['third_prior'] != 'spectral-posterior'
            or not (root / 'full_native/source-scores.json').is_file()):
        raise ValueError('Expected the independently scored spectral posterior')
    sys.path.insert(0, rt['training_release_root'])
    from experiments import pamf_lora_main
    from experiments.cric_dura_comparison.io import tensor

    pamf_lora_main.__path__.append(str(Path(rt['release_root']) / 'experiments/pamf_lora_main'))
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train
    from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
    from experiments.pamf_lora_main.w24_spectral_posterior import SpectralPosterior

    models = {'original_unet': PacketUNet(across_chirps=True)}
    models['original_unet'].load_state_dict(torch.load(
        root.parent / PARENT / 'unet_standard_long/w24_unet_standard_long-model.pt',
        weights_only=True, map_location='cpu')['weights'], strict=True)
    for mask, name in enumerate(ARMS):
        model = SpectralPosterior(exclude_current=bool(mask))
        checkpoint = root / name / f'w24_{name}-model.pt'
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['weights'],
                              strict=True)
        models[name] = model
    for model in models.values():
        model.eval()

    source = Path(rt['source_root'])
    contract = dura_w24_train.source_contract(source)
    iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
    _, _, held = channel_blind_source.partition(rx)
    selected = held.reshape(5, -1)[:, :8]
    tracker = dura_w24_train.WideTracker().eval()
    rng = torch.Generator().manual_seed(2026112996)
    fields = ('edit_rms_to_center_rms', 'common_edit_power_fraction', 'remaining_generic_disturbance_power')
    records = {name: {'clean': [], 'generic': []} for name in models}
    with torch.inference_mode():
        for ids in selected:
            raw = tensor(iq[ids].view('float32'), 'cpu').view(torch.complex64)
            raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            clean = tracker.aperture(raw)
            generic = tracker.aperture(dura_w24_model.condition_waveforms(
                raw.unflatten(-1, (8, 1024)), rng).flatten(-2))
            for scene, observed in (('clean', clean), ('generic', generic)):
                y, phase, quality, rms = dura_w24_model.coordinates(
                    tracker, dura_w24_model.real_iq(observed[..., :384]))
                target = clean[..., :384] - observed[..., :384]
                for name, model in models.items():
                    aligned = .5 * (model(y, quality, rms, observed[..., 384:])[-1] - y) * rms
                    physical = aligned * phase
                    edit_power = power(aligned)
                    relative = (edit_power / power(observed[..., :384]).clamp_min(1e-12)).sqrt()
                    common = power(aligned.mean(-2, keepdim=True)) / edit_power.clamp_min(1e-12)
                    error = power(target - physical) / power(target).clamp_min(1e-12)
                    terms = (relative, common, error) if scene == 'generic' else (relative, common)
                    records[name][scene].append(torch.stack(terms, dim=-1))
    summary = {name: {scene: dict(zip(fields[:rows[0].shape[-1]], torch.cat(rows).median(0).values.tolist(),
                                      strict=True)) for scene, rows in scenes.items()}
               for name, scenes in records.items()}
    return dict(status='VERIFIED', n_source_packets=int(selected.size), seed=2026112996, medians=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(audit(parser.parse_args().runtime)))
