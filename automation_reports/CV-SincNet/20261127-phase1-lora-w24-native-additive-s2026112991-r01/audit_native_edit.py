"""Audit native full-packet corrections and their W24 coordinates after source scoring."""
import argparse
import json
import sys
from pathlib import Path

import torch


RUN = '20261127-phase1-lora-w24-native-additive-s2026112991-r01'
ARMS = ('native_restore_original', 'native_remove_additive')


def rms(x, dims):
    return x.abs().square().mean(dims).sqrt()


def audit(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    root = Path(rt['run_root'])
    if rt['run_id'] != RUN or not (root / 'full_native/source-scores.json').exists():
        raise ValueError('Expected completed frozen Full source scoring')
    sys.path.insert(0, rt['training_release_root'])
    from experiments import pamf_lora_main
    from experiments.cric_dura_comparison.io import tensor

    pamf_lora_main.__path__.append(str(Path(rt['release_root']) / 'experiments/pamf_lora_main'))
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_train
    from experiments.pamf_lora_main.w24_packet_dura import NativePacketDura, native_additive_pair

    models = {}
    for arm in ARMS:
        model = NativePacketDura().eval()
        checkpoint = root / arm / f'w24_{arm}-model.pt'
        payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if payload['proof']['steps'] != 24000:
            raise ValueError('Expected both fixed final native checkpoints')
        model.load_state_dict(payload['weights'], strict=True)
        models[arm] = model

    source = Path(rt['source_root'])
    contract = dura_w24_train.source_contract(source)
    iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
    _, _, hold = channel_blind_source.partition(rx)
    rows = hold.reshape(5,-1)[:, :8]
    tracker = dura_w24_train.WideTracker().eval()
    rng = torch.Generator().manual_seed(2026112996)
    values = {arm: {'clean': [], 'generic': []} for arm in ARMS}
    with torch.inference_mode():
        for selection in rows:
            raw = tensor(iq[selection].view('float32'), 'cpu').view(torch.complex64)
            disturbed, (original, after_action) = native_additive_pair(raw, rng, with_identity=False)
            for scene, y in (('clean', original), ('generic', disturbed)):
                aperture = tracker.aperture(y)
                yscale = rms(aperture[..., :384], (-2, -1)).clamp_min(1e-9)
                for arm, model in models.items():
                    restored = torch.lerp(y, model.restore_native(y), .5)
                    native = rms(restored - y, -1) / rms(y, -1).clamp_min(1e-9)
                    corrected = tracker.aperture(restored)
                    zscale = rms(corrected[..., :384], (-2, -1)).clamp_min(1e-9)
                    delta = corrected / zscale[:, None, None] - aperture / yscale[:, None, None]
                    row = [native, rms(delta[..., :384], (-2, -1)), rms(delta[..., 384:], (-2, -1))]
                    if scene == 'generic':
                        additive = y - after_action
                        remaining = rms(restored - after_action, -1) / rms(additive, -1).clamp_min(1e-9)
                        change = y - restored
                        numerator = (change * additive.conj()).real.sum(-1)
                        cosine = numerator / (change.norm(dim=-1) * additive.norm(dim=-1)).clamp_min(1e-9)
                        row.extend((remaining.square(), cosine))
                    values[arm][scene].append(torch.stack(row, dim=-1))
    names = ('native_relative_edit', 'w24_center_edit_rms', 'w24_edge_edit_rms',
             'native_remaining_additive_power', 'removed_additive_cosine')
    summary = {arm: {scene: dict(zip(names[:row[0].shape[-1]],
                                     torch.cat(row).median(0).values.tolist(), strict=True))
                     for scene, row in scenes.items()} for arm, scenes in values.items()}
    return dict(status='VERIFIED', n_source_packets=int(rows.size), medians=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(audit(parser.parse_args().runtime)))
