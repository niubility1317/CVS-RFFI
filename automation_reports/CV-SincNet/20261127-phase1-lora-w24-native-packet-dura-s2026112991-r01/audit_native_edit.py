"""Audit native full-packet corrections and their W24 coordinates after source scoring."""
import argparse
import json
import sys
from pathlib import Path

import torch


RUN = '20261127-phase1-lora-w24-native-packet-dura-s2026112991-r01'


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
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train
    from experiments.pamf_lora_main.w24_packet_dura import NativePacketDura

    model = NativePacketDura().eval()
    checkpoint = root / 'native_dura_whole/w24_native_dura_whole-model.pt'
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if payload['proof']['steps'] != 24000:
        raise ValueError('Expected native DURA fixed final weights')
    model.load_state_dict(payload['weights'], strict=True)

    source = Path(rt['source_root'])
    contract = dura_w24_train.source_contract(source)
    iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
    _, _, hold = channel_blind_source.partition(rx)
    rows = hold.reshape(5,-1)[:, :8]
    tracker = dura_w24_train.WideTracker().eval()
    rng = torch.Generator().manual_seed(2026112996)
    values = {'clean': [], 'generic': []}
    with torch.inference_mode():
        for selection in rows:
            raw = tensor(iq[selection].view('float32'), 'cpu').view(torch.complex64)
            raw = raw / rms(raw, -1).clamp_min(1e-8)[:, None]
            disturbed = dura_w24_model.condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2)
            for scene, y in (('clean', raw), ('generic', disturbed)):
                restored = torch.lerp(y, model.restore_native(y), .5)
                error = restored - y
                native = rms(error, -1) / rms(y, -1).clamp_min(1e-9)
                aperture = tracker.aperture(y)
                corrected = tracker.aperture(restored)
                yscale = rms(aperture[..., :384], (-2,-1)).clamp_min(1e-9)
                zscale = rms(corrected[..., :384], (-2,-1)).clamp_min(1e-9)
                delta = corrected / zscale[:, None, None] - aperture / yscale[:, None, None]
                center = rms(delta[..., :384], (-2,-1))
                edges = rms(delta[..., 384:], (-2,-1))
                r = [native, center, edges]
                if scene == 'generic':
                    ratio = rms(restored-raw, -1) / rms(y-raw, -1).clamp_min(1e-9)
                    r.append(ratio.square())
                values[scene].append(torch.stack(r, dim=-1))
    names = ('native_relative_edit', 'w24_center_edit_rms', 'w24_edge_edit_rms', 'native_remaining_disturbance_power')
    summary = {s: dict(zip(names[:row[0].shape[-1]], torch.cat(row).median(0).values.tolist(), strict=True))
               for s, row in values.items()}
    return dict(status='VERIFIED', n_source_packets=int(rows.size), medians=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(audit(parser.parse_args().runtime)))
