"""Compare the two probe observations on a fixed unnamed held U_s subset."""

import argparse
import json
import sys
from pathlib import Path

import torch


RUN = '20261126-phase1-lora-w24-probe-transport-s2026112991-r01'
SEED = 2026112978


def check(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    if rt['run_id'] != RUN or rt['third_prior'] != 'probe-transport':
        raise ValueError('Expected the registered probe-transport run')

    sys.path.insert(0, rt['training_release_root'])
    from experiments import pamf_lora_main
    from experiments.cric_dura_comparison.io import tensor

    overlay = Path(rt['release_root']) / 'experiments/pamf_lora_main'
    pamf_lora_main.__path__.append(str(overlay))
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train
    from experiments.pamf_lora_main.w24_probe_transport import probe_views

    source = Path(rt['source_root'])
    contract = dura_w24_train.source_contract(source)
    iq, receiver, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
    _, _, held = channel_blind_source.partition(receiver)
    selected = held.reshape(5, -1)[:, :16].reshape(-1)
    tracker = dura_w24_train.WideTracker().eval()
    generators = [torch.Generator().manual_seed(SEED) for _ in range(2)]
    ratios = {'center_edit_change_fraction': [], 'center_probe_gain': [], 'native_edge_edit_change_fraction': []}
    with torch.no_grad():
        for ids in selected.reshape(-1, 16):
            raw = tensor(iq[ids].view('float32'), 'cpu').view(torch.complex64)
            raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            clean = tracker.aperture(raw)
            framed = dura_w24_model.coordinates(tracker, dura_w24_model.real_iq(clean[..., :384]))[0]
            weak = dura_w24_model.weak_difference(framed)
            (after, reference), (through, replay_reference) = (
                probe_views(raw, clean, tracker, dura_w24_model.condition_waveforms, rng, weak,
                            through_channel=channel, with_identity=False)
                for rng, channel in zip(generators, [False, True], strict=True)
            )
            if (not torch.equal(reference, replay_reference) or not torch.equal(after[:, :2], through[:, :2])
                    or not torch.equal(generators[0].get_state(), generators[1].get_state())):
                raise ValueError('The physical control failed to match the channel draws or clean reference')
            edit_control = after[:, 2] - after[:, 0]
            edit_observed = through[:, 2] - through[:, 0]
            difference = edit_observed - edit_control
            for key, value, reference, part in (
                ('center_edit_change_fraction', difference, edit_control, slice(0, 384)),
                ('center_probe_gain', edit_observed, edit_control, slice(0, 384)),
                ('native_edge_edit_change_fraction', difference, edit_control, slice(384, None)),
            ):
                norm = value[..., part].abs().square().sum((-2, -1)).sqrt()
                denominator = reference[..., part].abs().square().sum((-2, -1)).sqrt().clamp_min(1e-8)
                ratios[key].append(norm / denominator)
    return dict(status='MATCHED', n_source_held=len(selected), seed=SEED, matching_base=True,
                quantiles=[.1, .5, .9],
                **{name: torch.quantile(torch.cat(values), torch.tensor([.1, .5, .9])).tolist()
                   for name, values in ratios.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    torch.set_num_threads(2)
    print(json.dumps(check(parser.parse_args().runtime)))
