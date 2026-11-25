"""Evaluate fixed local DURA variance on the registered LoRa receive views."""

import argparse
from pathlib import Path
import sys

if __name__ == '__main__' and __spec__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_capacity_target as expanded, weak_guard_target as replay
from experiments.pamf_lora_main.dura_dual_aperture_evaluate import (PER_RECEIVER, ROUTES, SEED, SEEDS,
                                                                 bind_source_reference, load_expanded, source_panel)


RUN = '20261125-phase1-lora-dura-local-variance-pilot-s2026092702-r01'
MODE = 'local_variance_half'
MODES = ('direct', MODE)
SOURCE_MODES = ('direct', 'dura_half', 'expanded_half', MODE, 'shifted_variance_half')
SOURCE_COMPARISONS = tuple((MODE, control) for control in SOURCE_MODES if control != MODE)
EVENTS = 'LORA_LOCAL_VARIANCE_PILOT'
FROZEN = dict(status='SOURCE_FROZEN', checkpoint=expanded.ORIGIN, checkpoint_commit=expanded.ORIGIN_CODE,
              method=MODE, fixed_half_dose=True, optimizer_updates=0, threshold_updates=0,
              F_reads=0, V_reads=0, TX_reads=0, target_reads=0, trace_reads=0)


def completion():
    return replay.completion(origin=expanded.ORIGIN, origin_code=expanded.ORIGIN_CODE, modes=MODES,
                             seeds=SEEDS, per_receiver=PER_RECEIVER)


def expected_files():
    return replay.expected_files(MODES, seeds=SEEDS)


def freeze_rule(root, _state):
    write(root/'profile.json', FROZEN)


def bind_restoration(rt, device):
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_lora_main.runner import activate

    root = Path(rt['run_root'])
    if read(root/'profile.json') != FROZEN:
        raise ValueError('Expected the same frozen local variance rule on both routes')
    activate(rt)
    contract = source_contract(Path(rt['source_root']))
    (model,), threshold = load_expanded(rt, contract, device)
    return contract, model, threshold


def bind_target(rt, device):
    from experiments.pamf_lora_main.complex_vector_source import load_reader
    from experiments.pamf_lora_main.dura_repair import PacketRestorer
    from experiments.pamf_lora_main.dura_variance_profile import restore

    contract, model, threshold = bind_restoration(rt, device)
    tracker = PacketRestorer('consensus').to(device).eval()

    def restore_local(iq):
        return restore(iq, tracker, model, threshold)

    return {SEED: load_reader(Path(rt['run_root']), contract, device)}, restore_local, contract


def predict_source(rt, route):
    import torch
    from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
    from experiments.pamf_lora_main.complex_vector_source import score
    from experiments.pamf_lora_main.dura_capacity_source import READOUTS, recover
    from experiments.pamf_lora_main.dura_variance_profile import restore
    from experiments.pamf_lora_main.local_wiener_source import freeze

    device = 'cuda:0'
    contract, model, threshold = bind_restoration(rt, device)
    binding = bind_source_reference(rt, route, contract, device)

    def render(iq, _unused, baseline, tracker, _models):
        base, _ = baseline(iq)
        old, count, _ = recover(iq, tracker, (model,), threshold)
        local, active = restore(iq, tracker, model, threshold)
        shifted, shifted_active = restore(iq, tracker, model, threshold, shifted=True)
        if active != count or shifted_active != count:
            raise ValueError('All expanded DURA variants must use the same received gate')
        return torch.stack((iq, base, old[READOUTS.index('half')], local, shifted)), count

    score(rt, route, panel=panel, modes=SOURCE_MODES, render=render, binding=binding, frozen_models=(),
          require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)


def score_source(rt):
    from experiments.pamf_lora_main.local_wiener_source import score_frozen

    score_frozen(rt, SOURCE_MODES, comparisons=SOURCE_COMPARISONS)
    root = Path(rt['run_root'])
    pairs = {}
    for route in ROUTES:
        result = read(root/f'{route}-source-scores.json')
        candidate = result['paired_comparisons'][f'{MODE}_vs_expanded_half']
        clean = candidate['views']['clean']
        pairs[route] = dict(six_net=sum(candidate['six_net_by_rx']), clean_net=clean['repairs']-clean['harms'])
    write(root/'source-promotion.json', dict(pairs=pairs, reference_only=True, target_reads=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    publish = actions.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config'):
        publish.add_argument('--' + flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'source-predict', 'source-score'):
        command = actions.add_parser(action)
        command.add_argument('runtime', type=Path)
        if action != 'dispatch':
            command.add_argument('row')
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        replay.publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if rt['run_id'] != RUN:
        raise ValueError('Expected the immutable local variance evaluation root')
    if action == 'dispatch':
        replay.dispatch(rt, args['runtime'], entrypoint=Path(__file__).resolve(), modes=MODES,
                        complete=completion, events=EVENTS, seeds=SEEDS, before_smoke=freeze_rule,
                        after_smoke=source_panel)
    elif action == 'source-predict':
        predict_source(rt, args['row'])
    elif action == 'source-score':
        score_source(rt)
    else:
        replay.run_worker(rt, action, args['row'], bind=bind_target, modes=MODES, events=EVENTS,
                          seeds=SEEDS, per_receiver=PER_RECEIVER, time_restoration=True)


if __name__ == '__main__':
    main()
