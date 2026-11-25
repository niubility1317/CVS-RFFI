"""Isolate the signed covariance gate on the frozen expanded LoRa restorer."""

import argparse
from pathlib import Path
import sys


if __name__ == '__main__' and __spec__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_capacity_target as expanded, weak_guard_target as replay
from experiments.pamf_lora_main.dura_dual_aperture_evaluate import PER_RECEIVER, ROUTES, SEED, SEEDS


RUN = '20261125-phase1-lora-dura-signed-covariance-pilot-s2026092702-r01'
ORIGIN, ORIGIN_CODE = expanded.ORIGIN, expanded.ORIGIN_CODE
MODE = 'signed_covariance_half'
MODES = ('direct', MODE)
SOURCE_MODES = ('direct', 'dura_half', 'expanded_half', MODE)
SOURCE_COMPARISONS = tuple((MODE, name) for name in SOURCE_MODES[:-1])
EVENTS = 'LORA_SIGNED_COVARIANCE_PILOT'
THRESHOLD = 0.013684820942580701


def completion():
    return replay.completion(origin=ORIGIN, origin_code=ORIGIN_CODE, modes=MODES,
                             seeds=SEEDS, per_receiver=PER_RECEIVER)


def expected_files():
    return replay.expected_files(MODES, seeds=SEEDS)


def activation(value):
    from experiments.pamf_lora_main.dura_apertures import shared_guard_coefficients, signed_covariance_rms

    return signed_covariance_rms(shared_guard_coefficients(value))


def readouts(models, tracker, old_threshold):
    from experiments.pamf_lora_main.dura_capacity_source import READOUTS, recover

    half = READOUTS.index('half')

    def original(iq):
        output, active, _ = recover(iq, tracker, models, old_threshold)
        return output[half], active

    def candidate(iq):
        output, active, _ = recover(iq, tracker, models, THRESHOLD, activation_score=activation)
        return output[half], active

    return original, candidate


def bind_model(rt, device):
    from experiments.pamf_lora_main.dura_dual_aperture_evaluate import load_expanded
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_lora_main.runner import activate

    activate(rt)
    source = Path(rt['source_root'])
    project = Path(rt['project'])
    parent = read(project/'releases'/f'{ORIGIN}-release'/'runtime.json')
    if Path(parent['source_root']) != source or Path(parent['run_root']) != project/'runs'/ORIGIN:
        raise ValueError('Expanded DURA used a different source IQ contract or run root')
    contract = source_contract(source)
    return contract, load_expanded(rt, contract, device)


def bind(rt, device):
    from experiments.pamf_lora_main.complex_vector_source import load_reader
    from experiments.pamf_lora_main.dura_repair import PacketRestorer

    contract, (models, threshold) = bind_model(rt, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    _, candidate = readouts(models, tracker, threshold)
    return {SEED: load_reader(Path(rt['run_root']), contract, device)}, candidate, contract


def predict_source(rt, route):
    import torch
    from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
    from experiments.pamf_lora_main.complex_vector_source import score
    from experiments.pamf_lora_main.dura_dual_aperture_evaluate import bind_source_reference
    from experiments.pamf_lora_main.local_wiener_source import freeze

    device = 'cuda:0'
    contract, (models, threshold) = bind_model(rt, device)
    binding = bind_source_reference(rt, route, contract, device)
    original, candidate = readouts(models, binding[-1], threshold)

    def render(iq, _unused, base, _tracker, _models):
        out, active = candidate(iq)
        return torch.stack((iq, base(iq)[0], original(iq)[0], out)), active

    score(rt, route, panel=panel, modes=SOURCE_MODES, render=render, binding=binding,
          frozen_models=(), require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)


def score_source(rt):
    from experiments.pamf_lora_main.local_wiener_source import score_frozen

    score_frozen(rt, SOURCE_MODES, comparisons=SOURCE_COMPARISONS)
    original, candidate = (SOURCE_MODES.index(name) for name in ('expanded_half', MODE))
    rows = {}
    root = Path(rt['run_root'])
    for route in ROUTES:
        result = read(root/f'{route}-source-scores.json')
        six, clean = result['six'], result['metrics']['clean']['correct']
        rows[route] = dict(six_net=six[candidate]-six[original], clean_net=clean[candidate]-clean[original])
    write(root/'source-promotion.json', dict(reference_only=True, pairs=rows, target_reads=0))


def main():
    from experiments.pamf_lora_main.dura_dual_aperture_evaluate import source_panel

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    publisher = commands.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config'):
        publisher.add_argument('--'+flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'source-predict', 'source-score'):
        worker = commands.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('shard')
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        replay.publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if rt['run_id'] != RUN:
        raise ValueError('The signed covariance ablation requires its own immutable run')
    if action == 'dispatch':
        replay.dispatch(rt, args['runtime'], entrypoint=Path(__file__).resolve(), modes=MODES, complete=completion,
                        events=EVENTS, seeds=SEEDS, after_smoke=source_panel)
    elif action == 'source-predict':
        predict_source(rt, args['shard'])
    elif action == 'source-score':
        score_source(rt)
    else:
        replay.run_worker(rt, action, args['shard'], bind=bind, modes=MODES, events=EVENTS,
                          seeds=SEEDS, per_receiver=PER_RECEIVER, time_restoration=True)


if __name__ == '__main__':
    main()
