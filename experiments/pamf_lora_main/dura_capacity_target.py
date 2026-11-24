"""Replay the source-admitted LoRa DURA expanded half readout, truth-blind."""

import argparse
import json
from pathlib import Path
import sys


if __name__ == '__main__' and __spec__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pamf_lora_main import weak_guard_target as replay
from experiments.pamf_lora_main.weak_guard_target import ROUTES, SEEDS, matrix


RUN = '20261124-phase1-lora-dura-capacity-target-m8-r01'
ORIGIN = '20261124-phase1-lora-dura-capacity-source-m4-r01'
ORIGIN_CODE = '3de50180aaaa1179587a753d4f6dde201fc989c6'
MODES = ('direct', 'expanded_half')
EVENTS = 'LORA_CAPACITY_TARGET'


def completion():
    return replay.completion(origin=ORIGIN, origin_code=ORIGIN_CODE, modes=MODES)


def expected_files():
    return replay.expected_files(MODES)


def require_source_admission(root, source_modes):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.data import VIEWS

    if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('DURA capacity source evaluation must be complete before target access')
    direct, reference, candidate = (source_modes.index(mode) for mode in ('direct', 'reference', MODES[1]))
    for route in ROUTES:
        scores = read(root / f'{route}-source-scores.json')
        metrics = scores['metrics']
        if (scores['n'] != 900 or scores['modes'] != list(source_modes) or set(metrics) != set(VIEWS)
                or scores['target_reads'] != 0 or not scores['truth_after_predictions']):
            raise ValueError(f'Expected unchanged, truth-last source scores for {route}')
        clean = metrics['clean']['correct']
        gain = sum(metrics[view]['correct'][candidate]-metrics[view]['correct'][reference] for view in VIEWS[1:])
        if gain <= 0 or clean[candidate] < max(clean[direct], clean[reference]):
            raise ValueError(f'Fixed DURA capacity readout did not pass source admission for {route}')


def source_inference(rt, device):
    import torch
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.dura_capacity_source import MODES as source_modes
    from experiments.pamf_lora_main.dura_capacity_source import RECIPES, READOUTS, recover
    from experiments.pamf_lora_main.dura_repair import PacketRestorer
    from experiments.pamf_lora_main.full_adaptive import SET_METHOD, SET_RUN, case, complete_f
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_lora_main.residual_dura import recognizer
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.spectral_block_source import load_models

    project = Path(rt['project'])
    origin = read(project / 'releases' / f'{ORIGIN}-release' / 'runtime.json')
    if origin['run_id'] != ORIGIN or origin['code_commit'] != ORIGIN_CODE:
        raise ValueError('Unexpected source DURA capacity release')
    runs = project / 'runs'
    root = runs / ORIGIN
    require_source_admission(root, source_modes)
    activate(rt)
    contract = source_contract(Path(rt['source_root']))
    models, threshold = load_models(root, contract, device, {'expanded': RECIPES['expanded']})
    tracker = PacketRestorer('consensus').to(device).eval()
    readers = {}
    for seed in SEEDS:
        weights = complete_f(runs / SET_RUN, contract, case(seed, SET_METHOD)) / 'model.pt'
        reader = recognizer(device).eval().requires_grad_(False)
        reader.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
        readers[seed] = reader

    half = READOUTS.index('half')

    def restore(iq):
        output, active, _ = recover(iq, tracker, models, threshold)
        return output[half], active

    return readers, restore, contract


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate'):
        parser = actions.add_parser(action)
        parser.add_argument('runtime', type=Path)
        if action != 'dispatch':
            parser.add_argument('shard')
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        replay.publish(**args, run=RUN, runner=__spec__.name)
        return
    runtime = args['runtime'].resolve()
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    if rt['run_id'] != RUN:
        raise ValueError('The DURA capacity target run requires its own immutable output root')
    if action == 'dispatch':
        replay.dispatch(rt, runtime, entrypoint=Path(__file__).resolve(), modes=MODES, complete=completion,
                        events=EVENTS)
    else:
        replay.run_worker(rt, action, args['shard'], bind=source_inference, modes=MODES, events=EVENTS)


if __name__ == '__main__':
    main()
