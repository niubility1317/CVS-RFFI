"""Retrain expanded LoRa DURA with a source-energy-balanced weak difference loss."""

from dataclasses import replace
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.dura_capacity import objective
from experiments.pamf_lora_main.dura_capacity_source import READOUTS, RECIPES, recover
from experiments.pamf_lora_main.dura_dual_aperture_evaluate import SEED, bind_source_controls
from experiments.pamf_lora_main.spectral_block_source import load_models, main


RUN = '20261125-phase1-lora-dura-balanced-transfer-source-s2026112991-r01'
MODE = 'balanced_transfer_half'
RECIPE = replace(RECIPES['expanded'], name='balanced_transfer')
ARMS = {RECIPE.name: RECIPE}
MODES = ('direct', 'dura_half', 'expanded_half', MODE)
COMPARISONS = tuple((MODE, control) for control in MODES[:-1])


def bind_restoration(rt, device):
    from experiments.pamf_lora_main.dura_repair import PacketRestorer
    from experiments.pamf_lora_main.full_packet_views import source_contract

    root = Path(rt['run_root'])
    contract = source_contract(Path(rt['source_root']))
    frozen = read(root/f'{RECIPE.name}-complete.json')
    if (frozen['status'], frozen['steps'], frozen['estimator']) != ('SOURCE_FROZEN', RECIPE.steps, RECIPE.name):
        raise ValueError('The balanced-transfer pilot requires the fixed terminal source model')
    models, threshold = load_models(root, contract, device, ARMS)
    if frozen['threshold'] != threshold:
        raise ValueError('Source-only calibration differs from the frozen checkpoint')
    tracker = PacketRestorer('consensus').to(device).eval()

    def restore(iq):
        waves, active, _ = recover(iq, tracker, models, threshold)
        return waves[READOUTS.index('half')], active

    return contract, restore


def bind_reader(rt, device):
    from experiments.pamf_lora_main.complex_vector_source import load_reader

    contract, restore = bind_restoration(rt, device)
    reader = load_reader(Path(rt['run_root']), contract, device)
    return {SEED: reader}, restore, contract


def smoke(rt, _recipes, _modes):
    from experiments.pamf_lora_main import weak_guard_target as replay

    replay.run_worker(rt, 'smoke', 'source', bind=bind_reader, modes=('direct', MODE),
                      seeds=(SEED,), time_restoration=True)


def evaluate(rt, route, _recipes, modes):
    import torch
    from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
    from experiments.pamf_lora_main.complex_vector_source import score
    from experiments.pamf_lora_main.local_wiener_source import freeze

    device = 'cuda:0'
    contract, restore = bind_restoration(rt, device)
    binding, original = bind_source_controls(rt, route, contract, device)

    def render(iq, _unused, baseline, _tracker, _models):
        dura, _ = baseline(iq)
        old = original(iq)
        candidate, active = restore(iq)
        return torch.stack((iq, dura, old, candidate)), active

    score(rt, route, panel=panel, modes=modes, render=render, binding=binding, frozen_models=(),
          require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=ARMS, modes=MODES, comparisons=COMPARISONS,
         calculate=partial(objective, balance_transfer=True), run_smoke=smoke, run_evaluate=evaluate)
