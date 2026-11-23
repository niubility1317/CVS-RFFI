"""Test a LoRa training constraint for preserving private complex differences."""

import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind_source import FitRecipe, fit, panel, receive_only
from .chirp_spectrum_source import freeze
from .complex_vector_source import publish, score
from .crossfit_pilot import paired_action_inputs
from .factorized_chirps_source import calibrate
from .joint_unroll_source import source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective, private_difference
from .packet_runner import schedule
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import checkpoint_smoke, render, timing
from .runner import activate
from .spectral_covariance import SharedSpectrumRecovery
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-private-secant-source-m4-r01'
WEIGHT = .75
RECIPES = {
    name: FitRecipe(name, STEPS, partial(SharedSpectrumRecovery, WIDTHS, correlated=True),
                    colored_observation, SharedSpectrumRecovery.forward, initialize=calibrate)
    for name in ('paired', 'private_gain')
}
MODES = ('direct', 'reference', 'paired_coarse', 'paired', 'paired_half',
         'private_gain_coarse', 'private_gain', 'private_gain_half')
PAIRS = (('private_gain_coarse', 'paired_coarse'), ('private_gain', 'paired'),
         ('private_gain_half', 'paired_half'))


def frozen_models(root, contract, device):
    loaded = [load_model(root, recipe, contract, device) for recipe in RECIPES.values()]
    (control, a), (candidate, b) = loaded
    if a != b:
        raise ValueError('Source U clean threshold differs in matched arms')
    return (control, candidate), a


def evaluate_reserve(rt):
    def bind(root, contract, device, prior, tracker):
        models, threshold = frozen_models(root, contract, device)
        return partial(render, _unused=None, baseline=prior, tracker=tracker, models=models,
                       threshold=threshold)
    predict_clean_reserve(rt, modes=MODES, bind=bind)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, threshold = frozen_models(root, binding[2], device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=models, require_clean_bypass=False,
          replay_fn=receive_only, finalize=freeze, reader_chunk=144)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    launch = partial(start, module=rt['runner'])
    stages = (('checkpoint_smoke', ('source',)), ('smoke', RECIPES), ('fit', RECIPES),
              ('timing', ('source',)), ('evaluate', ROUTES), ('evaluate', ('reserve',)))
    try:
        for stage, rows in stages:
            schedule(rt, [(stage, name) for name in rows], launch)
        if launch(rt, 'score', 'source').wait():
            raise RuntimeError('Independent private-gain source scoring failed')
        write(root/'state.json', dict(status='SOURCE_SCORED'))
    except Exception as error:
        write(root/'state.json', dict(status='FAILED', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='action', required=True)
    pub = modes.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+key, type=Path, required=True)
    for stage in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        entry = modes.add_parser(stage)
        entry.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            rows = RECIPES if stage in ('smoke','fit') else (*ROUTES,'reserve') if stage == 'evaluate' else ('source',)
            entry.add_argument('row', choices=rows)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
        return
    rt = read(options['runtime'])
    row = options.get('row')
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'score':
        independent_score(rt, modes=MODES, comparisons=PAIRS)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt, RECIPES, MODES)
        elif action == 'timing':
            timing(rt, RECIPES['private_gain'])
        elif action == 'evaluate':
            if row == 'reserve':
                evaluate_reserve(rt)
            else:
                evaluate(rt, row)
        else:
            objective = partial(natural_objective, preserve=.35, difference=private_difference,
                                make_pair=partial(paired_action_inputs, echo_probability=.5),
                                tail_weight=TAIL_WEIGHT, half_private_weight=WEIGHT if row == 'private_gain' else 0.)
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[row], receiver_batches=True, calculate=objective)


if __name__ == '__main__':
    main()
