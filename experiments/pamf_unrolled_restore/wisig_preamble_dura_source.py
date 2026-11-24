"""Train matched WiSig DURA priors with and without the smooth observation anchor."""

from functools import partial

import torch

from . import wisig_anchored_window_source as runner
from .wisig_preamble_dura import PreambleDura, objective, perturb


RUN = '20261124-phase1-wisig-preamble-dura-source-m4-r01'
ARMS = ('identity', 'preamble')
READOUTS = ('coarse', 'prior_one', 'stage_one', 'prior_two', 'final', 'gated')
ABLATIONS = ('anchor_off', 'fixed_variance', 'cg_off')
MODES = ('direct', 'r69', *(f'{arm}_{name}' for arm in ARMS for name in READOUTS),
         *(f'preamble_{name}' for name in ABLATIONS))
COMPARISONS = (('preamble_gated', 'identity_gated'), ('preamble_final', 'identity_final'),
               ('preamble_final', 'preamble_stage_one'), ('preamble_final', 'preamble_prior_two'),
               *((f'preamble_gated', f'preamble_{item}') for item in ABLATIONS),
               *((mode, 'r69') for mode in MODES[2:]))


def build(arm):
    return PreambleDura(anchored=arm == 'preamble')


def trainable_heads(model):
    return model.frequency.out, model.coarse.out, model.estimator[-1], *(module.out for module in model.priors)


def gate_features(_model, analysis):
    return analysis[-1], analysis[6]


def proposals(model, gate, y):
    analysis = model.analyze(y)
    waves, shift, _, _ = model.restore(analysis)
    margin, noise = gate_features(model, analysis)
    active = (margin > gate[0]) | (noise > gate[1])
    result = (*waves, torch.where(active[:, None], waves[-1], y))
    if model.anchored:
        for option in ({'anchor': False}, {'use_noise': False}, {'use_cg': False}):
            states, _, _, level, *_ = model.stages(analysis, **option)
            result = (*result, torch.where(active[:, None], states[-1]*level, y))
    return result, active, shift


def measurements(model, y):
    analysis = model.analyze(y)
    _, action, variance, _, _, margin = model.stages(analysis)
    support, confidence = analysis[3], analysis[7]
    weight = support.sum(-1).clamp_min(1.)
    return dict(preamble_margin=margin, preamble_confidence=confidence, repeat_variance=analysis[6],
                action_offset=((action.gain-1.).abs()*support).sum(-1)/weight,
                effective_variance=(variance*support).sum(-1)/weight)


if __name__ == '__main__':
    loader = partial(runner.load_models, arms=ARMS, build=build)
    runner.main(run=RUN, runner=__spec__.name, arms=ARMS, modes=MODES, comparisons=COMPARISONS,
                fit_arm=partial(runner.fit, build=build, calculate=objective, gate_features=gate_features,
                                trainable_heads=trainable_heads),
                run_smoke=partial(runner.smoke, models_from=loader, variants=proposals, disturb=perturb,
                                  gate_features=gate_features),
                run_evaluation=partial(runner.evaluate, arms=ARMS, models_from=loader, variants=proposals,
                                       auxiliary=measurements))
