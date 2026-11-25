"""Train matched native WiSig FIR actions and score the frozen source panel."""

from functools import partial

import torch

from . import wisig_anchored_window_source as runner
from .wisig_native_fir import NativeFIR, disturb, objective


RUN = '20261125-phase1-wisig-native-fir-source-m4-r01'
ARMS = ('causal', 'centered')
READOUTS = ('coarse', 'prior_1', 'map_1', 'prior_2', 'final', 'gated')
MODES = ('direct', 'r69', *(f'{arm}_{name}' for arm in ARMS for name in READOUTS))
COMPARISONS = (*( (f'centered_{name}',f'causal_{name}') for name in READOUTS),
               ('centered_gated', 'direct'), ('centered_final', 'centered_map_1'),
               *((name,'r69') for name in MODES[2:]))


def build(arm):
    return NativeFIR(centered=arm=='centered')


def heads(model):
    return (model.spectral.out, model.coarse.out, model.estimator.out, model.global_head,
            *(network.out for network in model.priors))


def gate_features(model, analysis):
    return model.gate_scores(analysis)


def proposals(model, gate, received):
    analysis = model.analyze(received)
    stages, peak, score, variance = model.restore(analysis)
    active = (score > gate[0]) | (variance > gate[1])
    # The learned FIR never reindexes the output window or writes a separate pad.
    return (*stages, torch.where(active[:, None], stages[-1], received)), active, torch.zeros_like(peak)


def measurements(model, received):
    analysis = model.analyze(received)
    kernel, main, image, variance = analysis[3:]
    lag = kernel.abs().argmax(-1)-model.center
    return dict(dominant_fir_lag=lag.float(), diagonal_deviation=(main-1).abs().mean(-1),
                mirror_amplitude=image.abs().mean(-1), variance=variance[:, 0])


if __name__ == '__main__':
    load = partial(runner.load_models, arms=ARMS, build=build)
    runner.main(run=RUN, runner=__spec__.name, arms=ARMS, modes=MODES, comparisons=COMPARISONS,
                fit_arm=partial(runner.fit, build=build, calculate=objective, gate_features=gate_features,
                                trainable_heads=heads),
                run_smoke=partial(runner.smoke, models_from=load, variants=proposals,
                                  disturb=disturb, gate_features=gate_features),
                run_evaluation=partial(runner.evaluate, arms=ARMS, models_from=load, variants=proposals,
                                       auxiliary=measurements))
