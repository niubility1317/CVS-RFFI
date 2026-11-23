"""Run matched WiSig diagonal and coupled received-action units on source V."""

from functools import partial

import torch

from . import wisig_anchored_window_source as runner
from .wisig_observed_solve import ObservedSolve, gate_features, generic_observation, objective


RUN = '20261124-phase1-wisig-observed-solve-source-m4-r01'
ARMS = ('diagonal','coupled')
READOUTS = ('timed','prior_1','map_1','prior_2','final','gated')
SOLVE_ABLATIONS = ('same_diagonal','identity_action','constant_variance')
MODES = ('direct','r69',*(f'{arm}_{name}' for arm in ARMS for name in READOUTS),
         *(f'coupled_{name}' for name in SOLVE_ABLATIONS))
COMPARISONS = (*( (f'coupled_{name}',f'diagonal_{name}') for name in READOUTS),
               ('coupled_map_1','coupled_prior_1'),('coupled_final','coupled_prior_2'),
               ('coupled_final','coupled_map_1'),('coupled_gated','direct'),
               *((f'coupled_gated',f'coupled_{name}') for name in SOLVE_ABLATIONS),
               *((mode,'r69') for mode in MODES[2:]))


def build(arm):
    return ObservedSolve(arm=='coupled')


def proposals(model, gate, y):
    analysis = model.analyze(y)
    waves,shift,_,_ = model.restore(analysis)
    score,noise = model.gate_scores(analysis)
    active = (score>gate[0]) | (noise>gate[1])

    def guarded(estimate):
        return torch.where(active[:,None],estimate,y)

    result = (*waves,guarded(waves[-1]))
    if model.coupled:
        for readout in ({'coupled':False},{'use_action':False},{'use_noise':False}):
            stages,_,_,_,level,*_ = model.stages(analysis,**readout)
            result = (*result,guarded(stages[-1]*level))
    return result,active,shift


def measurements(model, y):
    analysis = model.analyze(y)
    gap,repeat = model.gate_scores(analysis)
    _,action,variance,support,*_ = model.stages(analysis)
    denominator = support.sum(-1).clamp_min(1.)
    mean = lambda value:(value*support).sum(-1)/denominator
    return dict(long_training_coherence_gain=gap,repeat_residual=repeat,
                gain_offset=mean((action.gain-1).abs()),mirror_amplitude=mean(action.image.abs()),
                off_diagonal_amplitude=mean(action.left.abs()+action.right.abs()),variance=mean(variance))


if __name__=='__main__':
    loader = partial(runner.load_models,arms=ARMS,build=build)
    runner.main(run=RUN,runner=__spec__.name,arms=ARMS,modes=MODES,comparisons=COMPARISONS,
                fit_arm=partial(runner.fit,build=build,calculate=objective,gate_features=gate_features),
                run_smoke=partial(runner.smoke,models_from=loader,variants=proposals,disturb=generic_observation,
                                  gate_features=gate_features),
                run_evaluation=partial(runner.evaluate,arms=ARMS,models_from=loader,variants=proposals,
                                       auxiliary=measurements))
