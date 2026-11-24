"""Fit DURA with the same receive-side off-band variance in training and inference."""
import argparse
from dataclasses import replace
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from .action_support_source import independent_score
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .carrier_coupled import packet_frame
from .channel_blind_source import fit
from .complex_vector_source import publish
from .dura_prior_width import PacketDura
from .dura_prior_width_source import CALCULATE, RECIPES as WIDTH_RECIPES, RUN as WIDTH_RUN
from .gated_slope_source import dispatch
from .relative_carrier import mix
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_signal_probe import observed_noise_rms


def receive_variance(observed):
    """Convert estimated complex noise power to the per-real-component MAP variance."""
    return (.5 * observed_noise_rms(observed).square()).flatten().clamp_min(1e-8)


class ObservedVarianceDura(PacketDura):
    def __init__(self):
        super().__init__(wide=True)

    def solve(self, native, observed, _rotation=None):
        coarse, action, _ = self.restorer.estimate_context(native)
        return self.restorer.stage_outputs(native, context=(coarse, action, receive_variance(observed)))


RUN = '20261124-phase1-lora-dura-observed-variance-source-m2-r01'
CONTROL = WIDTH_RECIPES['wide']
ARM = 'observed_variance'
RECIPES = {CONTROL.name: CONTROL, ARM: replace(CONTROL, name=ARM, model=ObservedVarianceDura)}
REUSED = {CONTROL.name: WIDTH_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{ARM}_{stage}', f'{CONTROL.name}_{stage}') for stage in ('coarse', 'half', 'full'))


def run_matched_source(*, run, arm, recipes, reused, modes, pairs, runner):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch','smoke','fit','timing','evaluate','score'):
        step = actions.add_parser(action)
        step.add_argument('runtime', type=Path)
        if action != 'dispatch':
            choices = (arm,) if action == 'fit' else (*ROUTES,'reserve') if action == 'evaluate' else ('source',)
            step.add_argument('row', choices=choices)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=run, runner=runner, allowed_gpus=(7,3,6,2,0,4,1,5))
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt, fit_arms=(arm,))
    if action == 'score':
        return independent_score(rt, modes=modes, comparisons=pairs)
    activate(rt)
    common = dict(recipes=recipes, reused=reused, reframe=packet_frame, mixer=mix)
    if action == 'smoke':
        return smoke(rt, modes=modes, split_risk=(False, False), calculations={arm: CALCULATE},
                     calibrate_fn=None, **common)
    if action == 'fit':
        return fit(rt, recipe=recipes[arm], calculate=CALCULATE, receiver_batches=True)
    if action == 'timing':
        return timing(rt, **common)
    if options['row'] == 'reserve':
        return reserve(rt, modes=modes, **common)
    return evaluate(rt, options['row'], modes=modes, **common)


if __name__ == '__main__':
    run_matched_source(run=RUN, arm=ARM, recipes=RECIPES, reused=REUSED,
                       modes=MODES, pairs=PAIRS, runner=__spec__.name)
