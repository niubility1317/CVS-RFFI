"""Compare time-resolved and pooled noise estimates in matched LoRa fits."""

from functools import partial
from pathlib import Path

from torch import nn

from .channel_blind_source import panel, receive_only
from .complex_vector_source import score
from .joint_unroll_source import source_control
from .lapped_posterior import LappedPosterior, objective
from .local_phase import generic_phase_observation
from .local_wiener_source import freeze
from .spectral_block_source import FitRecipe, load_models, main, render, restore, smoke


RUN = '20261124-phase1-lora-lapped-posterior-source-m4-r01'
ARMS = ('pooled', 'local')
READOUTS = ('coarse', 'full', 'half')
RECIPES = {
    arm: FitRecipe(arm, 40_000, partial(LappedPosterior, local=arm == 'local'),
                   generic_phase_observation, restore, coarse_first=True, batch=24, fit_per_rx=4000,
                   validation_steps=(1800, 6000, 12000, 20000, 30000))
    for arm in ARMS
}
TRAINED_MODES = ('direct', 'reference', *(f'{arm}_{kind}' for arm in ARMS for kind in READOUTS))
MODES = (*TRAINED_MODES, *(f'local_pooled_{kind}' for kind in READOUTS))
COMPARISONS = (*(('local_' + kind, 'pooled_' + kind) for kind in READOUTS),
               *(('local_' + kind, 'local_pooled_' + kind) for kind in READOUTS),
               ('local_full', 'local_coarse'), ('local_half', 'direct'),
               *((name, 'reference') for name in MODES[2:]))


class PooledReadout(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observed, quality):
        return self.model.restore(observed, quality, stationary_readout=True)[:2]


def run_smoke(rt, recipes, _modes):
    smoke(rt, recipes, TRAINED_MODES)


def evaluate(rt, route, recipes, modes):
    binding = source_control(rt, route, 'cuda:0')
    models, threshold = load_models(Path(rt['run_root']), binding[2], 'cuda:0', recipes)
    models = (*models, PooledReadout(models[-1]).eval())
    score(rt, route, panel=panel, modes=modes, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=run_smoke, run_evaluate=evaluate)
