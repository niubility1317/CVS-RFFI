"""Train the identity-action DURA ablation on the registered source-only data."""

from dataclasses import replace

from .dura_identity_action import IdentityActionDura
from .dura_observed_variance_source import run_matched_source
from .dura_prior_width_source import RECIPES as WIDTH_RECIPES, RUN as WIDTH_RUN


RUN = '20261124-phase1-lora-dura-identity-action-source-m2-r01'
CONTROL = WIDTH_RECIPES['wide']
ARM = 'identity_action'
RECIPES = {CONTROL.name: CONTROL, ARM: replace(CONTROL, name=ARM, model=IdentityActionDura)}
REUSED = {CONTROL.name: WIDTH_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{ARM}_{stage}', f'{CONTROL.name}_{stage}') for stage in ('coarse', 'half', 'full'))


if __name__ == '__main__':
    run_matched_source(run=RUN, arm=ARM, recipes=RECIPES, reused=REUSED,
                       modes=MODES, pairs=PAIRS, runner=__spec__.name,
                       allowed_gpus=(3,6,2,7,0,4,1,5))
