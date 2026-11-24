"""Train carrier-restricted and matched original DURA on the registered source roles."""

from dataclasses import replace

from .dura_carrier_action import CarrierRestrictedDura
from .dura_observed_variance_source import run_matched_source
from .dura_prior_width_source import RECIPES as WIDTH_RECIPES, RUN as WIDTH_RUN


RUN = '20261124-phase1-lora-dura-carrier-action-source-m2-r01'
CONTROL = WIDTH_RECIPES['wide']
ARM = 'carrier_action'
RECIPES = {CONTROL.name: CONTROL, ARM: replace(CONTROL, name=ARM, model=CarrierRestrictedDura)}
REUSED = {CONTROL.name: WIDTH_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{ARM}_{stage}', f'{CONTROL.name}_{stage}') for stage in ('coarse', 'half', 'full'))


if __name__ == '__main__':
    run_matched_source(run=RUN, arm=ARM, recipes=RECIPES, reused=REUSED,
                       modes=MODES, pairs=PAIRS, runner=__spec__.name)
