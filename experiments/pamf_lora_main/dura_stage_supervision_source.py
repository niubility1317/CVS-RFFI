"""Match native-coordinate DURA while supervising the first MAP waveform."""

from dataclasses import replace
from functools import partial

from .dura_identity_action_source import ARM as NATIVE, RECIPES as NATIVE_RECIPES, RUN as NATIVE_RUN
from .dura_observed_variance_source import run_matched_source
from .dura_prior_width import PacketDura
from .relative_carrier import training_objective


RUN = '20261124-phase1-lora-dura-stage-supervision-source-m2-r01'
ARM = 'stage_supervised'
CONTROL = NATIVE_RECIPES[NATIVE]
RECIPES = {
    NATIVE: CONTROL,
    ARM: replace(CONTROL, name=ARM, restore=partial(PacketDura.forward, include_first=True)),
}
REUSED = {NATIVE: NATIVE_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{ARM}_{stage}', f'{NATIVE}_{stage}') for stage in ('coarse', 'half', 'full'))
CALCULATE = partial(training_objective, source_shape=True, first_map_weight=.5)


if __name__ == '__main__':
    run_matched_source(run=RUN, arm=ARM, recipes=RECIPES, reused=REUSED,
                       modes=MODES, pairs=PAIRS, runner=__spec__.name, calculate=CALCULATE,
                       allowed_gpus=(5,2,7,6,0,1,4,3))
