"""Train a dechirped-coordinate DURA with the matched native-coordinate control."""

from dataclasses import replace

from .dura_baseband import BasebandDura
from .dura_identity_action_source import ARM as NATIVE, RECIPES as NATIVE_RECIPES, RUN as NATIVE_RUN
from .dura_observed_variance_source import run_matched_source


RUN = '20261124-phase1-lora-dura-baseband-source-m2-r01'
ARM = 'baseband'
CONTROL = NATIVE_RECIPES[NATIVE]
RECIPES = {
    NATIVE: CONTROL,
    ARM: replace(CONTROL, name=ARM, model=BasebandDura, restore=BasebandDura.forward),
}
REUSED = {NATIVE: NATIVE_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{ARM}_{stage}', f'{NATIVE}_{stage}') for stage in ('coarse', 'half', 'full'))


if __name__ == '__main__':
    run_matched_source(run=RUN, arm=ARM, recipes=RECIPES, reused=REUSED,
                       modes=MODES, pairs=PAIRS, runner=__spec__.name,
                       allowed_gpus=(5,2,3,0,7,6,1,4))
