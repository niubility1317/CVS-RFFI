"""Test the frozen baseband DURA versus the native-coordinate identity control."""

from .dura_baseband_source import ARM, RECIPES, RUN as SOURCE_RUN
from .dura_identity_action_target import MODES as CONTROL_MODES
from .dura_observed_variance_target import run_dose_target


RUN = '20261124-phase1-lora-dura-baseband-target-m2-r01'
MODES = ('baseband_half', 'baseband_quarter')


if __name__ == '__main__':
    run_dose_target(run=RUN, source_run=SOURCE_RUN, arm=ARM, recipe=RECIPES[ARM],
                    modes=MODES, runner=__spec__.name, matched_modes=CONTROL_MODES,
                       allowed_gpus=(5,2,3,0,7,6,1,4))
