"""Score the stage-supervised DURA against the frozen matching native arm."""

from .dura_identity_action_target import MODES as CONTROL_MODES
from .dura_observed_variance_target import run_dose_target
from .dura_stage_supervision_source import ARM, RECIPES, RUN as SOURCE_RUN


RUN = '20261124-phase1-lora-dura-stage-supervision-target-m2-r01'
MODES = ('stage_supervised_half', 'stage_supervised_quarter')


if __name__ == '__main__':
    run_dose_target(run=RUN, source_run=SOURCE_RUN, arm=ARM, recipe=RECIPES[ARM],
                    modes=MODES, runner=__spec__.name, matched_modes=CONTROL_MODES,
                    allowed_gpus=(5,2,7,6,0,1,4,3))
