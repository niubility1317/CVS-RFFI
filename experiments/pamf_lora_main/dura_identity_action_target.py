"""Evaluate the frozen identity-action DURA ablation on both LoRa routes."""

from .dura_identity_action_source import ARM, RECIPES, RUN as SOURCE_RUN
from .dura_observed_variance_target import run_dose_target


RUN = '20261124-phase1-lora-dura-identity-action-target-m2-r01'
MODES = ('identity_action_half', 'identity_action_quarter')


if __name__ == '__main__':
    run_dose_target(run=RUN, source_run=SOURCE_RUN, arm=ARM, recipe=RECIPES[ARM],
                    modes=MODES, runner=__spec__.name)
