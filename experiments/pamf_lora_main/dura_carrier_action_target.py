"""Evaluate frozen carrier-restricted DURA on two predefined LoRa target routes."""

from .dura_carrier_action_source import ARM, RECIPES, RUN as SOURCE_RUN
from .dura_observed_variance_target import run_dose_target


RUN = '20261124-phase1-lora-dura-carrier-action-target-m2-r01'
MODES = ('carrier_action_half', 'carrier_action_quarter')


if __name__ == '__main__':
    run_dose_target(run=RUN, source_run=SOURCE_RUN, arm=ARM, recipe=RECIPES[ARM],
                    modes=MODES, runner=__spec__.name)
