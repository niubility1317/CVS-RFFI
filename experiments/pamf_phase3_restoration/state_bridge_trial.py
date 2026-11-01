"""Matched source-only direct bridge without the original-observation shortcut."""
import argparse
import copy
from pathlib import Path
from . import direct_bridge_trial as parent
from .state_bridge import scratch_state_bridge
from .variance_trial import load_scale

RUN='20261101-phase1-Restore-StateBridge-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.state_bridge_trial'
ARM='MSAT_StateBridge'
TEMPLATE=parent.configuration(parent.ARM)


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed state/time direct IQ bridge allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,architecture='shared-MSAT state-time-only encoder+continuous160+four-feature/time-FiLM+four-direct-iterations',
        estimator_inputs='current_intermediate_state_and_time_only/no_original_observed_condition',
        inference_condition='receivedIQ onlyinitializes state/no original observed condition/no extra observations/no randomness',
        matched_control='20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/MSAT_DirectBridge',
        scientific_change='Only remove original received condition from estimator/two stem channels;remaining initial random state,paired time protocol,loss,order,optimizer and four-step execution matched')
    return c


def factory(arm):
    configuration(arm);return scratch_state_bridge(2026092701)


def install(scale):
    parent.RUN=RUN;parent.MODULE=MODULE;parent.ARM=ARM
    parent.configuration=configuration;parent.factory=factory
    parent.install(scale)


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,_=p.parse_known_args()
    install(load_scale(a.root));parent.base.main()
