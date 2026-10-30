"""Matched single-stage E20 with a scratch-appropriate peak learning rate.

The source-only E20 pilot remains underfit (final IQ NMSE .04014 versus
continued weights .02097). This fixed next trial changes only peak LR;
its schedule is frozen before its own target access, with no warm start.
"""
import math

from . import runner as base

RUN='20261030-phase1-DURA-direct-fastlr-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.fast_lr'


def learning_rate(update):
    if not 0<=update<4940:raise ValueError('Fixed complete source budget required')
    if update<247:return 1e-3*(update+1)/247
    fraction=(update-247)/(4940-247-1)
    return 2e-5+(1e-3-2e-5)*.5*(1+math.cos(math.pi*fraction))


def configure():
    original_configuration=base.configuration
    original_command=base.command
    def configuration():
        value=original_configuration()
        value.update(run=RUN,peak_lr=1e-3,recipe='single_stage_scratch_fast_lr')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item=='experiments.pamf_dura_direct_train.runner' else item
                for item in original_command(args,mode,arm,role)]
    base.RUN=RUN
    base.configuration=configuration
    base.learning_rate=learning_rate
    base.command=command


if __name__=='__main__':
    configure()
    base.main()
