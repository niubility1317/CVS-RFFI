"""Scratch E40 in one joint stage; source E20 curves still improve late in training."""
import math
from . import no_variance

RUN='20261030-phase1-DURA-direct-noVar-E40-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.long_budget'


def learning_rate(update):
    if not 0<=update<9880:raise ValueError('Fixed complete E40 source budget required')
    if update<247:return .001*(update+1)/247
    fraction=(update-247)/(9880-247-1)
    return .00002+(.001-.00002)*.5*(1+math.cos(math.pi*fraction))


def configure():
    no_variance.configure();base=no_variance.base
    previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_no_variance_aux_E40',epochs=40,updates=9880,
            L_exposures=252000,U_exposures=2268000,new_cumulative_epochs=40,
            corruption_epochs=20,corruption_schedule='Reuse fixed original20 source corruption views twice, same order with fresh per-epoch shuffle; no new data or trained ancestors',
            source_freeze='fixedfinalE40/9880;no early stop or source ranking',
            max_total_visible_GPU_processes=3,allowed_GPUs=[4,6,7],
            resource_scope='One own worker on one of user-authorized GPU4/6/7, >=12GB free and at most2 existing compute PIDs; no other-owner changes',
            source_rationale='Complete noVar E20 source curves: E16-20 mean IQNMSE .025564287546 vs E11-15 .028893018846; no epoch/target selection',
            host_numpy_conversion='DLPack_owned_FP32',
            comparison='Fixed original noVar E20 (cf40) and user-fixed DURA; different cumulative training budgets disclosed')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==no_variance.MODULE else item for item in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate


if __name__=='__main__':
    configure();no_variance.base.main()
