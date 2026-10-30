"""Matched scratch E20: only source operator-probe supervision is stronger."""
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from . import runner as base
from . import fast_lr

RUN='20261030-phase1-DURA-direct-action020-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.strong_action'


def objective(*args,**kwargs):
    _,terms=source_objective(*args,**kwargs)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.2*terms['action_probe_nmse']+.001*terms['log_variance_mse']
    return loss,terms


def configure():
    fast_lr.configure()
    original_configuration=base.configuration
    original_command=base.command

    def configuration():
        value=original_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_source_action020',action_aux_weight=.2,
            objective='finalIQNMSE+.1mean2internalpriorIQNMSE+.05multiscaleSTFT+.2mean2sourceactionprobeNMSE+.001logvarianceMSE',
            interpolation='original native linear;no slice execution change',
            action_weight_basis='Source oracle-action CG16 improves fixed-prior IQ over learned-CG16 >=10% at both cached epochs;weighted action gradient scaled about .076 to .305 of IQ median norm')
        return value

    def command(args,mode,arm=None,role=None):
        return [MODULE if item==fast_lr.MODULE else item for item in original_command(args,mode,arm,role)]

    base.RUN=RUN
    base.configuration=configuration
    base.command=command
    base.direct_objective=objective


if __name__=='__main__':
    configure()
    base.main()
