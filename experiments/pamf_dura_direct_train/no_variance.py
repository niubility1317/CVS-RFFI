"""Matched single-stage trial omitting only simulated variance calibration loss."""
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from . import runner as base
from . import fast_lr

RUN='20261030-phase1-DURA-direct-noVarAux-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.no_variance'


def objective(*args,**kwargs):
    _,terms=source_objective(*args,**kwargs)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']
    # The scalar variance mismatch stays a diagnostic log only. Native variance
    # still receives gradients through IQ restoration and the remaining losses.
    return loss,terms


def configure():
    fast_lr.configure()
    fast_configuration=base.configuration
    fast_command=base.command
    def configuration():
        value=fast_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_no_variance_aux',variance_aux_weight=0.,
            objective='finalIQNMSE+.1mean2internalpriorIQNMSE+.05multiscaleSTFT+.05mean2sourceactionprobeNMSE;logvariance diagnostic only',
            source_supervision='PairedcleanIQ and source simulator action; simulator variance logged only',
            variance_claim='Learned positive consistency parameter; not calibrated physical noise')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==fast_lr.MODULE else item
                for item in fast_command(args,mode,arm,role)]
    base.RUN=RUN
    base.configuration=configuration
    base.command=command
    base.direct_objective=objective


if __name__=='__main__':
    configure()
    base.main()
