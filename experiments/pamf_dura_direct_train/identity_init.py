"""Single-stage residual identity initialization, without learned ancestors."""
import torch

from . import runner as base
from . import fast_lr

RUN='20261030-phase1-DURA-direct-identityinit-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.identity_init'


@torch.no_grad()
def initialize_identity(model):
    """Start with coarse=y, A=I and both priors=y; retain random inner filters.

    This is an initialization only. Every component still trains jointly and
    only current received IQ is used at deployment.
    """
    estimator=model.frozen.estimator
    estimator.initial_kernel.zero_()
    estimator.initial_kernel[0]=1
    for head in [model.frozen.coarse.temporal.out,estimator.core.out,
                 estimator.global_head,*[prior.out for prior in model.prior]]:
        head.weight.zero_()
        if head.bias is not None:head.bias.zero_()
    if model.initialization_report is None:model.initialization_report={}
    model.initialization_report.update(mode='scratch_residual_identity',
        checkpoint_sources=[],ancestors=[],clean_identity_initialization=True,
        FIR_initial_impulse_index=0,coarse_residual_head_zero=True,
        action_heads_zero=True,prior_residual_heads_zero=True,
        initialization='Random inner filters, zero residual/action heads, identity FIR; no learned weights')
    return model


def configure():
    original_factory=base.scratch_model
    fast_lr.configure()
    fast_configuration=base.configuration
    fast_command=base.command
    def factory(seed):return initialize_identity(original_factory(seed))
    def configuration():
        value=fast_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_residual_identity',
                     initialization='scratch_residual_identity',FIR_initial_impulse_index=0)
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==fast_lr.MODULE else item
                for item in fast_command(args,mode,arm,role)]
    base.RUN=RUN
    base.scratch_model=factory
    base.configuration=configuration
    base.command=command


if __name__=='__main__':
    configure()
    base.main()
