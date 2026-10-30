"""Matched scratch E20 using fixed-grid decoder interpolation throughout."""
from . import runner as base
from . import fast_lr
from experiments.pamf_dura_efficiency.linear2x import install_switch

RUN='20261030-phase1-DURA-direct-fastinterp-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.fast_interp'


def configure():
    fast_lr.configure()
    original_configuration=base.configuration
    original_command=base.command
    original_factory=base.scratch_model

    def factory(seed):
        model=original_factory(seed)
        install_switch(model)('slice_linear2x')
        model.initialization_report['execution']='fixed_halfpixel_linear2x_slices;same random weights'
        return model

    def configuration():
        value=original_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_fast_interpolation',
            decoder_interpolation='half_pixel_linear2x_slices_in_four_Cores',
            interpolation_calls_changed=12,interpolation_calls_native=3,
            frozen_inference_interpolation='same slices;new execution version;not default DURA')
        return value

    def command(args,mode,arm=None,role=None):
        return [MODULE if item==fast_lr.MODULE else item for item in original_command(args,mode,arm,role)]

    base.RUN=RUN
    base.scratch_model=factory
    base.configuration=configuration
    base.command=command


if __name__=='__main__':
    configure()
    base.main()
