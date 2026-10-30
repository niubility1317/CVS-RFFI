from types import SimpleNamespace
from experiments.pamf_dura_direct_train import fast_interp


def test_interpolation_execution_and_output_identity_reaches_workers_and_scorers():
    base=fast_interp.base
    original=(base.RUN,base.configuration,base.command,base.scratch_model,base.learning_rate)
    try:
        fast_interp.configure()
        args=SimpleNamespace(root='root',commit='a'*40)
        for mode in ('worker','score'):
            assert base.command(args,mode,'scratch_joint','target')[3]==fast_interp.MODULE
        config=base.configuration()
        assert config['run']==fast_interp.RUN
        assert config['decoder_interpolation']=='half_pixel_linear2x_slices_in_four_Cores'
        assert (config['epochs'],config['updates'],config['peak_lr'])==(20,4940,1e-3)
        assert config['ancestor_checkpoints']==[] and config['teacher_model'] is None
    finally:
        base.RUN,base.configuration,base.command,base.scratch_model,base.learning_rate=original
