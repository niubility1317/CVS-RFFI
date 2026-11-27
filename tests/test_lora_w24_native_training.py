import torch

from experiments.pamf_lora_main import w24_joint_pamf as joint
from experiments.pamf_lora_main import w24_native_training as train
from experiments.pamf_lora_main.w24_native_dura import NativeDura, iq_error, source_pair
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_restoration.models import complex_iq, pair


def test_native_r02_r07_r17_preserve_aperture_length_and_unroll_gradients():
    iq = torch.randn(1,8192,dtype=torch.complex64,generator=torch.Generator().manual_seed(15))
    model = NativeDura()
    for stage in train.STAGES:
        with torch.no_grad():
            out = model(iq,stage=stage)
        assert out.shape == iq.shape and out.is_complex() and torch.isfinite(out).all()
    model.requires_grad_(False)
    for priors in model.priors:
        priors.out.requires_grad_(True)
    clean, observed, _, _ = source_pair(iq,torch.Generator().manual_seed(8))
    incoming = pair(observed)
    coarse, action, variance = model.estimate(incoming)
    loss = iq_error(complex_iq(model.unroll(incoming,coarse,action,variance)),clean)
    loss.backward()
    assert all(p.out.weight.grad is not None and torch.isfinite(p.out.weight.grad).all()
               for p in model.priors)
    assert all(p.grad is None for p in model.estimator.parameters())


def test_native_length_transfer_changes_only_positional_grid():
    model = PacketUNet(across_chirps=True)
    native = train.NativeTaskRestorer(torch.nn.Identity(),model)
    parent = model.state_dict()
    resized = native.core.state_dict()
    assert resized['temporal_position'].shape == (128,64)
    assert parent.keys() == resized.keys()
    assert all(torch.equal(value,resized[name]) for name,value in parent.items()
               if name != 'temporal_position')


def test_raw_formal_receipt_preserves_w24_parent_and_source_control():
    for run,stages in train.RUNS.items():
        mode=stages[-1]
        cases=joint.FULL_CASES
        for case in cases:
            proof=joint.formal_receipt(case,run)
            assert proof['reference_run']==joint.RUN_STRONG
            assert proof['modes'][-2:]==[joint.STRONG,mode]
        scores={name:dict(six_correct=1487,low_urban_correct=187,by_scene={'clean':297})
                for name in (*joint.parent_modes(joint.RUN_STRONG),*stages)}
        assert not joint.qualifies(scores,mode)
        scores[mode].update(six_correct=1492,low_urban_correct=189)
        assert joint.qualifies(scores,mode)
        scores[mode]['by_scene']['clean']=296
        assert not joint.qualifies(scores,mode)
