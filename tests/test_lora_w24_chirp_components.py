import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_lora_main.w24_third_prior import OUTER_BASELINE, UNET_BASELINE, fit_inputs, promotion, trial


def test_factorized_packet_edits_keep_the_shared_and_private_subspaces():
    torch.manual_seed(35)
    model = PacketUNet(across_chirps=True, separate_chirps=True)
    y = torch.randn(2, 8, 384, dtype=torch.complex64)
    edges = torch.randn(2, 8, 48, dtype=torch.complex64)
    quality = rms = torch.ones(2, 8, 1)
    shared, restored = model(y, quality, rms, edges)
    common_edit, private_edit = shared - y, restored - shared
    torch.testing.assert_close(common_edit, common_edit.mean(1, keepdim=True).expand_as(common_edit),
                               rtol=0, atol=4e-7)
    torch.testing.assert_close(private_edit.mean(1), torch.zeros_like(private_edit[:, 0]), rtol=0, atol=4e-7)
    torch.testing.assert_close(restored.mean(-1), y.mean(-1), rtol=0, atol=3e-7)
    assert common_edit.abs().sum() > 0 and private_edit.abs().sum() > 0


def test_separate_risk_distinguishes_common_bias_and_lost_local_structure():
    from experiments.pamf_lora_main.w24_packet_unet import chirp_component_risk

    torch.manual_seed(36)
    target = torch.randn(2, 2, 8, 384, dtype=torch.complex64)
    common = target.mean(-2, keepdim=True)
    wrong_common = target + common
    lost_private = common.expand_as(target).clone().requires_grad_()
    common_change, private_change = chirp_component_risk(wrong_common, target)
    lost_common, lost_local = chirp_component_risk(lost_private, target)
    torch.testing.assert_close(private_change, private_change.new_zeros(()), atol=1e-9, rtol=0)
    torch.testing.assert_close(lost_common, lost_common.new_zeros(()), atol=1e-9, rtol=0)
    assert common_change > 0 and lost_local > 0
    lost_local.backward()
    assert torch.isfinite(lost_private.grad).all() and lost_private.grad.abs().sum() > 0


def test_component_trial_uses_only_source_iq_in_fit_and_requires_frozen_baselines():
    kind = 'chirp-components'
    plan = trial(kind)
    assert plan.steps == 24000 and plan.routes == (FULL,)
    assert plan.source_modes[-2:] == (UNET_BASELINE, OUTER_BASELINE)
    assert set(cases(kind)) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior=kind)[-2:] == plan.source_modes[2:4]
    paths = dict(source_root='U', run_root='runs/new', source_parent='G', unet_parent='U-Net',
                 classifier_root='F', native_target_root='target')
    for arm in plan.arms:
        assert fit_inputs(paths, arm) == dict(source_root='U', run_root=f'runs/new/{arm}')

    rows = {mode: dict(six_correct=1400, low_urban_correct=175, by_scene=dict(clean=297))
            for mode in plan.source_modes}
    rows[UNET_BASELINE]['low_urban_correct'] = 181
    candidate = rows[plan.source_modes[3]]
    candidate.update(six_correct=1442, low_urban_correct=180)
    assert not promotion([rows], kind)
    candidate['low_urban_correct'] = 181
    assert promotion([rows], kind)
