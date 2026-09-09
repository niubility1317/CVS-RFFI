"""Regression checks for the source-only A1 mechanism revision."""
import pytest
import torch

from cvsrffi.muse_ssdg import _rc4_calibrated_probability, _rc4_risk_threshold


@pytest.mark.parametrize("device,dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_rc4_risk_remains_fp32_inside_training_autocast(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    design = torch.tensor([[1.0], [1.001]], device=device)
    weight = torch.tensor([7.65], device=device)
    expected = _rc4_calibrated_probability(design, weight)
    with torch.autocast(device_type=device, dtype=dtype):
        actual = _rc4_calibrated_probability(design, weight)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)


def test_rc4_threshold_cannot_cut_through_equal_score_group():
    # Sorting places the correct row first, but a >= threshold accepts both.
    scores = torch.tensor([0.99, 0.99])
    target = torch.tensor([1.0, 0.0])
    result = _rc4_risk_threshold(scores, target, torch.ones(2, dtype=torch.bool),
                                 torch.zeros(2, dtype=torch.long),
                                 precision_target=0.98, min_coverage=0.01)
    assert result[-1] is False
def test_response_identity_coupling_has_real_gradient_and_preserves_rng():
    import torch
    from cvsrffi.phase1_fcr_types import FCRConfig
    from cvsrffi.phase1_fcr_fingerprint import ExcitationConditionedFingerprintOperator, FingerprintFactorOutput
    torch.manual_seed(37)
    control = ExcitationConditionedFingerprintOperator(FCRConfig())
    rng = torch.get_rng_state()
    torch.manual_seed(37)
    revised = ExcitationConditionedFingerprintOperator(FCRConfig(identity_response_coupling=True))
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in control.state_dict().items():
        assert torch.equal(value, revised.state_dict()[key])
    z = torch.randn(4, 160, requires_grad=True)
    state = torch.randn(4, 16)
    s = torch.complex(torch.randn(4, 256), torch.randn(4, 256))
    revised(s, FingerprintFactorOutput(z, state)).delta_f.abs().square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.norm() > 0


def test_continuous_r3_schedule_has_no_odd_step_holes():
    from cvsrffi.a1_r3_objective import objective_stage
    for epoch in range(1, 201):
        for step in (0, 1):
            stage = objective_stage(epoch, step, 'continuous')
            assert stage.scales['self'] > 0 and 'self' in stage.active
    assert objective_stage(92, 1, 'legacy').active == frozenset({'transplant', 'necessity'})


def test_pending_cuda_process_reserves_slot_without_double_counting():
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_a1_fast_v2 import occupied_gpu_pids
    process = SimpleNamespace(pid=17, poll=lambda: None)
    running = {'new': ({'gpu': 0}, process, None)}
    assert occupied_gpu_pids(0, running, {'21'}) == {'17', '21'}
    assert occupied_gpu_pids(0, running, {'17', '21'}) == {'17', '21'}
    assert occupied_gpu_pids(1, running, set()) == set()


def test_probability_weights_preserve_risk_evidence_and_budget():
    from test_fasttrust_rc4 import _calibration_fixture
    from cvsrffi.muse_ssdg import build_rc4_calibration, route_fasttrust_rc4
    anchor, ema1, ema2, labels, domains, z_norm = _calibration_fixture()
    calibration = build_rc4_calibration(anchor, ema1, ema2, labels, domains, z_norm,
        num_classes=3, num_domains=2, folds=2, min_stratum_samples=2,
        hard_precision_target=.8, partial_coverage_target=.8, negative_false_exclusion_target=.2)
    kwargs = dict(domains=domains, receivers=domains, z_norm=z_norm, calibration=calibration,
                  total_identity_effective_budget=.15, enable_negative=False)
    original = route_fasttrust_rc4(anchor, ema1, ema2, **kwargs)
    revised = route_fasttrust_rc4(anchor, ema1, ema2, reliability_weight_mode='calibrated_probability', **kwargs)
    import torch
    for name in ('candidate_mask', 'risk', 'agreement'):
        assert torch.equal(getattr(original, name), getattr(revised, name))
    # Old squared margin can give an eligible threshold-tied row zero mass.
    # The new mode may retain that row but cannot bypass its original risk gate.
    assert (revised.risk[revised.hard] >= calibration.hard_risk_threshold).all()
    assert revised.agreement[revised.hard].all()
    assert (revised.candidate_mask.sum(1)[revised.hard] == 1).all()
    selected = revised.hard | revised.partial | revised.negative
    assert (revised.weights * selected).sum() / len(labels) <= .150001
