import torch

from cvsrffi.jmrs02_j1 import J1Config, build_j1_module, validate_j1_rows


def _inputs(batch=4, length=256, z_dim=24, classes=6):
    torch.manual_seed(7)
    iq = torch.randn(batch, 2, length)
    z_id = torch.randn(batch, z_dim)
    base_logits = torch.randn(batch, classes)
    domain = torch.zeros(batch, dtype=torch.long)
    return iq, z_id, base_logits, domain


def test_j1_matrix_is_role_correct_and_has_no_joint_row():
    rows = validate_j1_rows(("B0", "RZ0", "RZ1", "RX1", "D1P", "P0"))
    assert rows == ("B0", "RZ0", "RZ1", "RX1", "D1P", "P0")
    for forbidden in ("RD", "RP", "DP", "RDP"):
        try:
            validate_j1_rows((forbidden,))
        except ValueError:
            pass
        else:
            raise AssertionError(f"joint row {forbidden} must not enter J1")


def test_all_residual_rows_are_exact_core90_bypasses_at_initialization():
    cfg = J1Config(z_dim=24, num_classes=6)
    iq, z_id, base_logits, domain = _inputs()
    for row in ("RZ0", "RZ1", "D1P"):
        model = build_j1_module(row, cfg)
        output = model(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
        assert torch.equal(output.final_logits, base_logits)
        assert torch.count_nonzero(output.residual_logits) == 0


def test_rz1_is_iq_conditioned_while_rz0_is_not():
    cfg = J1Config(z_dim=24, num_classes=6)
    iq, z_id, base_logits, domain = _inputs()
    rz0 = build_j1_module("RZ0", cfg)
    rz1 = build_j1_module("RZ1", cfg)
    out0a = rz0(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    out0b = rz0(iq=iq.flip(-1), z_id=z_id, base_logits=base_logits, domain=domain)
    out1a = rz1(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    out1b = rz1(iq=iq.flip(-1), z_id=z_id, base_logits=base_logits, domain=domain)
    assert torch.equal(out0a.diagnostics["conditioning"], out0b.diagnostics["conditioning"])
    assert not torch.equal(out1a.diagnostics["conditioning"], out1b.diagnostics["conditioning"])


def test_rx1_is_identity_initialized_fftshifted_and_power_normalized():
    cfg = J1Config(z_dim=24, num_classes=6)
    iq, z_id, base_logits, domain = _inputs()
    model = build_j1_module("RX1", cfg)
    output = model(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    assert torch.allclose(output.corrected_iq, iq, atol=1e-6, rtol=1e-6)
    assert output.diagnostics["fftshifted"].all()
    raw_power = iq.square().mean((1, 2))
    corrected_power = output.corrected_iq.square().mean((1, 2))
    assert torch.allclose(raw_power, corrected_power, atol=1e-6, rtol=1e-5)


def test_rx1_zero_initialized_correction_penalty_has_finite_gradients():
    cfg = J1Config(z_dim=24, num_classes=6)
    iq, z_id, base_logits, domain = _inputs()
    model = build_j1_module("RX1", cfg)
    output = model(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    loss = output.diagnostics["correction_norm"].square().mean()
    loss.backward()
    estimator_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith("estimator.") and parameter.grad is not None
    ]
    assert estimator_gradients
    assert all(torch.isfinite(gradient).all() for gradient in estimator_gradients)


def test_d1p_uses_cepstral_residual_without_spectral_ratio_or_roll():
    cfg = J1Config(z_dim=24, num_classes=6)
    iq, z_id, base_logits, domain = _inputs()
    model = build_j1_module("D1P", cfg)
    output = model(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    assert output.diagnostics["feature_family"] == "cepstral_log_spectrum_residual_no_ratio"
    assert output.diagnostics["unknown_symbol_invariant_claim"] is False
    assert output.diagnostics["valid_bin_fraction"].shape == (iq.shape[0],)


def test_p0_is_phase_nuisance_only_and_circular():
    cfg = J1Config(z_dim=24, num_classes=6)
    iq, z_id, base_logits, domain = _inputs()
    model = build_j1_module("P0", cfg)
    output = model(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    assert torch.equal(output.final_logits, base_logits)
    assert torch.count_nonzero(output.residual_logits) == 0
    assert output.nuisance_prediction.shape == (iq.shape[0], 4)
    assert output.diagnostics["phase_representation"] == "unit_phasor_circular_statistics"


def test_j1_parameter_budget_is_small():
    cfg = J1Config(z_dim=24, num_classes=6)
    for row in ("RZ0", "RZ1", "RX1", "D1P", "P0"):
        model = build_j1_module(row, cfg)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 0 < count <= 50_000
