import torch

from cvsrffi.phase1_fcr_factors import excitation_features
from cvsrffi.phase1_fcr_fingerprint import (
    ExcitationConditionedFingerprintOperator,
    FingerprintFactorEncoder,
    fixed_response_basis,
)
from cvsrffi.phase1_fcr_types import FCRConfig


def _inputs(batch_size: int = 3) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(19)
    return (
        torch.randn(batch_size, 160),
        torch.randn(batch_size, 2, 256),
        torch.randn(batch_size, 2, 256),
        torch.randn(batch_size, 256, dtype=torch.complex64),
    )


def test_fixed_excitation_and_response_basis_have_physical_terms() -> None:
    signal = torch.tensor([[1 + 0j, 2 + 0j]], dtype=torch.complex64)
    features = excitation_features(signal)
    expected_features = torch.tensor(
        [[[1.0, 1.0, 1.0, 0.0], [2.0, 4.0, 8.0, 1.0]]]
    )
    torch.testing.assert_close(features, expected_features)

    basis = fixed_response_basis(signal)
    expected_basis = torch.tensor(
        [[[1 + 0j, 1 + 0j, 1 + 0j, 8 + 0j], [2 + 0j, 2 + 0j, 8 + 0j, 1 + 0j]]],
        dtype=torch.complex64,
    )
    torch.testing.assert_close(basis, expected_basis)


def test_fingerprint_factor_splits_unit_identity_and_state() -> None:
    raw_id, canonical, residual, s_hat = _inputs()
    encoder = FingerprintFactorEncoder(FCRConfig())

    factor = encoder(raw_id, canonical, residual, excitation_features(s_hat))

    assert factor.z_f_id.shape == (3, 160)
    assert factor.z_tx_state.shape == (3, 16)
    torch.testing.assert_close(factor.z_f_id.norm(dim=1), torch.ones(3), atol=1e-6, rtol=0)
    assert torch.isfinite(factor.z_tx_state).all()


def test_operator_is_common_phase_equivariant() -> None:
    raw_id, canonical, residual, s_hat = _inputs()
    encoder = FingerprintFactorEncoder(FCRConfig())
    operator = ExcitationConditionedFingerprintOperator(FCRConfig(), residual_ratio_max=0.12)
    factor = encoder(raw_id, canonical, residual, excitation_features(s_hat))
    phase = torch.exp(1j * torch.tensor(0.4))

    rotated = operator(s_hat * phase, factor)
    base = operator(s_hat, factor)

    torch.testing.assert_close(rotated.delta_f, base.delta_f * phase, atol=1e-4, rtol=1e-4)


def test_operator_zero_input_is_finite_and_zero() -> None:
    raw_id, canonical, residual, _ = _inputs(batch_size=2)
    s_hat = torch.zeros(2, 256, dtype=torch.complex64)
    encoder = FingerprintFactorEncoder(FCRConfig())
    operator = ExcitationConditionedFingerprintOperator(FCRConfig(), residual_ratio_max=0.12)
    factor = encoder(raw_id, canonical, residual, excitation_features(s_hat))

    out = operator(s_hat, factor)

    assert out.delta_f.dtype == torch.complex64
    assert torch.isfinite(out.delta_f.real).all()
    assert torch.isfinite(out.delta_f.imag).all()
    torch.testing.assert_close(out.delta_f, torch.zeros_like(out.delta_f))
    assert torch.isfinite(out.response_quality["energy_ratio"]).all()


def test_operator_caps_each_example_and_bounded_residual_uses_excitation_state() -> None:
    raw_id, canonical, residual, s_hat = _inputs()
    encoder = FingerprintFactorEncoder(FCRConfig())
    operator = ExcitationConditionedFingerprintOperator(FCRConfig(), residual_ratio_max=0.08)
    factor = encoder(raw_id, canonical, residual, excitation_features(s_hat))

    residual_only = operator.bounded_residual(excitation_features(s_hat), factor.z_tx_state)
    out = operator(s_hat, factor)
    ratio = out.delta_f.norm(dim=1) / s_hat.norm(dim=1)

    assert residual_only.shape == s_hat.shape
    assert residual_only.dtype == torch.float32
    assert torch.isfinite(residual_only).all()
    assert (ratio <= operator.residual_ratio_max + 1e-5).all()
