import torch

from cvsrffi.phase1_fcr_losses import compute_cross_losses
from cvsrffi.phase1_fcr_types import FCRConfig, FCRDecodeOutput, FCRFactorOutput, FCRPairBatch


def _factors(batch_size: int = 2, *, constant: bool = False, requires_grad: bool = False) -> FCRFactorOutput:
    if constant:
        z_s = torch.ones(batch_size, 2, 3, requires_grad=requires_grad)
        z_f = torch.ones(batch_size, 4, requires_grad=requires_grad)
        z_n = torch.ones(batch_size, 3, requires_grad=requires_grad)
    else:
        values = torch.arange(batch_size * 6, dtype=torch.float32).reshape(batch_size, 2, 3)
        z_s = values.clone().requires_grad_(requires_grad)
        z_f = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]][:batch_size]).requires_grad_(requires_grad)
        z_n = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]][:batch_size]).requires_grad_(requires_grad)
    return FCRFactorOutput(
        z_s=z_s,
        z_f_id=z_f,
        z_tx_state=torch.zeros(batch_size, 2),
        z_n_parts={"structured": z_n, "eta_pred": torch.zeros(batch_size, 3, requires_grad=requires_grad)},
        s_hat=torch.zeros(batch_size, 8, dtype=torch.complex64),
        content_confidence=torch.ones(batch_size),
    )


def _decode(value: float, *, requires_grad: bool = True) -> FCRDecodeOutput:
    mu = torch.full((2, 2, 8), value, requires_grad=requires_grad)
    return FCRDecodeOutput(
        mu_iq=mu,
        log_variance=torch.zeros(2, 8, requires_grad=requires_grad),
        delta_f=torch.zeros(2, 8, dtype=torch.complex64),
    )


def _pair(*, valid: bool = True, nuisance_valid: torch.Tensor | None = None) -> FCRPairBatch:
    mask = torch.tensor([valid, valid])
    return FCRPairBatch(
        clean_iq=torch.zeros(2, 2, 8),
        leo_iq=torch.ones(2, 2, 8),
        labels=torch.tensor([0, -1]),
        label_mask=torch.tensor([True, False]),
        receiver_id=torch.tensor([1, 2]),
        day_id=torch.tensor([3, 4]),
        nuisance=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nuisance_valid=torch.ones(2, 3, dtype=torch.bool) if nuisance_valid is None else nuisance_valid,
        physical_sample_id=("one", "two"),
        pair_id=("one", "two"),
        clean_crop_offset=torch.tensor([0, 0]),
        leo_crop_offset=torch.tensor([0, 0]),
        nuisance_pair_index=torch.tensor([0, 1]) if valid else torch.tensor([-1, -1]),
        content_pair_index=torch.tensor([-1, -1]),
        fingerprint_pair_index=torch.tensor([-1, -1]),
        pair_valid_mask={"nuisance": mask, "content": torch.zeros_like(mask), "fingerprint": torch.zeros_like(mask)},
    )


def _cross_kwargs(pair: FCRPairBatch, *, clean: FCRFactorOutput | None = None, leo: FCRFactorOutput | None = None, callbacks: tuple | None = None):
    clean = clean or _factors(requires_grad=True)
    leo = leo or _factors(requires_grad=True)
    if callbacks is None:
        callbacks = (lambda _x: _factors(), lambda _x: _factors())
    return dict(
        clean_factors=clean,
        leo_factors=leo,
        clean_self=_decode(0.0),
        leo_self=_decode(1.0),
        clean_to_leo=_decode(1.0),
        leo_to_clean=_decode(0.0),
        pair=pair,
        reencode_clean_to_leo=callbacks[0],
        reencode_leo_to_clean=callbacks[1],
        config=FCRConfig(input_len=8),
    )


def test_cross_loss_exposes_components_and_binds_both_swaps_to_destination_targets() -> None:
    result = compute_cross_losses(**_cross_kwargs(_pair()))
    reversed_result = compute_cross_losses(
        **{
            **_cross_kwargs(_pair()),
            "clean_to_leo": _decode(0.0),
            "leo_to_clean": _decode(1.0),
        }
    )

    assert {"self", "swap", "swap_clean_to_leo", "swap_leo_to_clean", "shared", "latent_cycle", "eta", "factor", "anti_collapse"} <= set(result.components)
    assert result.components["swap_clean_to_leo"].requires_grad
    assert result.components["swap_leo_to_clean"].requires_grad
    assert result.components["swap_clean_to_leo"] < reversed_result.components["swap_clean_to_leo"]
    assert result.components["swap_leo_to_clean"] < reversed_result.components["swap_leo_to_clean"]


def test_invalid_pair_mask_zeroes_pair_specific_terms_without_fallback() -> None:
    result = compute_cross_losses(**_cross_kwargs(_pair(valid=False)))

    for name in ("swap", "swap_clean_to_leo", "swap_leo_to_clean", "latent_cycle"):
        assert result.components[name].item() == 0.0
        assert torch.isfinite(result.components[name])


def test_shared_is_symmetric_stop_gradient_and_constant_codes_trigger_anti_collapse() -> None:
    clean = _factors(requires_grad=True)
    leo = _factors(requires_grad=True)
    leo.z_s = (leo.z_s.detach() + 1.0).requires_grad_()
    result = compute_cross_losses(**_cross_kwargs(_pair(), clean=clean, leo=leo))
    result.components["shared"].backward()
    assert clean.z_s.grad is not None and clean.z_s.grad.abs().sum() > 0
    assert leo.z_s.grad is not None and leo.z_s.grad.abs().sum() > 0

    collapsed_clean = _factors(constant=True, requires_grad=True)
    collapsed_leo = _factors(constant=True, requires_grad=True)
    collapsed = compute_cross_losses(
        **_cross_kwargs(_pair(), clean=collapsed_clean, leo=collapsed_leo)
    )
    assert collapsed.components["anti_collapse"].item() > 0.0
    collapsed.components["anti_collapse"].backward()
    gradients = (
        collapsed_clean.z_s.grad,
        collapsed_clean.z_f_id.grad,
        collapsed_leo.z_s.grad,
        collapsed_leo.z_f_id.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_latent_cycle_reencodes_both_syntheses_and_detaches_latent_targets() -> None:
    calls: list[torch.Tensor] = []
    clean = _factors(requires_grad=True)
    leo = _factors(requires_grad=True)

    def reencode(x: torch.Tensor) -> FCRFactorOutput:
        calls.append(x)
        return FCRFactorOutput(
            z_s=x[:, :1, :3].repeat(1, 2, 1),
            z_f_id=x[:, 0, :4],
            z_tx_state=torch.zeros(2, 2),
            z_n_parts={"structured": x[:, 1, :3], "eta_pred": torch.zeros(2, 3)},
            s_hat=torch.zeros(2, 8, dtype=torch.complex64),
            content_confidence=torch.ones(2),
        )

    result = compute_cross_losses(**_cross_kwargs(_pair(), clean=clean, leo=leo, callbacks=(reencode, reencode)))
    assert len(calls) == 2
    assert all(call.shape == (2, 2, 8) for call in calls)
    result.components["latent_cycle"].backward()
    assert clean.z_s.grad is None
    assert leo.z_s.grad is None


def test_eta_uses_only_known_fields_and_factor_accepts_external_probe_metric() -> None:
    clean = _factors(requires_grad=True)
    leo = _factors(requires_grad=True)
    clean.z_n_parts["eta_pred"] = torch.tensor([[1.0, 99.0, 3.0], [4.0, 99.0, 6.0]], requires_grad=True)
    leo.z_n_parts["eta_pred"] = torch.tensor([[1.0, 99.0, 3.0], [4.0, 99.0, 6.0]], requires_grad=True)
    valid = torch.tensor([[True, False, True], [True, False, True]])
    result = compute_cross_losses(
        **_cross_kwargs(_pair(nuisance_valid=valid), clean=clean, leo=leo),
        probe_metric=lambda _factors, _domains: 0.75,
    )

    assert result.components["eta"].item() == 0.0
    result.components["eta"].backward()
    assert leo.z_n_parts["eta_pred"].grad is not None
    assert leo.z_n_parts["eta_pred"].grad[:, 1].abs().sum().item() == 0.0
    assert result.metrics["factor_probe"] == 0.75
