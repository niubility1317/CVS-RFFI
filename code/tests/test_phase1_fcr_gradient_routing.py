from __future__ import annotations

import torch

from cvsrffi.phase1_fcr_schedule import FCRLambdaConfig, stage_for_epoch
from cvsrffi.phase1_fcr_transplant import TransplantLossOutput
from cvsrffi.phase1_fcr_types import FCRLossOutput, FCRPairBatch
from model_dual_cvsincnet import build_dual_model
from train import combine_fcr_training_losses, compute_fcr_pair_objective


def _pair(label_mask: torch.Tensor) -> FCRPairBatch:
    batch_size = int(label_mask.numel())
    labels = torch.arange(batch_size, dtype=torch.long)
    labels = torch.where(label_mask, labels, torch.full_like(labels, -1))
    false = torch.zeros(batch_size, dtype=torch.bool)
    invalid = torch.full((batch_size,), -1, dtype=torch.long)
    return FCRPairBatch(
        clean_iq=torch.zeros(batch_size, 2, 8),
        leo_iq=torch.zeros(batch_size, 2, 8),
        labels=labels,
        label_mask=label_mask,
        receiver_id=torch.zeros(batch_size, dtype=torch.long),
        day_id=torch.zeros(batch_size, dtype=torch.long),
        nuisance=torch.zeros(batch_size, 3),
        nuisance_valid=torch.zeros(batch_size, 3, dtype=torch.bool),
        physical_sample_id=tuple(f"p{i}" for i in range(batch_size)),
        pair_id=tuple(f"p{i}" for i in range(batch_size)),
        clean_crop_offset=torch.zeros(batch_size, dtype=torch.long),
        leo_crop_offset=torch.zeros(batch_size, dtype=torch.long),
        nuisance_pair_index=invalid,
        content_pair_index=invalid.clone(),
        fingerprint_pair_index=invalid.clone(),
        pair_valid_mask={"nuisance": false, "content": false.clone(), "fingerprint": false.clone()},
    )


def _cross(reference: torch.Tensor) -> FCRLossOutput:
    zero = reference * 0.0
    components = {
        "self": reference,
        "swap": reference,
        "swap_clean_to_leo": reference,
        "swap_leo_to_clean": reference,
        "shared": reference,
        "latent_cycle": reference,
        "eta": reference,
        "factor": reference,
        "anti_collapse": reference,
    }
    return FCRLossOutput(total=sum(components.values(), zero), components=components, metrics={})


def _transplant(reference: torch.Tensor, active_pairs: int) -> TransplantLossOutput:
    components = {name: reference for name in ("target_id", "preserve_s", "preserve_n", "same_f", "drop_f")}
    return TransplantLossOutput(
        active_pairs=active_pairs,
        total=sum(components.values(), reference * 0.0),
        components=components,
        metrics={"active_pairs": float(active_pairs)},
    )


def test_label_terms_use_explicit_label_mask_and_unlabeled_allowed_losses_keep_gradients() -> None:
    allowed = torch.tensor(2.0, requires_grad=True)
    label_per_sample = torch.tensor([3.0, 7.0], requires_grad=True)
    labeled_pair = _pair(torch.tensor([True, False]))
    result = combine_fcr_training_losses(
        pair=labeled_pair,
        cross=_cross(allowed),
        transplant=_transplant(allowed, active_pairs=1),
        identity_per_sample=label_per_sample,
        physical_components={"features": allowed},
        stage=stage_for_epoch(151),
        configured=FCRLambdaConfig(),
    )
    result.total.backward()

    assert result.components["id"].item() == 3.0
    assert label_per_sample.grad is not None
    assert label_per_sample.grad[0].item() != 0.0
    assert label_per_sample.grad[1].item() == 0.0

    allowed_u = torch.tensor(2.0, requires_grad=True)
    unlabeled = _pair(torch.tensor([False, False]))
    unlabeled_result = combine_fcr_training_losses(
        pair=unlabeled,
        cross=_cross(allowed_u),
        transplant=_transplant(allowed_u, active_pairs=99),
        identity_per_sample=torch.tensor([11.0, 13.0], requires_grad=True),
        physical_components={"features": allowed_u},
        stage=stage_for_epoch(151),
        configured=FCRLambdaConfig(),
    )
    unlabeled_result.total.backward()

    assert unlabeled_result.components["id"].item() == 0.0
    assert unlabeled_result.components["transplant"].item() == 0.0
    assert unlabeled_result.components["factor"].item() == 0.0
    assert unlabeled_result.components["phys"].item() > 0.0
    assert unlabeled_result.components["swap"].requires_grad
    assert allowed_u.grad is not None and allowed_u.grad.item() != 0.0


def test_necessity_step_routes_only_transplant_and_freezes_decoder_flag() -> None:
    reference = torch.tensor(1.0, requires_grad=True)
    state = stage_for_epoch(91, optimizer_step=1)
    result = combine_fcr_training_losses(
        pair=_pair(torch.tensor([True, True])),
        cross=_cross(reference),
        transplant=_transplant(reference, active_pairs=1),
        identity_per_sample=torch.ones(2, requires_grad=True),
        physical_components={"features": reference},
        stage=state,
        configured=FCRLambdaConfig(),
    )

    assert state.freeze_decoder_for_necessity is True
    assert result.components["transplant"].item() > 0.0
    for name in ("id", "self", "swap", "shared", "latent_cycle", "eta", "factor", "phys"):
        assert result.components[name].item() == 0.0


def test_task9_fingerprint_excitation_detach_is_explicit_in_gradient_route() -> None:
    torch.manual_seed(1009)
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
    )
    assert model.fcr is not None
    aggregate = model.fcr(torch.randn(2, 2, 64), torch.randn(2, 160, requires_grad=True))
    aggregate.response.delta_f.abs().mean().backward()

    content_grads = [parameter.grad for parameter in model.fcr.content.parameters()]
    operator_grads = [parameter.grad for parameter in model.fcr.fingerprint_operator.parameters()]
    assert all(gradient is None for gradient in content_grads)
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in operator_grads)


def test_unlabeled_complete_objective_reuses_task6_7_8_and_stays_finite() -> None:
    torch.manual_seed(1010)
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
    )
    pair = _pair(torch.tensor([False, False]))
    pair.clean_iq = torch.randn(2, 2, 64)
    pair.leo_iq = pair.clean_iq + 0.01 * torch.randn_like(pair.clean_iq)
    pair.nuisance_pair_index = torch.arange(2)
    pair.pair_valid_mask["nuisance"] = torch.ones(2, dtype=torch.bool)

    result = compute_fcr_pair_objective(
        model=model,
        raw_model=model,
        pair=pair,
        role="U_s",
        stage=stage_for_epoch(90),
        configured=FCRLambdaConfig(),
        frozen_identity_classifier=torch.nn.Identity(),
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert result.components["id"].item() == 0.0
    assert result.components["factor"].item() == 0.0
    assert result.components["transplant"].item() == 0.0
    assert result.components["self"].requires_grad


def test_strict_labeled_pair_activates_existing_task8_necessity_route() -> None:
    torch.manual_seed(1011)
    model = build_dual_model(
        num_classes=2,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
    )
    pair = _pair(torch.tensor([True, True]))
    pair.clean_iq = torch.randn(2, 2, 64)
    pair.leo_iq = pair.clean_iq + 0.01 * torch.randn_like(pair.clean_iq)
    pair.nuisance_pair_index = torch.arange(2)
    pair.fingerprint_pair_index = torch.tensor([1, 0])
    pair.pair_valid_mask["nuisance"] = torch.ones(2, dtype=torch.bool)
    pair.pair_valid_mask["fingerprint"] = torch.ones(2, dtype=torch.bool)
    frozen_classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(128, 2))

    result = compute_fcr_pair_objective(
        model=model,
        raw_model=model,
        pair=pair,
        role="L_s",
        stage=stage_for_epoch(91, optimizer_step=1),
        configured=FCRLambdaConfig(),
        frozen_identity_classifier=frozen_classifier,
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert result.metrics["active_fingerprint_pairs"] == 2.0
    assert result.components["transplant"].item() > 0.0
    assert all(not parameter.requires_grad for parameter in frozen_classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.fcr.decoder.parameters())
