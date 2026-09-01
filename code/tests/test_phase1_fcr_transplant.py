import torch
from torch import nn

from cvsrffi.phase1_fcr_losses import compute_transplant_losses
from cvsrffi.phase1_fcr_transplant import TransplantOutput
from cvsrffi.phase1_fcr_types import FCRDecodeOutput, FCRFactorOutput, FCRPairBatch


def _factors(*, requires_grad: bool = False) -> FCRFactorOutput:
    return FCRFactorOutput(
        z_s=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=requires_grad),
        z_f_id=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=requires_grad),
        z_tx_state=torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=requires_grad),
        z_n_parts={"structured": torch.tensor([[0.5, 0.6], [0.7, 0.8]], requires_grad=requires_grad)},
        s_hat=torch.zeros(2, 6, dtype=torch.complex64),
        content_confidence=torch.ones(2),
    )


def _pair(*, valid: bool = True, target_visible: bool = True) -> FCRPairBatch:
    mask = torch.tensor([valid, False])
    return FCRPairBatch(
        clean_iq=torch.zeros(2, 2, 6),
        leo_iq=torch.zeros(2, 2, 6),
        labels=torch.tensor([0, 1]),
        label_mask=torch.tensor([True, target_visible]),
        receiver_id=torch.zeros(2, dtype=torch.long),
        day_id=torch.zeros(2, dtype=torch.long),
        nuisance=torch.zeros(2, 3),
        nuisance_valid=torch.zeros(2, 3, dtype=torch.bool),
        physical_sample_id=("a", "b"),
        pair_id=("a", "b"),
        clean_crop_offset=torch.zeros(2, dtype=torch.long),
        leo_crop_offset=torch.zeros(2, dtype=torch.long),
        nuisance_pair_index=torch.full((2,), -1, dtype=torch.long),
        content_pair_index=torch.full((2,), -1, dtype=torch.long),
        fingerprint_pair_index=torch.tensor([1 if valid else -1, -1]),
        pair_valid_mask={"fingerprint": mask},
    )


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, z_s, z_f_id, z_tx_state, z_n_parts):
        self.calls.append((z_s, z_f_id, z_tx_state, z_n_parts))
        source = torch.cat((z_s[:, 0], z_n_parts["structured"], torch.zeros(z_s.size(0), 2, device=z_s.device)), dim=1)
        fingerprint = torch.nn.functional.pad(z_f_id, (0, 2))
        mu = torch.stack((source, fingerprint), dim=1) * self.scale
        if mu.requires_grad:
            mu.retain_grad()
        self.outputs.append(mu)
        return FCRDecodeOutput(mu_iq=mu, log_variance=torch.zeros(mu.size(0), 6), delta_f=torch.zeros(mu.size(0), 6, dtype=torch.complex64))


def _reencode(iq: torch.Tensor) -> FCRFactorOutput:
    z_f = iq[:, 1, :4]
    return FCRFactorOutput(
        z_s=iq[:, 0, :2].unsqueeze(1),
        z_f_id=z_f,
        z_tx_state=torch.zeros(iq.size(0), 2, device=iq.device),
        z_n_parts={"structured": iq[:, 0, 2:4]},
        s_hat=torch.zeros(iq.size(0), 6, dtype=torch.complex64, device=iq.device),
        content_confidence=torch.ones(iq.size(0), device=iq.device),
    )


class _IdentityClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2, 4))
        self.calls = 0
        self.last_input = None

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        iq.retain_grad()
        self.last_input = iq
        return iq[:, 1, :4].matmul(self.weight.transpose(0, 1))


def _loss_kwargs(pair: FCRPairBatch, *, requires_grad: bool = False, freeze_decoder: bool = False):
    decoder = _Decoder()
    classifier = _IdentityClassifier()
    return dict(
        pair=pair,
        source_factors=_factors(requires_grad=requires_grad),
        target_factors=_factors(requires_grad=requires_grad),
        decoder=decoder,
        reencode=_reencode,
        identity_classifier=classifier,
        fingerprint_residual_error=lambda iq: iq[:, 1, :4].abs().mean(),
        freeze_decoder=freeze_decoder,
        necessity_margin=0.1,
    ), decoder, classifier


def test_invalid_fingerprint_pairs_are_exact_zero_and_never_call_generated_chain() -> None:
    kwargs, decoder, classifier = _loss_kwargs(_pair(valid=False))

    result = compute_transplant_losses(**kwargs)

    assert result.active_pairs == 0
    assert set(result.components) == {"target_id", "preserve_s", "preserve_n", "same_f", "drop_f"}
    assert result.total.item() == 0.0
    assert all(value.item() == 0.0 and torch.isfinite(value) for value in result.components.values())
    assert decoder.calls == []
    assert classifier.calls == 0


def test_strict_index_routes_source_content_nuisance_and_target_fingerprint() -> None:
    kwargs, decoder, _classifier = _loss_kwargs(_pair(valid=True))

    result = compute_transplant_losses(**kwargs)

    assert result.active_pairs == 1
    directed = decoder.calls[0]
    torch.testing.assert_close(directed[0], kwargs["source_factors"].z_s[:1])
    torch.testing.assert_close(directed[1], kwargs["target_factors"].z_f_id[1:2])
    torch.testing.assert_close(directed[2], kwargs["target_factors"].z_tx_state[1:2])
    torch.testing.assert_close(directed[3]["structured"], kwargs["source_factors"].z_n_parts["structured"][:1])
    assert result.components["preserve_s"].item() == 0.0
    assert result.components["preserve_n"].item() == 0.0
    assert result.components["target_id"].item() < 0.5


def test_target_id_component_includes_target_fingerprint_recovery() -> None:
    kwargs, _decoder, _classifier = _loss_kwargs(_pair(valid=True))
    baseline = compute_transplant_losses(**kwargs)

    def wrong_fingerprint_reencode(iq: torch.Tensor) -> FCRFactorOutput:
        output = _reencode(iq)
        output.z_f_id = torch.zeros_like(output.z_f_id)
        return output

    kwargs["reencode"] = wrong_fingerprint_reencode
    wrong = compute_transplant_losses(**kwargs)

    assert wrong.components["target_id"] > baseline.components["target_id"] + 0.20


def test_hidden_target_label_cannot_activate_target_identity_or_generated_chain() -> None:
    kwargs, decoder, classifier = _loss_kwargs(_pair(valid=True, target_visible=False))

    result = compute_transplant_losses(**kwargs)

    assert result.active_pairs == 0
    assert result.components["target_id"].item() == 0.0
    assert decoder.calls == []
    assert classifier.calls == 0


def test_classifier_is_frozen_eval_but_identity_loss_reaches_generated_iq_and_factors() -> None:
    kwargs, decoder, classifier = _loss_kwargs(_pair(), requires_grad=True)

    result = compute_transplant_losses(**kwargs)
    result.total.backward()

    assert classifier.training is False
    assert all(parameter.requires_grad is False and parameter.grad is None for parameter in classifier.parameters())
    assert classifier.last_input is not None and classifier.last_input.grad is not None
    assert classifier.last_input.grad.abs().sum().item() > 0.0
    assert kwargs["source_factors"].z_s.grad is not None
    assert kwargs["target_factors"].z_f_id.grad is not None
    assert decoder.outputs[0].grad is not None


def test_same_tx_control_preserves_source_identity_and_fingerprint() -> None:
    kwargs, _decoder, _classifier = _loss_kwargs(_pair())

    result = compute_transplant_losses(**kwargs)

    assert result.metrics["same_tx_accuracy"] == 1.0
    assert result.components["same_f"].item() < 0.5


def test_drop_f_uses_stop_gradient_correct_reference_and_penalizes_no_worsening() -> None:
    kwargs, decoder, _classifier = _loss_kwargs(_pair(), requires_grad=True)

    result = compute_transplant_losses(**kwargs)
    result.components["drop_f"].backward()

    assert result.components["drop_f"].item() > 0.0
    assert decoder.outputs[0].grad is None
    assert decoder.outputs[-1].grad is not None
    drop_call = decoder.calls[-1]
    assert torch.count_nonzero(drop_call[1]) == 0
    assert torch.count_nonzero(drop_call[2]) == 0


def test_drop_f_cannot_update_source_content_or_nuisance_paths() -> None:
    kwargs, _decoder, _classifier = _loss_kwargs(_pair(), requires_grad=True)
    kwargs["fingerprint_residual_error"] = lambda iq: iq[:, 0].square().mean()

    result = compute_transplant_losses(**kwargs)
    result.components["drop_f"].backward()

    assert kwargs["source_factors"].z_s.grad is None
    assert kwargs["source_factors"].z_n_parts["structured"].grad is None


def test_freeze_decoder_blocks_decoder_gradients_without_blocking_fingerprint_path() -> None:
    kwargs, decoder, _classifier = _loss_kwargs(_pair(), requires_grad=True, freeze_decoder=True)

    result = compute_transplant_losses(**kwargs)
    result.total.backward()

    assert decoder.scale.grad is None
    assert kwargs["target_factors"].z_f_id.grad is not None
    assert kwargs["target_factors"].z_f_id.grad.abs().sum().item() > 0.0


def test_transplant_output_exposes_generated_iq_logits_and_reencoding() -> None:
    kwargs, _decoder, _classifier = _loss_kwargs(_pair())
    result = compute_transplant_losses(**kwargs)

    assert isinstance(result.output, TransplantOutput)
    assert result.output.iq.shape == (1, 2, 6)
    assert result.output.target_logits.shape == (1, 2)
    assert result.output.reencoded.z_f_id.shape == (1, 4)
