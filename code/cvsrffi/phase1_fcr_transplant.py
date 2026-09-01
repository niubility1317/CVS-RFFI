from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as functional
from torch import nn

from .phase1_fcr_types import FCRDecodeOutput, FCRFactorOutput, FCRPairBatch


@dataclass
class TransplantOutput:
    """Generated directed transplant together with its normal identity readout."""

    iq: torch.Tensor
    target_logits: torch.Tensor
    reencoded: FCRFactorOutput


@dataclass
class TransplantLossOutput:
    """Local directed-transplant losses; ``active_pairs`` never fabricates pairs."""

    active_pairs: int
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, float]
    output: TransplantOutput | None = None


Decoder = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], FCRDecodeOutput]
Reencoder = Callable[[torch.Tensor], FCRFactorOutput]
FingerprintError = Callable[[torch.Tensor], torch.Tensor]


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.reshape(-1).sum() * 0.0


def _nuisance_distance(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> torch.Tensor:
    if set(left) != set(right):
        raise ValueError("re-encoded nuisance parts must match source nuisance parts")
    terms = []
    for name in left:
        if left[name].shape != right[name].shape:
            raise ValueError(f"nuisance part {name} has an unexpected shape")
        terms.append((left[name] - right[name].detach()).square().mean())
    return torch.stack(terms).mean() if terms else _zero(next(iter(left.values())))


def _select_factors(factors: FCRFactorOutput, index: torch.Tensor) -> FCRFactorOutput:
    return FCRFactorOutput(
        z_s=factors.z_s[index],
        z_f_id=factors.z_f_id[index],
        z_tx_state=factors.z_tx_state[index],
        z_n_parts={name: value[index] for name, value in factors.z_n_parts.items()},
        s_hat=factors.s_hat[index],
        content_confidence=factors.content_confidence[index],
        response_coef=None if factors.response_coef is None else factors.response_coef[index],
        response_quality=None
        if factors.response_quality is None
        else {name: value[index] for name, value in factors.response_quality.items()},
    )


def _strict_fingerprint_rows(pair: FCRPairBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only Task2-authorized, visible, in-bounds cross-TX pair rows."""

    batch_size = pair.clean_iq.size(0)
    device = pair.clean_iq.device
    index = torch.as_tensor(pair.fingerprint_pair_index, device=device, dtype=torch.long).reshape(-1)
    mask = torch.as_tensor(
        pair.pair_valid_mask.get("fingerprint", torch.zeros(batch_size, device=device)),
        device=device,
        dtype=torch.bool,
    ).reshape(-1)
    visible = torch.as_tensor(pair.label_mask, device=device, dtype=torch.bool).reshape(-1)
    labels = torch.as_tensor(pair.labels, device=device, dtype=torch.long).reshape(-1)
    if index.numel() != batch_size or mask.numel() != batch_size or visible.numel() != batch_size:
        raise ValueError("fingerprint pair fields must have one entry per batch row")
    candidate = mask & (index >= 0) & (index < batch_size) & visible
    source_rows = torch.arange(batch_size, device=device)[candidate]
    target_rows = index[candidate]
    if target_rows.numel() == 0:
        return source_rows, target_rows
    target_visible = visible[target_rows]
    cross_tx = labels[source_rows] != labels[target_rows]
    keep = target_visible & cross_tx
    return source_rows[keep], target_rows[keep]


def freeze_identity_classifier(identity_classifier: nn.Module | Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Permanently freeze a normal ADV3B02 classifier without detaching its IQ input."""

    if isinstance(identity_classifier, nn.Module):
        identity_classifier.eval()
        for parameter in identity_classifier.parameters():
            parameter.requires_grad_(False)


def _decode(
    decoder: Decoder,
    source: FCRFactorOutput,
    target: FCRFactorOutput,
) -> FCRDecodeOutput:
    return decoder(source.z_s, target.z_f_id, target.z_tx_state, source.z_n_parts)


def _same_tx_decode(decoder: Decoder, source: FCRFactorOutput) -> FCRDecodeOutput:
    return decoder(source.z_s, source.z_f_id, source.z_tx_state, source.z_n_parts)


def _drop_f_decode(decoder: Decoder, source: FCRFactorOutput) -> FCRDecodeOutput:
    return decoder(
        source.z_s,
        torch.zeros_like(source.z_f_id),
        torch.zeros_like(source.z_tx_state),
        source.z_n_parts,
    )


def _run_decoder(
    decoder: Decoder,
    source: FCRFactorOutput,
    target: FCRFactorOutput,
    *,
    freeze_decoder: bool,
) -> FCRDecodeOutput:
    if not freeze_decoder or not isinstance(decoder, nn.Module):
        return _decode(decoder, source, target)
    original_flags = [parameter.requires_grad for parameter in decoder.parameters()]
    try:
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        return _decode(decoder, source, target)
    finally:
        for parameter, required in zip(decoder.parameters(), original_flags):
            parameter.requires_grad_(required)


def compute_directed_transplant_losses(
    *,
    pair: FCRPairBatch,
    source_factors: FCRFactorOutput,
    target_factors: FCRFactorOutput,
    decoder: Decoder,
    reencode: Reencoder,
    identity_classifier: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    fingerprint_residual_error: FingerprintError,
    necessity_margin: float = 0.05,
    freeze_decoder: bool = False,
) -> TransplantLossOutput:
    """Evaluate strict directed fingerprint transplantation and necessity.

    Only strict Task2 fingerprint pairs with visible source and target labels are
    allowed to activate the generated path.  The classifier receives generated
    IQ only, while label use is confined to CE after that normal forward.
    """

    source_rows, target_rows = _strict_fingerprint_rows(pair)
    if source_rows.numel() == 0:
        zero = _zero(source_factors.z_s)
        components = {name: zero for name in ("target_id", "preserve_s", "preserve_n", "same_f", "drop_f")}
        return TransplantLossOutput(
            active_pairs=0,
            total=zero,
            components=components,
            metrics={"active_pairs": 0.0, "same_tx_accuracy": 0.0},
        )

    freeze_identity_classifier(identity_classifier)
    source = _select_factors(source_factors, source_rows)
    target = _select_factors(target_factors, target_rows)
    target_labels = pair.labels[target_rows].to(device=source.z_s.device, dtype=torch.long)
    source_labels = pair.labels[source_rows].to(device=source.z_s.device, dtype=torch.long)

    directed_decode = _run_decoder(
        decoder, source, target, freeze_decoder=freeze_decoder
    )
    directed_reencoded = reencode(directed_decode.mu_iq)
    target_logits = identity_classifier(directed_decode.mu_iq)
    if target_logits.ndim != 2 or target_logits.size(0) != source_rows.numel():
        raise ValueError("identity classifier must return [active_pairs,num_classes] logits")
    if int(target_labels.max()) >= target_logits.size(1) or int(target_labels.min()) < 0:
        raise ValueError("visible target labels must index identity classifier logits")

    target_id = functional.cross_entropy(target_logits, target_labels)
    preserve_s = (directed_reencoded.z_s - source.z_s.detach()).square().mean()
    preserve_n = _nuisance_distance(directed_reencoded.z_n_parts, source.z_n_parts)
    target_f = (directed_reencoded.z_f_id - target.z_f_id.detach()).square().mean()
    # The named target-identity component binds the normal frozen-classifier
    # CE to the independently re-encoded target fingerprint recovery.
    target_id = target_id + target_f

    same_decode = _run_decoder(decoder, source, source, freeze_decoder=freeze_decoder)
    same_reencoded = reencode(same_decode.mu_iq)
    same_logits = identity_classifier(same_decode.mu_iq)
    same_identity = functional.cross_entropy(same_logits, source_labels)
    same_f = same_identity + (same_reencoded.z_f_id - source.z_f_id.detach()).square().mean()

    correct_error = fingerprint_residual_error(directed_decode.mu_iq)
    # Necessity must not let content/nuisance compensate for an erased
    # fingerprint.  The deleted-fingerprint branch therefore treats those
    # source factors as fixed context; a caller can route the residual error
    # through its re-encoded fingerprint path without opening E_s/E_n.
    frozen_drop_source = FCRFactorOutput(
        z_s=source.z_s.detach(),
        z_f_id=source.z_f_id.detach(),
        z_tx_state=source.z_tx_state.detach(),
        z_n_parts={name: value.detach() for name, value in source.z_n_parts.items()},
        s_hat=source.s_hat.detach(),
        content_confidence=source.content_confidence.detach(),
    )
    drop_decode = _run_decoder(
        decoder,
        frozen_drop_source,
        FCRFactorOutput(
            z_s=frozen_drop_source.z_s,
            z_f_id=torch.zeros_like(frozen_drop_source.z_f_id),
            z_tx_state=torch.zeros_like(frozen_drop_source.z_tx_state),
            z_n_parts=frozen_drop_source.z_n_parts,
            s_hat=frozen_drop_source.s_hat,
            content_confidence=frozen_drop_source.content_confidence,
        ),
        freeze_decoder=freeze_decoder,
    )
    dropped_error = fingerprint_residual_error(drop_decode.mu_iq)
    if correct_error.ndim != 0:
        correct_error = correct_error.mean()
    if dropped_error.ndim != 0:
        dropped_error = dropped_error.mean()
    drop_f = (correct_error.detach() + float(necessity_margin) - dropped_error).clamp_min(0.0)

    components = {
        "target_id": target_id,
        "preserve_s": preserve_s,
        "preserve_n": preserve_n,
        "same_f": same_f,
        "drop_f": drop_f,
    }
    total = torch.stack(tuple(components.values())).sum()
    output = TransplantOutput(
        iq=directed_decode.mu_iq,
        target_logits=target_logits,
        reencoded=directed_reencoded,
    )
    metrics = {name: float(value.detach().cpu()) for name, value in components.items()}
    metrics["active_pairs"] = float(source_rows.numel())
    metrics["same_tx_accuracy"] = float((same_logits.detach().argmax(dim=1) == source_labels).float().mean().cpu())
    metrics["target_tx_accuracy"] = float((target_logits.detach().argmax(dim=1) == target_labels).float().mean().cpu())
    metrics["target_f"] = float(target_f.detach().cpu())
    return TransplantLossOutput(
        active_pairs=int(source_rows.numel()),
        total=total,
        components=components,
        metrics=metrics,
        output=output,
    )
