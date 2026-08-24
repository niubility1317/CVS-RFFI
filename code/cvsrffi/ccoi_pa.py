"""Challenge-conditioned PA operator components for the Phase1 CCOI-PA study.

The module is deliberately self-contained: it consumes received source IQ and
frozen PA feature maps, and it never stores sample-level source representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CONTENT_STAT_DIM = 9


def _require_iq(x: Tensor) -> None:
    if x.ndim != 3 or x.size(1) != 2:
        raise ValueError(f"expected IQ tensor [B,2,L], got {tuple(x.shape)}")


def make_dual_iq_views(x: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Return a coarse content view and the untouched fingerprint view.

    Normalization is packet-level rather than token-level so local PA amplitude
    relationships are not independently erased in every window.
    """

    _require_iq(x)
    centered = x - x.mean(dim=-1, keepdim=True)
    packet_rms = centered.square().mean(dim=(1, 2), keepdim=True).clamp_min(eps).sqrt()
    return centered / packet_rms, x


def tokenize_iq(x: Tensor, token_length: int = 64, stride: int = 16) -> Tensor:
    _require_iq(x)
    token_length = int(token_length)
    stride = int(stride)
    if token_length <= 1 or stride <= 0 or x.size(-1) < token_length:
        raise ValueError("invalid token_length/stride for IQ length")
    return x.unfold(-1, token_length, stride).permute(0, 2, 1, 3).contiguous()


def fixed_content_statistics(tokens: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute fixed, detached content targets for each IQ token."""

    if tokens.ndim != 4 or tokens.size(2) != 2:
        raise ValueError(f"expected tokens [B,T,2,W], got {tuple(tokens.shape)}")
    with torch.no_grad():
        tok = tokens.detach().float()
        i = tok[:, :, 0]
        q = tok[:, :, 1]
        amp = (i.square() + q.square() + eps).sqrt()
        rms = amp.square().mean(dim=-1).clamp_min(eps).sqrt()
        papr = amp.amax(dim=-1) / rms
        diff = amp.diff(dim=-1).abs().mean(dim=-1)
        lag_num = (amp[..., 1:] * amp[..., :-1]).mean(dim=-1)
        lag_den = (
            amp[..., 1:].square().mean(dim=-1).clamp_min(eps).sqrt()
            * amp[..., :-1].square().mean(dim=-1).clamp_min(eps).sqrt()
        )
        phase_real = i[..., 1:] * i[..., :-1] + q[..., 1:] * q[..., :-1]
        phase_imag = q[..., 1:] * i[..., :-1] - i[..., 1:] * q[..., :-1]
        phase_norm = (phase_real.square() + phase_imag.square() + eps).sqrt()
        stats = torch.stack(
            (
                i.mean(dim=-1),
                i.std(dim=-1, unbiased=False),
                q.mean(dim=-1),
                q.std(dim=-1, unbiased=False),
                rms,
                papr,
                diff,
                lag_num / lag_den.clamp_min(eps),
                (phase_imag / phase_norm).mean(dim=-1),
            ),
            dim=-1,
        )
    return stats.to(dtype=tokens.dtype, device=tokens.device)


@dataclass
class ChallengeOutput:
    q: Tensor
    code_prob: Tensor
    content_stats: Tensor


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.scale * grad_output, None


class PAChallengeEncoder(nn.Module):
    """Encode packet content tokens into a compact challenge representation."""

    def __init__(
        self,
        token_length: int = 64,
        stride: int = 16,
        q_dim: int = 32,
        codebook_size: int = 48,
        hidden_dim: int = 64,
        num_tx: int = 0,
        num_rx: int = 0,
    ) -> None:
        super().__init__()
        self.token_length = int(token_length)
        self.stride = int(stride)
        self.q_dim = int(q_dim)
        self.codebook_size = int(codebook_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.q_head = nn.Linear(hidden_dim, q_dim)
        self.code_head = nn.Linear(q_dim, codebook_size)
        self.mask_context = nn.Conv1d(q_dim, q_dim, kernel_size=3, padding=1, bias=False)
        self.masked_stats_head = nn.Linear(q_dim, CONTENT_STAT_DIM)
        self.temporal_stats_head = nn.Linear(q_dim, CONTENT_STAT_DIM)
        self.tx_probe = nn.Linear(q_dim, int(num_tx)) if int(num_tx) > 1 else None
        self.rx_probe = nn.Linear(q_dim, int(num_rx)) if int(num_rx) > 1 else None

    def forward(self, content: Tensor) -> ChallengeOutput:
        tokens = tokenize_iq(content, self.token_length, self.stride)
        b, t, _, w = tokens.shape
        hidden = self.encoder(tokens.reshape(b * t, 2, w)).flatten(1)
        q = self.q_head(hidden).reshape(b, t, self.q_dim)
        q = F.normalize(q, dim=-1)
        code_prob = F.softmax(self.code_head(q), dim=-1)
        return ChallengeOutput(q=q, code_prob=code_prob, content_stats=fixed_content_statistics(tokens))

    def adversarial_probe_logits(self, q: Tensor, grl_scale: float = 1.0) -> Dict[str, Tensor]:
        pooled = q.mean(dim=1)
        reversed_q = _GradientReverse.apply(pooled, float(grl_scale))
        result: Dict[str, Tensor] = {}
        if self.tx_probe is not None:
            result["tx"] = self.tx_probe(reversed_q)
        if self.rx_probe is not None:
            result["rx"] = self.rx_probe(reversed_q)
        return result


class PAConditionalResponseHead(nn.Module):
    """Apply FiLM challenge conditioning to a frozen PA feature sequence."""

    def __init__(self, pa_channels: int, q_dim: int, response_dim: int = 64) -> None:
        super().__init__()
        self.q_dim = int(q_dim)
        self.pa_proj = nn.Linear(int(pa_channels), int(response_dim))
        self.film = nn.Linear(self.q_dim, 2 * int(response_dim))
        self.constant_condition = nn.Parameter(torch.zeros(self.q_dim))
        nn.init.xavier_uniform_(self.film.weight, gain=0.05)
        nn.init.zeros_(self.film.bias)

    def forward(self, pa_map: Tensor, q: Tensor, conditioned: bool = True) -> Tensor:
        if pa_map.ndim != 3 or q.ndim != 3 or pa_map.size(0) != q.size(0):
            raise ValueError("pa_map and q must be [B,C,L] and [B,T,Q]")
        token_count = q.size(1)
        pa_tokens = F.adaptive_avg_pool1d(pa_map, token_count).transpose(1, 2)
        base = self.pa_proj(pa_tokens)
        condition = q if conditioned else self.constant_condition.view(1, 1, -1).expand_as(q)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        return base * (1.0 + torch.tanh(gamma)) + beta


@dataclass
class OperatorOutput:
    theta: Tensor
    attention: Tensor
    coverage: Tensor
    entropy: Tensor


class OperatorPool(nn.Module):
    """Permutation-invariant, validity-aware pooling of local PA responses."""

    def __init__(self, response_dim: int, q_dim: int, operator_dim: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(int(response_dim) + int(q_dim), int(operator_dim)),
            nn.Tanh(),
            nn.Linear(int(operator_dim), 1),
        )
        self.value = nn.Linear(int(response_dim), int(operator_dim))

    def forward(self, response: Tensor, q: Tensor, valid_mask: Optional[Tensor] = None) -> OperatorOutput:
        if response.ndim != 3 or q.ndim != 3 or response.shape[:2] != q.shape[:2]:
            raise ValueError("response and q must share [B,T] geometry")
        b, t, _ = response.shape
        if valid_mask is None:
            valid_mask = torch.ones((b, t), device=response.device, dtype=torch.bool)
        valid_mask = valid_mask.to(device=response.device, dtype=torch.bool)
        if tuple(valid_mask.shape) != (b, t):
            raise ValueError(f"valid_mask must have shape {(b, t)}")

        scores = self.score(torch.cat((response, q), dim=-1)).squeeze(-1)
        safe_scores = scores.masked_fill(~valid_mask, -1e4)
        raw_attention = F.softmax(safe_scores, dim=1) * valid_mask.to(scores.dtype)
        attention = raw_attention / raw_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        theta = torch.sum(attention.unsqueeze(-1) * self.value(response), dim=1)
        coverage = valid_mask.float().mean(dim=1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
        return OperatorOutput(theta=theta, attention=attention, coverage=coverage, entropy=entropy)


def nonoverlap_anchor_indices(token_count: int, token_length: int, stride: int) -> Tensor:
    token_count = int(token_count)
    token_length = int(token_length)
    stride = int(stride)
    if token_count <= 0 or token_length <= 0 or stride <= 0:
        raise ValueError("token_count, token_length and stride must be positive")
    step = (token_length + stride - 1) // stride
    return torch.arange(0, token_count, step, dtype=torch.long)


def nonoverlap_holdout_masks(
    token_count: int,
    token_length: int = 64,
    stride: int = 16,
    fold: int = 0,
) -> Tuple[Tensor, Tensor]:
    anchors = nonoverlap_anchor_indices(token_count, token_length, stride)
    if anchors.numel() < 2:
        raise ValueError("at least two non-overlapping anchors are required")
    holdout_anchor = anchors[int(fold) % int(anchors.numel())]
    support = torch.zeros(int(token_count), dtype=torch.bool)
    holdout = torch.zeros(int(token_count), dtype=torch.bool)
    support[anchors] = True
    support[holdout_anchor] = False
    holdout[holdout_anchor] = True
    return support, holdout


def raw_intersection_count(
    support: Tensor,
    holdout: Tensor,
    token_length: int = 64,
    stride: int = 16,
) -> int:
    support_ids = torch.nonzero(support, as_tuple=False).flatten().tolist()
    holdout_ids = torch.nonzero(holdout, as_tuple=False).flatten().tolist()
    count = 0
    for left in support_ids:
        left_start, left_end = left * int(stride), left * int(stride) + int(token_length)
        for right in holdout_ids:
            right_start, right_end = right * int(stride), right * int(stride) + int(token_length)
            if max(left_start, right_start) < min(left_end, right_end):
                count += 1
    return count


def raw_support_holdout_masks(
    signal_length: int,
    token_count: int,
    token_length: int = 64,
    stride: int = 16,
    fold: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Map disjoint anchor selections back to disjoint raw-sample masks."""

    support_tokens, holdout_tokens = nonoverlap_holdout_masks(
        token_count,
        token_length=token_length,
        stride=stride,
        fold=fold,
    )
    support_raw = torch.zeros(int(signal_length), dtype=torch.bool)
    holdout_raw = torch.zeros(int(signal_length), dtype=torch.bool)
    for token_index in torch.nonzero(support_tokens, as_tuple=False).flatten().tolist():
        start = int(token_index) * int(stride)
        support_raw[start : min(int(signal_length), start + int(token_length))] = True
    for token_index in torch.nonzero(holdout_tokens, as_tuple=False).flatten().tolist():
        start = int(token_index) * int(stride)
        holdout_raw[start : min(int(signal_length), start + int(token_length))] = True
    if bool(torch.logical_and(support_raw, holdout_raw).any()):
        raise RuntimeError("raw support and holdout masks overlap")
    return support_raw, holdout_raw


class HeldoutChallengePredictor(nn.Module):
    """Predict frozen PA targets only at raw-sample-disjoint holdout anchors."""

    def __init__(self, operator_dim: int, q_dim: int, target_dim: int) -> None:
        super().__init__()
        hidden = max(int(operator_dim), int(q_dim), int(target_dim))
        self.net = nn.Sequential(
            nn.Linear(int(operator_dim) + int(q_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(target_dim)),
        )

    def forward(self, theta: Tensor, q_holdout: Tensor) -> Tensor:
        if theta.ndim != 2 or q_holdout.ndim != 3 or theta.size(0) != q_holdout.size(0):
            raise ValueError("theta and q_holdout must be [B,D] and [B,H,Q]")
        expanded = theta.unsqueeze(1).expand(-1, q_holdout.size(1), -1)
        return self.net(torch.cat((expanded, q_holdout), dim=-1))


class CCOIPASidecar(nn.Module):
    """Capacity-matched C1--C4 sidecar attached to a frozen Core90 model."""

    def __init__(
        self,
        pa_channels: int,
        num_classes: int,
        challenge_encoder: Optional[PAChallengeEncoder] = None,
        q_dim: int = 32,
        response_dim: int = 64,
        operator_dim: int = 64,
    ) -> None:
        super().__init__()
        self.challenge_encoder = challenge_encoder or PAChallengeEncoder(q_dim=q_dim)
        self.response_head = PAConditionalResponseHead(pa_channels, q_dim, response_dim)
        self.operator_pool = OperatorPool(response_dim, q_dim, operator_dim)
        self.classifier = nn.Linear(operator_dim, int(num_classes))
        self.heldout_predictor = HeldoutChallengePredictor(operator_dim, q_dim, pa_channels)

    def freeze_challenge_encoder(self) -> None:
        self.challenge_encoder.eval()
        for parameter in self.challenge_encoder.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        x: Tensor,
        pa_map: Tensor,
        *,
        conditioned: bool,
        valid_mask: Optional[Tensor] = None,
        holdout_fold: int = 0,
        holdout_support_pa_map: Optional[Tensor] = None,
        holdout_target_pa_map: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | OperatorOutput]:
        if pa_map.ndim != 3 or pa_map.size(1) == 0:
            raise ValueError("CCOI-PA requires a non-empty frozen pa_token_map")
        content, _ = make_dual_iq_views(x)
        challenge = self.challenge_encoder(content)
        condition_q = (
            challenge.q
            if bool(conditioned)
            else self.response_head.constant_condition.view(1, 1, -1).expand_as(challenge.q)
        )
        response = self.response_head(pa_map, challenge.q, conditioned=bool(conditioned))
        operator = self.operator_pool(response, condition_q, valid_mask)
        correction = self.classifier(operator.theta)

        support_mask, holdout_mask = nonoverlap_holdout_masks(
            challenge.q.size(1),
            token_length=self.challenge_encoder.token_length,
            stride=self.challenge_encoder.stride,
            fold=int(holdout_fold),
        )
        support_mask = support_mask.to(device=x.device).unsqueeze(0).expand(x.size(0), -1)
        holdout_mask = holdout_mask.to(device=x.device).unsqueeze(0).expand(x.size(0), -1)
        support_map = holdout_support_pa_map if holdout_support_pa_map is not None else pa_map
        support_response = self.response_head(support_map, challenge.q, conditioned=bool(conditioned))
        support_operator = self.operator_pool(support_response, condition_q, support_mask)
        q_holdout = condition_q[holdout_mask].reshape(x.size(0), -1, condition_q.size(-1))
        heldout_prediction = self.heldout_predictor(support_operator.theta, q_holdout)
        target_map = holdout_target_pa_map if holdout_target_pa_map is not None else pa_map
        frozen_pa_tokens = F.adaptive_avg_pool1d(target_map, challenge.q.size(1)).transpose(1, 2).detach()
        heldout_target = frozen_pa_tokens[holdout_mask].reshape(x.size(0), -1, pa_map.size(1))
        return {
            "logit_correction": correction,
            "theta": operator.theta,
            "attention": operator.attention,
            "coverage": operator.coverage,
            "attention_entropy": operator.entropy,
            "q": challenge.q,
            "condition_q": condition_q,
            "code_prob": challenge.code_prob,
            "response": response,
            "heldout_prediction": heldout_prediction,
            "heldout_target": heldout_target,
            "support_theta": support_operator.theta,
            "q_holdout": q_holdout,
            "support_mask": support_mask,
            "holdout_mask": holdout_mask,
        }


def _masked_mean(loss: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(dtype=loss.dtype)
    denom = mask_f.sum()
    if int(denom.detach().item()) == 0:
        return loss.sum() * 0.0
    return (loss * mask_f).sum() / denom


def codebook_balance_regularizer(
    code_prob: Tensor,
    min_effective_fraction: float = 0.75,
    max_mean_probability: float = 0.10,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Penalize only insufficient soft coverage or excessive code concentration."""

    if code_prob.ndim < 2 or code_prob.size(-1) < 2:
        raise ValueError("code_prob must end with at least two code probabilities")
    if not 0.0 < float(min_effective_fraction) <= 1.0:
        raise ValueError("min_effective_fraction must be in (0, 1]")
    if not 0.0 < float(max_mean_probability) <= 1.0:
        raise ValueError("max_mean_probability must be in (0, 1]")
    mean_code = code_prob.reshape(-1, code_prob.size(-1)).mean(dim=0).clamp_min(1e-8)
    mean_code = mean_code / mean_code.sum().clamp_min(1e-8)
    entropy = -(mean_code * mean_code.log()).sum()
    effective_codes = entropy.exp()
    minimum = math.log(float(code_prob.size(-1)) * float(min_effective_fraction))
    coverage_hinge = F.relu(entropy.new_tensor(minimum) - entropy).square()
    max_probability = mean_code.max()
    concentration_hinge = F.relu(max_probability - float(max_mean_probability)).square()
    return coverage_hinge + concentration_hinge, {
        "effective_codes": effective_codes,
        "max_mean_code_probability": max_probability,
        "mean_code_entropy": entropy,
    }


def challenge_pretrain_losses(
    encoder: PAChallengeEncoder,
    clean: Tensor,
    satellite: Tensor,
    mask: Optional[Tensor] = None,
    tx_labels: Optional[Tensor] = None,
    rx_labels: Optional[Tensor] = None,
    grl_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """Return source-only challenge-pretraining losses.

    TX/RX probe labels are optional. A caller must omit ``tx_labels`` for U_s;
    the function has no path that can infer or recover hidden labels.
    """

    clean_view, _ = make_dual_iq_views(clean)
    satellite_view, _ = make_dual_iq_views(satellite)
    clean_out = encoder(clean_view)
    satellite_out = encoder(satellite_view)
    if clean_out.q.shape != satellite_out.q.shape:
        raise ValueError("clean and satellite views must have matching token geometry")

    b, t, _ = clean_out.q.shape
    if mask is None:
        token_ids = torch.arange(t, device=clean.device)
        mask = (token_ids.remainder(3) == 1).unsqueeze(0).expand(b, -1)
    else:
        mask = mask.to(device=clean.device, dtype=torch.bool)
        if tuple(mask.shape) != (b, t):
            raise ValueError(f"mask must have shape {(b, t)}")

    masked_q = satellite_out.q.masked_fill(mask.unsqueeze(-1), 0.0)
    context = encoder.mask_context(masked_q.transpose(1, 2)).transpose(1, 2)
    masked_pred = encoder.masked_stats_head(context)
    masked_error = (masked_pred - satellite_out.content_stats).square().mean(dim=-1)
    loss_masked = _masked_mean(masked_error, mask)

    temporal_step = max(1, encoder.token_length // encoder.stride)
    if t > temporal_step:
        temporal_pred = encoder.temporal_stats_head(satellite_out.q[:, :-temporal_step])
        temporal_target = satellite_out.content_stats[:, temporal_step:]
        loss_temporal = F.smooth_l1_loss(temporal_pred, temporal_target)
    else:
        loss_temporal = satellite_out.q.sum() * 0.0

    loss_consistency = (1.0 - (clean_out.q * satellite_out.q).sum(dim=-1)).mean()
    pooled_std = satellite_out.q.reshape(-1, encoder.q_dim).std(dim=0, unbiased=False)
    loss_variance = F.relu(0.10 - pooled_std).mean()
    loss_code_consistency = F.mse_loss(clean_out.code_prob, satellite_out.code_prob)
    mean_code = satellite_out.code_prob.mean(dim=(0, 1)).clamp_min(1e-8)
    uniform = 1.0 / float(encoder.codebook_size)
    loss_code_utilization = (mean_code * (mean_code / uniform).log()).sum()
    loss_code_balance, code_balance_stats = codebook_balance_regularizer(satellite_out.code_prob)
    loss_code_confidence = -(
        satellite_out.code_prob.clamp_min(1e-8) * satellite_out.code_prob.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()

    probe_logits = encoder.adversarial_probe_logits(satellite_out.q, grl_scale=grl_scale)
    loss_tx = satellite_out.q.sum() * 0.0
    loss_rx = satellite_out.q.sum() * 0.0
    if tx_labels is not None and "tx" in probe_logits:
        loss_tx = F.cross_entropy(probe_logits["tx"], tx_labels.long())
    if rx_labels is not None and "rx" in probe_logits:
        loss_rx = F.cross_entropy(probe_logits["rx"], rx_labels.long())

    total = (
        loss_consistency
        + loss_masked
        + loss_temporal
        + 0.10 * loss_variance
        + 0.25 * loss_code_consistency
        + 0.20 * loss_code_balance
        + 0.005 * loss_code_confidence
        + 0.10 * loss_tx
        + 0.10 * loss_rx
    )
    return {
        "total": total,
        "consistency": loss_consistency,
        "masked_content": loss_masked,
        "temporal_prediction": loss_temporal,
        "variance": loss_variance,
        "code_consistency": loss_code_consistency,
        "code_utilization": loss_code_utilization,
        "code_balance": loss_code_balance,
        "effective_codes": code_balance_stats["effective_codes"],
        "max_mean_code_probability": code_balance_stats["max_mean_code_probability"],
        "code_confidence": loss_code_confidence,
        "tx_adversarial": loss_tx,
        "rx_adversarial": loss_rx,
    }


__all__ = [
    "CONTENT_STAT_DIM",
    "CCOIPASidecar",
    "ChallengeOutput",
    "HeldoutChallengePredictor",
    "OperatorOutput",
    "OperatorPool",
    "PAConditionalResponseHead",
    "PAChallengeEncoder",
    "codebook_balance_regularizer",
    "challenge_pretrain_losses",
    "fixed_content_statistics",
    "make_dual_iq_views",
    "nonoverlap_anchor_indices",
    "nonoverlap_holdout_masks",
    "raw_intersection_count",
    "raw_support_holdout_masks",
    "tokenize_iq",
]
