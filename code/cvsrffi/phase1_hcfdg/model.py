"""Single-backbone HCF-DG model components for Phase1 training.

The module deliberately keeps the identity path and the environment path
separate.  A backbone call produces the identity feature once; the
environment encoder only sees a detached copy of that feature.  The common
head is consequently usable as the complete inference path, while the
factorized specific head and auxiliary classifiers remain training-only.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "receiver": ("receiver", "receiver_id", "receiver_ids", "rx", "rx_id", "rx_ids"),
    "day": ("day", "day_id", "day_ids"),
    "channel": ("channel", "channel_id", "channel_ids"),
    "q_phys": (
        "q_phys",
        "channel_factors",
        "physical_channel_factors",
        "quality",
        "quality_score",
        "q",
    ),
}
_MAX_GRL_STRENGTH = 0.05


def _validate_grl_strength(value: float) -> float:
    strength = float(value)
    if not torch.isfinite(torch.tensor(strength)) or not 0.0 <= strength <= _MAX_GRL_STRENGTH:
        raise ValueError("grl_strength must be finite and in [0, 0.05]")
    return strength


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * grad_output, None


def _gradient_reverse(x: Tensor, strength: float) -> Tensor:
    return _GradientReverse.apply(x, strength)


def _lookup_value(source: Any, aliases: tuple[str, ...]) -> Any | None:
    """Read one metadata field from a mapping or a small metadata object."""

    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in aliases:
            if name in source:
                return source[name]
        return None
    for name in aliases:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _metadata_field(env_meta: Any, name: str, batch_size: int, device: torch.device) -> Tensor:
    """Return a one-dimensional long environment label, defaulting to zero."""

    value = _lookup_value(env_meta, _ENV_ALIASES[name])
    if value is None and torch.is_tensor(env_meta) and env_meta.ndim >= 2:
        column = {"receiver": 0, "day": 1, "channel": 2}[name]
        if env_meta.size(1) > column:
            value = env_meta[:, column]
    if value is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    labels = torch.as_tensor(value, device=device)
    if labels.ndim == 0:
        labels = labels.expand(batch_size)
    elif labels.ndim > 1:
        if labels.size(0) != batch_size:
            raise ValueError(f"{name} metadata must have batch dimension {batch_size}")
        labels = labels.argmax(dim=-1)
    labels = labels.reshape(-1)
    if labels.numel() != batch_size:
        raise ValueError(f"{name} metadata has {labels.numel()} rows, expected {batch_size}")
    return labels.long()


def _quality_tensor(
    q_phys: Tensor | None,
    env_meta: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
) -> Tensor:
    """Normalize q_phys to a finite ``[batch, width]`` feature tensor."""

    value: Any | None = q_phys
    if value is None:
        value = _lookup_value(env_meta, _ENV_ALIASES["q_phys"])
    if value is None:
        return torch.zeros(batch_size, width, device=device, dtype=dtype)

    quality = torch.as_tensor(value, device=device, dtype=dtype)
    if quality.ndim == 0:
        quality = quality.expand(batch_size, 1)
    elif quality.ndim == 1:
        if quality.numel() != batch_size:
            raise ValueError(f"q_phys has {quality.numel()} rows, expected {batch_size}")
        quality = quality.reshape(batch_size, 1)
    else:
        if quality.size(0) != batch_size:
            raise ValueError(f"q_phys has batch dimension {quality.size(0)}, expected {batch_size}")
        quality = quality.reshape(batch_size, -1)
    if quality.size(1) < width:
        quality = F.pad(quality, (0, width - quality.size(1)))
    elif quality.size(1) > width:
        quality = quality[:, :width]
    return torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_embedding_indices(labels: Tensor, size: int) -> tuple[Tensor, Tensor]:
    """Convert possibly invalid labels into embedding indices and a validity mask."""

    valid = (labels >= 0) & (labels < size)
    return labels.clamp(0, max(size - 1, 0)), valid


@dataclass
class HCFDGOutput:
    """Typed output shared by the HCF-DG trainer and its losses."""

    common_logits: Tensor
    specific_logits: Tensor | None
    z_id: Tensor
    z_rx: Tensor
    z_day: Tensor
    z_channel: Tensor
    z_env: Tensor
    receiver_logits: Tensor | None
    day_logits: Tensor | None
    channel_logits: Tensor | None
    tx_from_env_logits: Tensor | None
    conditional_receiver_logits: Tensor | None
    fused_feature: Tensor


@dataclass
class FactorizedEnvironmentOutput:
    """Environment features and auxiliary predictions produced by the encoder."""

    z_rx: Tensor
    z_day: Tensor
    z_channel: Tensor
    z_env: Tensor
    receiver_logits: Tensor
    day_logits: Tensor
    channel_logits: Tensor
    tx_from_env_logits: Tensor


class FactorizedEnvironmentEncoder(nn.Module):
    """Encode receiver/day/channel factors into a fixed 48D environment code.

    ``h_early`` is expected to be detached by the caller when it comes from a
    trainable identity backbone.  The encoder itself does not detach inputs so
    it can also be unit-tested and reused independently.
    """

    def __init__(
        self,
        input_dim: int = 160,
        env_dim: int = 48,
        num_receivers: int = 4,
        num_days: int = 3,
        num_channels: int = 5,
        q_phys_dim: int = 5,
        hidden_dim: int = 96,
        *,
        feature_dim: int | None = None,
        receiver_classes: int | None = None,
        day_classes: int | None = None,
        channel_classes: int | None = None,
    ) -> None:
        super().__init__()
        if feature_dim is not None:
            input_dim = feature_dim
        if receiver_classes is not None:
            num_receivers = receiver_classes
        if day_classes is not None:
            num_days = day_classes
        if channel_classes is not None:
            num_channels = channel_classes
        if env_dim != 48:
            raise ValueError("HCF-DG environment dimension is fixed at 48")
        if input_dim <= 0 or q_phys_dim <= 0 or hidden_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        if min(num_receivers, num_days, num_channels) <= 0:
            raise ValueError("environment class counts must be positive")

        self.input_dim = int(input_dim)
        self.env_dim = int(env_dim)
        self.factor_dim = env_dim // 3
        self.num_receivers = int(num_receivers)
        self.num_days = int(num_days)
        self.num_channels = int(num_channels)
        self.q_phys_dim = int(q_phys_dim)

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim + self.q_phys_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.rx_projection = nn.Linear(hidden_dim, self.factor_dim)
        self.day_projection = nn.Linear(hidden_dim, self.factor_dim)
        self.channel_projection = nn.Linear(hidden_dim, self.factor_dim)

        self.receiver_embedding = nn.Embedding(self.num_receivers, self.factor_dim)
        self.day_embedding = nn.Embedding(self.num_days, self.factor_dim)
        self.channel_embedding = nn.Embedding(self.num_channels, self.factor_dim)

        self.receiver_head = nn.Linear(self.factor_dim, self.num_receivers)
        self.day_head = nn.Linear(self.factor_dim, self.num_days)
        self.channel_head = nn.Linear(self.factor_dim, self.num_channels)
        self.tx_from_env_head: nn.Linear | None = None

    def set_tx_classes(self, num_classes: int) -> None:
        """Install the optional environment-to-TX probe used during training."""

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.tx_from_env_head = nn.Linear(self.env_dim, num_classes)

    def forward(
        self,
        h_early: Tensor,
        q_phys: Tensor | None = None,
        env_meta: Any | None = None,
        *,
        receiver_labels: Tensor | None = None,
        day_labels: Tensor | None = None,
        channel_labels: Tensor | None = None,
        grl_strength: float = _MAX_GRL_STRENGTH,
    ) -> FactorizedEnvironmentOutput:
        grl_strength = _validate_grl_strength(grl_strength)
        if h_early.ndim != 2:
            raise ValueError("h_early must be a rank-2 tensor")
        if h_early.size(1) != self.input_dim:
            raise ValueError(
                f"h_early has feature dimension {h_early.size(1)}, expected {self.input_dim}"
            )
        batch_size = h_early.size(0)
        q = _quality_tensor(
            q_phys,
            env_meta,
            batch_size,
            h_early.device,
            h_early.dtype,
            self.q_phys_dim,
        )
        shared = self.shared(torch.cat((h_early, q), dim=1))

        rx_labels = (
            receiver_labels.to(device=h_early.device).reshape(-1).long()
            if receiver_labels is not None
            else _metadata_field(env_meta, "receiver", batch_size, h_early.device)
        )
        day = (
            day_labels.to(device=h_early.device).reshape(-1).long()
            if day_labels is not None
            else _metadata_field(env_meta, "day", batch_size, h_early.device)
        )
        channel = (
            channel_labels.to(device=h_early.device).reshape(-1).long()
            if channel_labels is not None
            else _metadata_field(env_meta, "channel", batch_size, h_early.device)
        )
        for name, labels in (("receiver", rx_labels), ("day", day), ("channel", channel)):
            if labels.numel() != batch_size:
                raise ValueError(f"{name} labels have {labels.numel()} rows, expected {batch_size}")

        rx_indices, rx_valid = _safe_embedding_indices(rx_labels, self.num_receivers)
        day_indices, day_valid = _safe_embedding_indices(day, self.num_days)
        channel_indices, channel_valid = _safe_embedding_indices(channel, self.num_channels)

        z_rx = self.rx_projection(shared) + self.receiver_embedding(rx_indices) * rx_valid.unsqueeze(1)
        z_day = self.day_projection(shared) + self.day_embedding(day_indices) * day_valid.unsqueeze(1)
        z_channel = self.channel_projection(shared) + self.channel_embedding(channel_indices) * channel_valid.unsqueeze(1)
        z_env = torch.cat((z_rx, z_day, z_channel), dim=1)

        tx_logits = (
            self.tx_from_env_head(_gradient_reverse(z_env, grl_strength))
            if self.tx_from_env_head is not None
            else z_env.new_empty((batch_size, 0))
        )
        return FactorizedEnvironmentOutput(
            z_rx=z_rx,
            z_day=z_day,
            z_channel=z_channel,
            z_env=z_env,
            receiver_logits=self.receiver_head(z_rx),
            day_logits=self.day_head(z_day),
            channel_logits=self.channel_head(z_channel),
            tx_from_env_logits=tx_logits,
        )


class CommonSpecificLowRankHead(nn.Module):
    """Common classifier with an optional factorized rank-4 specific update.

    For a sample ``n`` the specific weight is exactly

    ``W_e[n] = W0 + U @ diag(a_rx[n] + a_day[n] + a_channel[n]) @ V.T``.

    ``U`` has shape ``[classes, rank]`` and ``V`` has shape
    ``[features, rank]``; the per-sample implementation below is the
    equivalent batched contraction.
    """

    def __init__(
        self,
        feature_dim: int = 160,
        num_classes: int = 6,
        rank: int = 4,
        *,
        specific: bool = True,
        dropout: float = 0.5,
        factor_dim: int = 16,
        input_dim: int | None = None,
        classes: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            feature_dim = input_dim
        if classes is not None:
            num_classes = classes
        if feature_dim <= 0 or num_classes <= 0 or rank <= 0:
            raise ValueError("head dimensions must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.specific = bool(specific)
        self.W0 = nn.Parameter(torch.empty(self.num_classes, self.feature_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.U = nn.Parameter(torch.empty(self.num_classes, self.rank), requires_grad=self.specific)
        self.V = nn.Parameter(torch.empty(self.feature_dim, self.rank), requires_grad=self.specific)
        self.dropout = nn.Dropout(dropout if self.specific else 0.0)

        if self.specific:
            self.a_rx = nn.Linear(factor_dim, self.rank)
            self.a_day = nn.Linear(factor_dim, self.rank)
            self.a_channel = nn.Linear(factor_dim, self.rank)
        else:
            self.a_rx = None
            self.a_day = None
            self.a_channel = None
        self.reset_parameters()

    @property
    def w0(self) -> Tensor:
        """Lowercase compatibility alias for the common weight matrix."""

        return self.W0

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W0)
        nn.init.normal_(self.U, mean=0.0, std=0.02)
        nn.init.normal_(self.V, mean=0.0, std=0.02)
        if self.a_rx is not None and self.a_day is not None and self.a_channel is not None:
            for layer in (self.a_rx, self.a_day, self.a_channel):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _factor_coefficients(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        z_rx: Tensor | None,
        z_day: Tensor | None,
        z_channel: Tensor | None,
        a_rx: Tensor | None,
        a_day: Tensor | None,
        a_channel: Tensor | None,
    ) -> Tensor:
        if not self.specific:
            return torch.zeros(batch_size, self.rank, device=device, dtype=dtype)

        def resolve(direct: Tensor | None, latent: Tensor | None, layer: nn.Linear) -> Tensor:
            if direct is not None:
                value = direct
            elif latent is not None:
                value = layer(latent)
            else:
                value = torch.zeros(batch_size, self.rank, device=device, dtype=dtype)
            value = torch.as_tensor(value, device=device, dtype=dtype)
            if value.ndim != 2 or value.shape != (batch_size, self.rank):
                raise ValueError(f"factor coefficients must have shape {(batch_size, self.rank)}")
            return value

        assert self.a_rx is not None and self.a_day is not None and self.a_channel is not None
        return (
            resolve(a_rx, z_rx, self.a_rx)
            + resolve(a_day, z_day, self.a_day)
            + resolve(a_channel, z_channel, self.a_channel)
        )

    def effective_weight(
        self,
        batch_size: int,
        *,
        z_rx: Tensor | None = None,
        z_day: Tensor | None = None,
        z_channel: Tensor | None = None,
        a_rx: Tensor | None = None,
        a_day: Tensor | None = None,
        a_channel: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return per-sample ``W_e`` for diagnostics and exact formula tests."""

        if device is None:
            device = self.W0.device
        if dtype is None:
            dtype = self.W0.dtype
        coefficients = self._factor_coefficients(
            batch_size,
            device,
            dtype,
            z_rx=z_rx,
            z_day=z_day,
            z_channel=z_channel,
            a_rx=a_rx,
            a_day=a_day,
            a_channel=a_channel,
        )
        low_rank = torch.einsum("cr,nr,dr->ncd", self.U, coefficients, self.V)
        return self.W0.unsqueeze(0) + low_rank

    def forward(
        self,
        features: Tensor,
        z_rx: Tensor | None = None,
        z_day: Tensor | None = None,
        z_channel: Tensor | None = None,
        *,
        a_rx: Tensor | None = None,
        a_day: Tensor | None = None,
        a_channel: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError(f"features must have shape [N, {self.feature_dim}]")
        sample_features = self.dropout(features) if self.specific else features
        weights = self.effective_weight(
            features.size(0),
            z_rx=z_rx,
            z_day=z_day,
            z_channel=z_channel,
            a_rx=a_rx,
            a_day=a_day,
            a_channel=a_channel,
            device=features.device,
            dtype=features.dtype,
        )
        return torch.einsum("nd,ncd->nc", sample_features, weights) + self.bias


@dataclass
class CounterfactualPair:
    """A same-TX receiver-swap batch and its explicit target labels."""

    features: Tensor
    source_indices: Tensor
    target_indices: Tensor
    source_environment: Tensor
    target_environment: Tensor
    target_env_labels: Any | None


class CounterfactualTransport(nn.Module):
    """Bounded feature transport for explicit counterfactual environments."""

    def __init__(
        self,
        feature_dim: int = 160,
        env_dim: int = 48,
        gamma_cap: float = 0.25,
        beta_cap: float = 0.25,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or env_dim <= 0:
            raise ValueError("transport dimensions must be positive")
        if gamma_cap < 0.0 or beta_cap < 0.0:
            raise ValueError("transport caps must be non-negative")
        self.feature_dim = int(feature_dim)
        self.env_dim = int(env_dim)
        self.gamma_cap = float(gamma_cap)
        self.beta_cap = float(beta_cap)
        hidden = int(hidden_dim or max(32, min(feature_dim, 128)))
        self.gamma_head = nn.Sequential(nn.Linear(self.env_dim, hidden), nn.GELU(), nn.Linear(hidden, self.feature_dim))
        self.beta_head = nn.Sequential(nn.Linear(self.env_dim, hidden), nn.GELU(), nn.Linear(hidden, self.feature_dim))

    def forward(
        self,
        h: Tensor,
        source_env: Tensor | None = None,
        target_env: Tensor | None = None,
        *,
        delta_env: Tensor | None = None,
    ) -> Tensor:
        if h.ndim != 2 or h.size(1) != self.feature_dim:
            raise ValueError(f"h must have shape [N, {self.feature_dim}]")

        if delta_env is not None:
            if source_env is not None or target_env is not None:
                raise ValueError("pass either delta_env or source/target environments")
            resolved_delta = delta_env
        elif source_env is None:
            raise ValueError("source_env or delta_env is required")
        elif target_env is None:
            # The two-argument form treats the second tensor as an already
            # computed delta_env, matching the transport equation directly.
            resolved_delta = source_env
        else:
            if source_env.ndim != 2 or target_env.ndim != 2:
                raise ValueError("source_env and target_env must be rank-2 tensors")
            if source_env.shape != target_env.shape or source_env.size(0) != h.size(0):
                raise ValueError("counterfactual environment batches must align with h")
            resolved_delta = target_env - source_env
        if resolved_delta.ndim != 2 or resolved_delta.size(0) != h.size(0):
            raise ValueError("delta_env must be a rank-2 tensor aligned with h")
        if resolved_delta.size(1) != self.env_dim:
            raise ValueError(f"environment features must have dimension {self.env_dim}")

        gamma = self.gamma_head(resolved_delta).clamp(-self.gamma_cap, self.gamma_cap)
        beta = self.beta_head(resolved_delta).clamp(-self.beta_cap, self.beta_cap)
        normalized_h = F.layer_norm(h, h.shape[1:])
        return (1.0 + gamma) * normalized_h + beta

    @staticmethod
    def same_tx_receiver_pairs(
        tx_labels: Tensor,
        receiver_labels: Tensor,
        *,
        target_receiver_labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Choose one deterministic receiver alternative for each same-TX row."""

        tx = tx_labels.reshape(-1)
        receivers = receiver_labels.reshape(-1).to(device=tx.device)
        if tx.numel() != receivers.numel():
            raise ValueError("tx_labels and receiver_labels must have equal length")
        target_receivers = (
            target_receiver_labels.reshape(-1).to(device=receivers.device)
            if target_receiver_labels is not None
            else None
        )
        if target_receivers is not None and target_receivers.numel() != receivers.numel():
            raise ValueError("target_receiver_labels must have equal length")

        source: list[int] = []
        target: list[int] = []
        for index in range(tx.numel()):
            candidates = torch.nonzero(tx == tx[index], as_tuple=False).reshape(-1)
            candidates = candidates[receivers[candidates] != receivers[index]]
            if target_receivers is not None:
                candidates = candidates[receivers[candidates] == target_receivers[index]]
            if candidates.numel() == 0:
                continue
            source.append(index)
            target.append(int(candidates[0]))
        return (
            torch.tensor(source, dtype=torch.long, device=tx.device),
            torch.tensor(target, dtype=torch.long, device=tx.device),
        )

    def receiver_swap(
        self,
        h: Tensor,
        environment: Tensor,
        *,
        tx_labels: Tensor,
        receiver_labels: Tensor,
        target_env_labels: Any | None = None,
        target_receiver_labels: Tensor | None = None,
        target_environment: Tensor | None = None,
    ) -> CounterfactualPair:
        """Transport each row to another receiver only within the same TX.

        ``target_env_labels`` is intentionally returned unchanged in the
        result so the caller can connect the generated feature to an explicit
        target receiver/day/channel label record rather than an inferred one.
        """

        source_indices, target_indices = self.same_tx_receiver_pairs(
            tx_labels,
            receiver_labels,
            target_receiver_labels=target_receiver_labels,
        )
        if environment.ndim != 2 or environment.size(1) != self.env_dim:
            raise ValueError(f"environment must have shape [N, {self.env_dim}]")
        if environment.size(0) != h.size(0):
            raise ValueError("h and environment must have equal batch length")

        target_pool = environment if target_environment is None else target_environment
        if target_pool.ndim != 2 or target_pool.shape != environment.shape:
            raise ValueError("target_environment must align with environment")
        source_environment = environment[source_indices]
        target_environment_rows = target_pool[target_indices]
        features = self(
            h[source_indices],
            source_environment,
            target_environment_rows,
        )
        return CounterfactualPair(
            features=features,
            source_indices=source_indices,
            target_indices=target_indices,
            source_environment=source_environment,
            target_environment=target_environment_rows,
            target_env_labels=target_env_labels,
        )

    def transport(self, h: Tensor, source_env: Tensor, target_env: Tensor) -> Tensor:
        """Named alias for callers that prefer an explicit transport verb."""

        return self(h, source_env, target_env)


def _call_backbone(backbone: nn.Module, x: Tensor) -> Any:
    """Request auxiliary features without risking a second backbone call."""

    try:
        signature = inspect.signature(backbone.forward)
        parameters = signature.parameters.values()
        accepts_aux = "return_aux" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_aux = False
    if accepts_aux:
        return backbone(x, return_aux=True)
    return backbone(x)


def _nested_lookup(output: Any, names: tuple[str, ...]) -> Any | None:
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
        for value in output.values():
            found = _nested_lookup(value, names)
            if found is not None:
                return found
    else:
        for name in names:
            if hasattr(output, name):
                return getattr(output, name)
    return None


def _extract_backbone_feature(output: Any) -> Tensor:
    feature = _nested_lookup(output, ("feat_joint", "id_feat_joint", "z_id", "features", "feature"))
    if feature is None and isinstance(output, (tuple, list)):
        tensors = [item for item in output if torch.is_tensor(item)]
        if len(tensors) >= 2:
            feature = tensors[1]
        elif tensors:
            feature = tensors[0]
    if feature is None and torch.is_tensor(output):
        feature = output
    if not torch.is_tensor(feature):
        raise TypeError("identity backbone output must contain tensor feat_joint")
    if feature.ndim > 2:
        feature = feature.flatten(1)
    if feature.ndim != 2:
        raise ValueError("identity backbone feat_joint must be rank-2 after flattening")
    return feature


def _infer_num_classes(backbone: nn.Module) -> int | None:
    for name in ("num_classes", "n_classes", "classes"):
        value = getattr(backbone, name, None)
        if isinstance(value, int) and value > 0:
            return value
    for name in ("classifier", "head", "cls_head", "tx_head"):
        module = getattr(backbone, name, None)
        out_features = getattr(module, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            return out_features
    return None


def _infer_backbone_feature_dim(backbone: nn.Module) -> int | None:
    """Resolve the pre-P_id fusion width without running the backbone."""

    for name in ("fusion_feature_dim", "feature_dim", "embedding_dim", "emb_dim"):
        value = getattr(backbone, name, None)
        if isinstance(value, int) and value > 0:
            return value
    for name in ("classifier", "head", "cls_head", "tx_head"):
        module = getattr(backbone, name, None)
        in_features = getattr(module, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return in_features
    return None


class HCFDGModel(nn.Module):
    """Single-backbone HCF-DG training model.

    The constructor accepts the explicit factor cardinalities used by the
    Phase1 episode metadata.  ``inference_logits`` bypasses every training-only
    component and calls only ``identity_features`` followed by ``common_head``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int | None = None,
        num_receivers: int = 4,
        num_days: int = 3,
        num_channels: int = 5,
        *,
        identity_dim: int = 160,
        env_dim: int = 48,
        q_phys_dim: int = 5,
        backbone_feature_dim: int | None = None,
        rank: int = 4,
        specific_dropout: float = 0.5,
        gamma_cap: float = 0.25,
        beta_cap: float = 0.25,
        receiver_classes: int | None = None,
        day_classes: int | None = None,
        channel_classes: int | None = None,
        num_rx: int | None = None,
        num_day: int | None = None,
        num_channel: int | None = None,
    ) -> None:
        super().__init__()
        if receiver_classes is not None:
            num_receivers = receiver_classes
        if day_classes is not None:
            num_days = day_classes
        if channel_classes is not None:
            num_channels = channel_classes
        if num_rx is not None:
            num_receivers = num_rx
        if num_day is not None:
            num_days = num_day
        if num_channel is not None:
            num_channels = num_channel
        resolved_classes = num_classes if num_classes is not None else _infer_num_classes(backbone)
        if resolved_classes is None:
            raise ValueError("num_classes is required when the backbone exposes no class count")
        if identity_dim != 160:
            raise ValueError("HCF-DG identity dimension is fixed at 160")
        if env_dim != 48:
            raise ValueError("HCF-DG environment dimension is fixed at 48")
        inferred_feature_dim = _infer_backbone_feature_dim(backbone)
        if backbone_feature_dim is None:
            backbone_feature_dim = inferred_feature_dim
        elif inferred_feature_dim is not None and int(backbone_feature_dim) != inferred_feature_dim:
            raise ValueError(
                "backbone_feature_dim conflicts with the backbone's declared fusion dimension"
            )
        if not isinstance(backbone_feature_dim, int) or backbone_feature_dim <= 0:
            raise ValueError(
                "backbone_feature_dim must be provided or safely inferred before checkpoint construction"
            )

        self.backbone = backbone
        self.num_classes = int(resolved_classes)
        self.backbone_feature_dim = int(backbone_feature_dim)
        self.identity_dim = int(identity_dim)
        self.env_dim = int(env_dim)
        self.num_receivers = int(num_receivers)
        self.num_days = int(num_days)
        self.num_channels = int(num_channels)

        self.p_id = nn.Linear(self.backbone_feature_dim, self.identity_dim)
        self.environment_encoder = FactorizedEnvironmentEncoder(
            input_dim=self.backbone_feature_dim,
            env_dim=self.env_dim,
            num_receivers=self.num_receivers,
            num_days=self.num_days,
            num_channels=self.num_channels,
            q_phys_dim=q_phys_dim,
        )
        self.environment_encoder.set_tx_classes(self.num_classes)
        self.common_head = CommonSpecificLowRankHead(
            feature_dim=self.identity_dim,
            num_classes=self.num_classes,
            rank=rank,
            specific=False,
            dropout=0.0,
        )
        self.specific_head = CommonSpecificLowRankHead(
            feature_dim=self.identity_dim,
            num_classes=self.num_classes,
            rank=rank,
            specific=True,
            dropout=specific_dropout,
        )
        self.conditional_receiver_head = nn.Linear(
            self.identity_dim + self.num_classes,
            self.num_receivers,
        )
        self.counterfactual_transport = CounterfactualTransport(
            feature_dim=self.backbone_feature_dim,
            env_dim=self.env_dim,
            gamma_cap=gamma_cap,
            beta_cap=beta_cap,
        )

    def _project_identity(self, feature: Tensor) -> Tensor:
        """Apply the checkpointed P_id projection to the fusion feature."""

        if feature.size(1) != self.backbone_feature_dim:
            raise ValueError(
                "backbone fusion feature dimension changed: "
                f"got {feature.size(1)}, expected {self.backbone_feature_dim}"
            )
        return self.p_id(feature)

    def _identity_features_and_backbone_output(self, x: Tensor) -> tuple[Tensor, Tensor, Any]:
        output = _call_backbone(self.backbone, x)
        fused_feature = _extract_backbone_feature(output)
        z_id = self._project_identity(fused_feature)
        return z_id, fused_feature, output

    def identity_features(self, x: Tensor) -> Tensor:
        """Run the identity backbone once and return its 160D joint feature."""

        z_id, _, _ = self._identity_features_and_backbone_output(x)
        return z_id

    def forward(
        self,
        x: Tensor,
        tx_labels: Tensor | None = None,
        env_meta: Any | None = None,
        q_phys: Tensor | None = None,
        training_aux: bool = False,
        *,
        receiver_labels: Tensor | None = None,
        day_labels: Tensor | None = None,
        channel_labels: Tensor | None = None,
        grl_strength: float = _MAX_GRL_STRENGTH,
    ) -> HCFDGOutput:
        grl_strength = _validate_grl_strength(grl_strength)
        z_id, fused_feature, _ = self._identity_features_and_backbone_output(x)

        env_output = self.environment_encoder(
            fused_feature.detach(),
            q_phys=q_phys,
            env_meta=env_meta,
            receiver_labels=receiver_labels,
            day_labels=day_labels,
            channel_labels=channel_labels,
            grl_strength=grl_strength,
        )
        common_logits = self.common_head(z_id)
        specific_logits = (
            self.specific_head(
                z_id,
                z_rx=env_output.z_rx,
                z_day=env_output.z_day,
                z_channel=env_output.z_channel,
            )
            if training_aux or self.training
            else None
        )
        auxiliary_enabled = bool(training_aux)
        tx_from_env_logits = env_output.tx_from_env_logits if auxiliary_enabled else None
        if tx_from_env_logits is not None and tx_from_env_logits.size(1) == 0:
            tx_from_env_logits = None
        conditional_receiver_logits = None
        if auxiliary_enabled and tx_labels is not None:
            labels = torch.as_tensor(tx_labels, device=z_id.device).reshape(-1).long()
            if labels.numel() != z_id.size(0):
                raise ValueError(
                    f"tx_labels have {labels.numel()} rows, expected {z_id.size(0)}"
                )
            if torch.any((labels < 0) | (labels >= self.num_classes)):
                raise ValueError(f"tx_labels must be in [0, {self.num_classes})")
            tx_onehot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=z_id.dtype)
            conditional_input = torch.cat((z_id, tx_onehot), dim=1)
            conditional_receiver_logits = self.conditional_receiver_head(
                _gradient_reverse(conditional_input, grl_strength)
            )
        return HCFDGOutput(
            common_logits=common_logits,
            specific_logits=specific_logits,
            z_id=z_id,
            z_rx=env_output.z_rx,
            z_day=env_output.z_day,
            z_channel=env_output.z_channel,
            z_env=env_output.z_env,
            receiver_logits=env_output.receiver_logits if auxiliary_enabled else None,
            day_logits=env_output.day_logits if auxiliary_enabled else None,
            channel_logits=env_output.channel_logits if auxiliary_enabled else None,
            tx_from_env_logits=tx_from_env_logits,
            conditional_receiver_logits=conditional_receiver_logits,
            fused_feature=fused_feature,
        )

    def inference_logits(self, x: Tensor) -> Tensor:
        """Compute deployment logits through the common identity head only."""

        return self.common_head(self.identity_features(x))


__all__ = [
    "CommonSpecificLowRankHead",
    "CounterfactualPair",
    "CounterfactualTransport",
    "FactorizedEnvironmentEncoder",
    "FactorizedEnvironmentOutput",
    "HCFDGModel",
    "HCFDGOutput",
]
