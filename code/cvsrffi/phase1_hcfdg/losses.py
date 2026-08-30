"""Losses used by the Phase1 HCF-DG training path.

The module deliberately keeps the Phase1 objective small and explicit.  The
LODO functions build prototypes from source support rows only, the
counterfactual function exposes each of its four terms, and HDRO operates on
the supplied group masks without reaching into any Phase2 or open-world
state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Hashable

import torch
from torch import Tensor
from torch.nn import functional as F


LODO_TEMPERATURE = 0.10
LODO_WEIGHT = 0.40
COUNTERFACTUAL_WEIGHT = 0.15
CF_WEIGHT = COUNTERFACTUAL_WEIGHT
HDRO_WEIGHT = 0.10
CSD_WEIGHT = 0.15
FAC_WEIGHT = 0.05
HCFDG_LOSS_WEIGHTS = {
    "lodo": LODO_WEIGHT,
    "counterfactual": COUNTERFACTUAL_WEIGHT,
    "hdro": HDRO_WEIGHT,
    "csd": CSD_WEIGHT,
    "fac": FAC_WEIGHT,
}
_EPS = 1e-8


def _get_field(value: Any, *names: str, default: Any = None) -> Any:
    """Read a field from either a typed output object or a mapping."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _first_tensor(*values: Any) -> Tensor | None:
    for value in values:
        if isinstance(value, Tensor):
            return value
        if isinstance(value, Mapping):
            found = _first_tensor(*value.values())
            if found is not None:
                return found
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            found = _first_tensor(*value)
            if found is not None:
                return found
    return None


def _zero_from(*values: Any) -> Tensor:
    """Return a differentiable scalar zero on the first available device."""

    reference = _first_tensor(*values)
    if reference is None:
        return torch.tensor(0.0)
    if reference.is_floating_point() or reference.is_complex():
        return reference.sum() * 0.0
    return reference.to(dtype=torch.get_default_dtype()).sum() * 0.0


def _as_tensor(value: Any, *, device: torch.device, name: str) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device)
    try:
        return torch.as_tensor(value, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a tensor or numeric sequence") from exc


def _as_vector(value: Any, *, device: torch.device, n: int, name: str) -> Tensor:
    tensor = _as_tensor(value, device=device, name=name).reshape(-1)
    if tensor.numel() != n:
        raise ValueError(f"{name} must contain exactly {n} rows")
    return tensor


def _as_bool_mask(value: Any, *, device: torch.device, n: int, name: str) -> Tensor:
    tensor = _as_tensor(value, device=device, name=name).reshape(-1)
    if tensor.numel() != n:
        raise ValueError(f"{name} must contain exactly {n} rows")
    return tensor.to(dtype=torch.bool)


def _python_scalar(value: Any) -> Hashable:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("a label value must be scalar")
        value = value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (AttributeError, ValueError, TypeError):
            pass
    return value


def _validate_temperature(value: float, *, name: str = "temperature") -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _validate_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_z(z: Tensor, *, name: str = "z") -> Tensor:
    if not isinstance(z, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if z.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D]")
    if not z.is_floating_point():
        z = z.float()
    return z


def _resolve_query_domain(
    query_domain: Any,
    domain: Tensor,
    query_mask: Tensor | None,
) -> tuple[Any, Tensor]:
    if query_domain is None:
        if query_mask is None:
            raise ValueError("query_domain is required when query_mask is absent")
        observed = domain[query_mask].detach().unique()
        if observed.numel() != 1:
            raise ValueError("query_mask must identify exactly one query domain")
        query_domain = observed[0]
    query_value = _python_scalar(query_domain)
    domain_query = domain == query_value
    return query_value, domain_query


def _lodo_masks(
    z: Tensor,
    domain: Any,
    *,
    query_domain: Any,
    support_mask: Any | None,
    query_mask: Any | None,
) -> tuple[Tensor, Tensor, Tensor, Any]:
    domain_tensor = _as_vector(domain, device=z.device, n=z.shape[0], name="domain")
    supplied_query_mask = (
        None
        if query_mask is None
        else _as_bool_mask(query_mask, device=z.device, n=z.shape[0], name="query_mask")
    )
    query_value, domain_query = _resolve_query_domain(
        query_domain,
        domain_tensor,
        supplied_query_mask,
    )
    support = ~domain_query
    query = domain_query
    if support_mask is not None:
        support = support & _as_bool_mask(
            support_mask,
            device=z.device,
            n=z.shape[0],
            name="support_mask",
        )
    if supplied_query_mask is not None:
        query = query & supplied_query_mask
    return (
        _as_vector(domain, device=z.device, n=z.shape[0], name="domain"),
        support,
        query,
        query_value,
    )


def _class_tensor(y: Any, *, device: torch.device, n: int) -> Tensor:
    labels = _as_vector(y, device=device, n=n, name="y")
    if labels.is_floating_point() or labels.is_complex():
        if not torch.equal(labels, labels.round()):
            raise ValueError("y must contain integer class labels")
    return labels.to(dtype=torch.long)


def _normalized_prototype(rows: Tensor) -> Tensor:
    return F.normalize(rows.mean(dim=0), dim=0, eps=_EPS)


def _masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return values.mean() if values.numel() else _zero_from(values)
    mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != values.shape[0]:
        raise ValueError("loss mask and per-row loss have different lengths")
    if not bool(mask.any()):
        return _zero_from(values)
    return values[mask].mean()


@dataclass(frozen=True)
class LODOInfo:
    """Diagnostics for one leave-one-domain-out prototype calculation."""

    support_domains: frozenset[Hashable]
    query_domain: Hashable
    query_count: int
    support_count: int
    class_count: int
    prototype_counts: Mapping[Hashable, int]
    unseen_query_classes: frozenset[Hashable] = frozenset()
    prototype_classes: tuple[Hashable, ...] = ()
    prototypes: Tensor | None = None


def lodo_prototype_loss(
    z: Tensor,
    y: Any,
    domain: Any,
    query_domain: Any = None,
    *,
    support_mask: Any | None = None,
    query_mask: Any | None = None,
    temperature: float = LODO_TEMPERATURE,
) -> tuple[Tensor, LODOInfo]:
    """Classify one held-out domain using prototypes from the other domains.

    Query-domain rows are excluded before class means are formed.  When a
    query class has no support prototype, that query is reported as unseen and
    omitted from the cross-entropy rather than producing an invalid target.
    """

    z = _validate_z(z)
    temperature = _validate_temperature(temperature)
    labels = _class_tensor(y, device=z.device, n=z.shape[0])
    domain_tensor, support, query, query_value = _lodo_masks(
        z,
        domain,
        query_domain=query_domain,
        support_mask=support_mask,
        query_mask=query_mask,
    )

    support_domains = frozenset(
        _python_scalar(value) for value in domain_tensor[support].detach().cpu().tolist()
    )
    support_count = int(support.sum().item())
    query_count = int(query.sum().item())
    support_classes = torch.unique(labels[support], sorted=True)
    query_classes = torch.unique(labels[query], sorted=True)
    support_class_values = tuple(
        _python_scalar(value) for value in support_classes.detach().cpu().tolist()
    )
    unseen = frozenset(
        _python_scalar(value)
        for value in query_classes.detach().cpu().tolist()
        if not bool((support_classes == value).any())
    )

    prototype_counts: dict[Hashable, int] = {}
    prototype_list: list[Tensor] = []
    for class_value in support_classes:
        class_mask = support & labels.eq(class_value)
        prototype_counts[_python_scalar(class_value)] = int(class_mask.sum().item())
        prototype_list.append(_normalized_prototype(z[class_mask]))

    prototypes = torch.stack(prototype_list) if prototype_list else None
    info = LODOInfo(
        support_domains=support_domains,
        query_domain=query_value,
        query_count=query_count,
        support_count=support_count,
        class_count=len(support_class_values),
        prototype_counts=prototype_counts,
        unseen_query_classes=unseen,
        prototype_classes=support_class_values,
        prototypes=prototypes,
    )
    if not support_count or not query_count or prototypes is None:
        return _zero_from(z), info

    query_indices = torch.nonzero(query, as_tuple=False).flatten()
    query_labels = labels[query_indices]
    valid_query = torch.zeros_like(query_labels, dtype=torch.bool)
    target = torch.zeros_like(query_labels)
    for class_index, class_value in enumerate(support_classes):
        matches = query_labels.eq(class_value)
        valid_query |= matches
        target[matches] = class_index
    if not bool(valid_query.any()):
        return _zero_from(z), info

    query_features = F.normalize(z[query_indices[valid_query]], dim=1, eps=_EPS)
    logits = query_features @ prototypes.transpose(0, 1)
    logits = logits / temperature
    loss = F.cross_entropy(logits, target[valid_query])
    return loss, info


def _prepare_content_keys(keys: Any, *, device: torch.device, n: int) -> tuple[str, Any]:
    if isinstance(keys, Tensor):
        tensor = keys.to(device=device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1).expand(n, 1)
        elif tensor.ndim == 1:
            if tensor.shape[0] != n:
                raise ValueError(f"content_keys must contain exactly {n} rows")
            tensor = tensor.reshape(n, 1)
        elif tensor.shape[0] != n:
            raise ValueError(f"content_keys must contain exactly {n} rows")
        else:
            tensor = tensor.reshape(n, -1)
        if not (tensor.is_floating_point() or tensor.is_complex()):
            tensor = tensor.float()
        return "tensor", tensor

    try:
        length = len(keys)
    except TypeError as exc:
        raise TypeError("content_keys must be a tensor or a sequence") from exc
    if length != n:
        raise ValueError(f"content_keys must contain exactly {n} rows")
    try:
        tensor = torch.as_tensor(keys, device=device)
    except (TypeError, ValueError):
        return "objects", list(keys)
    if tensor.ndim == 1:
        tensor = tensor.reshape(n, 1)
    else:
        tensor = tensor.reshape(n, -1)
    if not (tensor.is_floating_point() or tensor.is_complex()):
        tensor = tensor.float()
    return "tensor", tensor


def _content_distances(kind: str, keys: Any, query_index: int, support_indices: Tensor) -> Tensor:
    if kind == "tensor":
        difference = keys[support_indices] - keys[query_index]
        return torch.linalg.vector_norm(difference, dim=1)
    query_key = keys[query_index]
    return torch.as_tensor(
        [0.0 if key == query_key else 1.0 for key in (keys[index] for index in support_indices.tolist())],
        device=support_indices.device,
        dtype=torch.get_default_dtype(),
    )


@dataclass(frozen=True)
class ContentLODOInfo:
    """Diagnostics for content-conditioned LODO prototypes."""

    support_domains: frozenset[Hashable]
    query_domain: Hashable
    query_count: int
    support_count: int
    fallback_classes: frozenset[Hashable]
    weighted_classes: frozenset[Hashable]
    prototype_counts: Mapping[Hashable, int]
    unseen_query_classes: frozenset[Hashable] = frozenset()


def content_conditioned_lodo_loss(
    z: Tensor,
    y: Any,
    domain: Any,
    content_keys: Any,
    query_domain: Any = None,
    *,
    support_mask: Any | None = None,
    query_mask: Any | None = None,
    max_distance: float = 1.0,
    temperature: float = LODO_TEMPERATURE,
    key_temperature: float | None = None,
) -> tuple[Tensor, ContentLODOInfo]:
    """Use soft content-key neighbors for LODO prototypes with safe fallback."""

    z = _validate_z(z)
    temperature = _validate_temperature(temperature)
    max_distance = float(max_distance)
    if not math.isfinite(max_distance) or max_distance < 0.0:
        raise ValueError("max_distance must be finite and non-negative")
    key_temperature = (
        max(max_distance, 1e-6)
        if key_temperature is None
        else _validate_temperature(key_temperature, name="key_temperature")
    )
    labels = _class_tensor(y, device=z.device, n=z.shape[0])
    kind, prepared_keys = _prepare_content_keys(
        content_keys,
        device=z.device,
        n=z.shape[0],
    )
    domain_tensor, support, query, query_value = _lodo_masks(
        z,
        domain,
        query_domain=query_domain,
        support_mask=support_mask,
        query_mask=query_mask,
    )

    support_domains = frozenset(
        _python_scalar(value) for value in domain_tensor[support].detach().cpu().tolist()
    )
    support_indices = torch.nonzero(support, as_tuple=False).flatten()
    query_indices = torch.nonzero(query, as_tuple=False).flatten()
    support_classes = torch.unique(labels[support], sorted=True)
    query_classes = torch.unique(labels[query], sorted=True)
    support_class_values = tuple(
        _python_scalar(value) for value in support_classes.detach().cpu().tolist()
    )
    unseen = frozenset(
        _python_scalar(value)
        for value in query_classes.detach().cpu().tolist()
        if not bool((support_classes == value).any())
    )
    prototype_counts = {
        _python_scalar(class_value): int((support & labels.eq(class_value)).sum().item())
        for class_value in support_classes
    }

    fallback_classes: set[Hashable] = set()
    weighted_classes: set[Hashable] = set()
    losses: list[Tensor] = []
    if support_indices.numel() and query_indices.numel() and support_classes.numel():
        for query_index in query_indices.tolist():
            distances = _content_distances(kind, prepared_keys, query_index, support_indices)
            logits_for_query: list[Tensor] = []
            for class_value in support_classes:
                class_support = support_indices[labels[support_indices].eq(class_value)]
                class_distances = distances[labels[support_indices].eq(class_value)]
                close = class_distances.le(max_distance + 1e-12)
                if bool(close.any()):
                    close_distances = class_distances[close]
                    weights = F.softmax(-close_distances / key_temperature, dim=0)
                    class_rows = z[class_support[close]]
                    prototype = (weights.unsqueeze(1) * class_rows).sum(dim=0)
                    weighted_classes.add(_python_scalar(class_value))
                else:
                    class_rows = z[class_support]
                    prototype = class_rows.mean(dim=0)
                logits_for_query.append(F.normalize(prototype, dim=0, eps=_EPS))
            query_feature = F.normalize(z[query_index], dim=0, eps=_EPS)
            logits = query_feature @ torch.stack(logits_for_query).transpose(0, 1)
            target = None
            for class_index, class_value in enumerate(support_classes):
                if bool(labels[query_index].eq(class_value)):
                    target = class_index
                    break
            if target is not None:
                losses.append(F.cross_entropy((logits / temperature).reshape(1, -1), torch.tensor([target], device=z.device)))

    if query_indices.numel():
        fallback_classes = set(support_class_values) - weighted_classes

    info = ContentLODOInfo(
        support_domains=support_domains,
        query_domain=query_value,
        query_count=int(query.sum().item()),
        support_count=int(support.sum().item()),
        fallback_classes=frozenset(fallback_classes),
        weighted_classes=frozenset(weighted_classes),
        prototype_counts=prototype_counts,
        unseen_query_classes=unseen,
    )
    if not losses:
        return _zero_from(z), info
    return torch.stack(losses).mean(), info


@dataclass(frozen=True)
class CounterfactualLossInfo:
    """The four independently inspectable counterfactual terms."""

    cf_id: Tensor
    cf_inv: Tensor
    cf_env: Tensor
    style: Tensor
    total: Tensor

    @property
    def id_loss(self) -> Tensor:
        return self.cf_id

    @property
    def inv_loss(self) -> Tensor:
        return self.cf_inv

    @property
    def env_loss(self) -> Tensor:
        return self.cf_env

    @property
    def style_loss(self) -> Tensor:
        return self.style

    @property
    def total_loss(self) -> Tensor:
        return self.total


@dataclass(frozen=True)
class CounterfactualLossResult:
    """Counterfactual total plus components.

    Iteration yields ``(total, info)`` so callers can use the same unpacking
    convention as the other loss helpers while named component access remains
    available for logging and tests.
    """

    total: Tensor
    info: CounterfactualLossInfo
    cf_id: Tensor
    cf_inv: Tensor
    cf_env: Tensor
    style: Tensor

    @property
    def id_loss(self) -> Tensor:
        return self.cf_id

    @property
    def inv_loss(self) -> Tensor:
        return self.cf_inv

    @property
    def env_loss(self) -> Tensor:
        return self.cf_env

    @property
    def style_loss(self) -> Tensor:
        return self.style

    @property
    def total_loss(self) -> Tensor:
        return self.total

    @property
    def components(self) -> Mapping[str, Tensor]:
        return {
            "total": self.total,
            "cf_id": self.cf_id,
            "cf_inv": self.cf_inv,
            "cf_env": self.cf_env,
            "style": self.style,
        }

    def __iter__(self):
        yield self.total
        yield self.info

    def __getitem__(self, key: int | str) -> Any:
        if key == 0:
            return self.total
        if key == 1:
            return self.info
        if not isinstance(key, str):
            raise IndexError(key)
        aliases = {
            "total": self.total,
            "cf": self.total,
            "cf_id": self.cf_id,
            "CF-ID": self.cf_id,
            "id": self.cf_id,
            "cf_inv": self.cf_inv,
            "CF-INV": self.cf_inv,
            "inv": self.cf_inv,
            "cf_env": self.cf_env,
            "CF-ENV": self.cf_env,
            "env": self.cf_env,
            "style": self.style,
            "style_loss": self.style,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


def _cross_entropy_if_available(
    logits: Any,
    labels: Any,
    *,
    reference: Any,
    mask: Tensor | None = None,
) -> Tensor:
    if logits is None or labels is None:
        return _zero_from(reference)
    if not isinstance(logits, Tensor):
        return _zero_from(reference, logits)
    if logits.ndim != 2 or logits.shape[1] == 0:
        return _zero_from(reference, logits)
    labels = _as_vector(labels, device=logits.device, n=logits.shape[0], name="labels").long()
    values = F.cross_entropy(logits, labels, reduction="none")
    return _masked_mean(values, mask)


def _environment_cross_entropy(
    logits: Any,
    labels: Any,
    *,
    reference: Any,
    mask: Tensor | None,
) -> Tensor:
    if logits is None or labels is None:
        return _zero_from(reference, logits)
    if isinstance(logits, Mapping):
        if not isinstance(labels, Mapping):
            return _zero_from(reference, logits)
        terms = [
            _cross_entropy_if_available(
                logits[name],
                labels[name],
                reference=reference,
                mask=mask,
            )
            for name in logits.keys()
            if name in labels
        ]
    elif isinstance(logits, (tuple, list)):
        if not isinstance(labels, (tuple, list)):
            return _zero_from(reference, logits)
        terms = [
            _cross_entropy_if_available(item, label, reference=reference, mask=mask)
            for item, label in zip(logits, labels)
        ]
    else:
        terms = [
            _cross_entropy_if_available(logits, labels, reference=reference, mask=mask)
        ]
    if not terms:
        return _zero_from(reference, logits)
    return torch.stack(terms).mean()


def _style_loss(cf_h: Any, target_h: Any, *, mask: Tensor | None, reference: Any) -> Tensor:
    if cf_h is None or target_h is None:
        return _zero_from(reference, cf_h, target_h)
    if not isinstance(cf_h, Tensor) or not isinstance(target_h, Tensor):
        return _zero_from(reference, cf_h, target_h)
    if cf_h.shape != target_h.shape or cf_h.ndim < 2:
        raise ValueError("cf_h and target_h must have the same shape [N, ...]")
    cf_flat = cf_h.reshape(cf_h.shape[0], -1)
    target_flat = target_h.to(device=cf_h.device, dtype=cf_h.dtype).reshape(cf_h.shape[0], -1)
    cf_mean = cf_flat.mean(dim=1)
    target_mean = target_flat.mean(dim=1)
    cf_std = cf_flat.std(dim=1, unbiased=False)
    target_std = target_flat.std(dim=1, unbiased=False)
    values = (cf_mean - target_mean).abs() + (cf_std - target_std).abs()
    return _masked_mean(values, mask)


def counterfactual_losses(
    cf_logits: Any = None,
    labels: Any = None,
    *,
    output: Any = None,
    cf_z_id: Tensor | None = None,
    z_id: Tensor | None = None,
    cf_env_logits: Any = None,
    target_env: Any = None,
    cf_h: Tensor | None = None,
    target_h: Tensor | None = None,
    eta_inv: float = 1.0,
    eta_env: float = 1.0,
    style_weight: float = 1.0,
    mask: Any | None = None,
) -> CounterfactualLossResult:
    """Compute CF-ID, CF-INV, CF-ENV and style matching losses."""

    if output is None and cf_logits is not None and not isinstance(cf_logits, Tensor):
        output = cf_logits
        cf_logits = None
    if cf_logits is None:
        cf_logits = _get_field(
            output,
            "cf_logits",
            "counterfactual_logits",
            "counterfactual_common_logits",
        )
    if labels is None:
        labels = _get_field(output, "labels", "tx_labels", "y")
    if cf_z_id is None:
        cf_z_id = _get_field(
            output,
            "cf_z_id",
            "counterfactual_z_id",
            "counterfactual_identity",
        )
    if z_id is None:
        z_id = _get_field(output, "z_id", "identity_embedding")
    if cf_env_logits is None:
        cf_env_logits = _get_field(
            output,
            "cf_env_logits",
            "counterfactual_env_logits",
            "environment_logits",
        )
    if target_env is None:
        target_env = _get_field(
            output,
            "target_env",
            "target_environment",
            "target_environment_labels",
        )
    if cf_h is None:
        cf_h = _get_field(output, "cf_h", "counterfactual_feature")
    if target_h is None:
        target_h = _get_field(output, "target_h", "target_feature")

    required = (
        ("cf_logits", cf_logits),
        ("cf_z_id", cf_z_id),
        ("cf_env_logits", cf_env_logits),
        ("target_env", target_env),
        ("cf_h", cf_h),
        ("target_h", target_h),
        ("labels", labels),
        ("z_id", z_id),
    )
    missing = [name for name, value in required if value is None]
    if missing:
        if len(missing) == 1:
            required_text = missing[0]
        else:
            required_text = f"{', '.join(missing[:-1])}, and {missing[-1]}"
        raise ValueError(f"counterfactual requires {required_text}")

    eta_inv = _validate_nonnegative(eta_inv, name="eta_inv")
    eta_env = _validate_nonnegative(eta_env, name="eta_env")
    style_weight = _validate_nonnegative(style_weight, name="style_weight")
    reference = _first_tensor(cf_logits, cf_z_id, z_id, cf_env_logits, cf_h, target_h)
    if reference is None:
        reference = torch.tensor(0.0)
    row_mask = None
    if mask is not None:
        row_count = next(
            (item.shape[0] for item in (cf_logits, cf_z_id, z_id, cf_h, target_h) if isinstance(item, Tensor)),
            None,
        )
        if row_count is None:
            raise ValueError("mask requires at least one row-wise tensor")
        row_mask = _as_bool_mask(mask, device=reference.device, n=row_count, name="mask")

    cf_id = _cross_entropy_if_available(
        cf_logits,
        labels,
        reference=reference,
        mask=row_mask,
    )
    if cf_z_id is None or z_id is None:
        cf_inv = _zero_from(reference, cf_z_id, z_id)
    else:
        if cf_z_id.shape != z_id.shape or cf_z_id.ndim < 2:
            raise ValueError("cf_z_id and z_id must have the same shape [N, D]")
        values = 1.0 - F.cosine_similarity(cf_z_id, z_id.detach(), dim=-1, eps=_EPS)
        cf_inv = _masked_mean(values, row_mask)
    cf_env = _environment_cross_entropy(
        cf_env_logits,
        target_env,
        reference=reference,
        mask=row_mask,
    )
    style = _style_loss(cf_h, target_h, mask=row_mask, reference=reference)
    total = cf_id + eta_inv * cf_inv + eta_env * cf_env + style_weight * style
    info = CounterfactualLossInfo(
        cf_id=cf_id,
        cf_inv=cf_inv,
        cf_env=cf_env,
        style=style,
        total=total,
    )
    return CounterfactualLossResult(
        total=total,
        info=info,
        cf_id=cf_id,
        cf_inv=cf_inv,
        cf_env=cf_env,
        style=style,
    )


@dataclass(frozen=True)
class HDROGroup:
    """A group mask and an optional named or tensor parent."""

    mask: Tensor
    parent: str | Tensor | None = None


@dataclass(frozen=True)
class HDROInfo:
    raw_risks: Mapping[str, Tensor]
    shrunk_risks: Mapping[str, Tensor]
    parent_risks: Mapping[str, Tensor]
    group_counts: Mapping[str, int]
    active_groups: tuple[str, ...]
    shrinkage: Mapping[str, float]


@dataclass(frozen=True)
class _GroupSpec:
    mask: Tensor
    parent: str | Tensor | None = None


_FACTOR_ALIASES = {
    "rx": "receiver",
    "receiver": "receiver",
    "day": "day",
    "channel": "channel",
    "tx_rx": "tx_receiver",
    "tx_receiver": "tx_receiver",
    "tx_day": "tx_day",
    "tx_channel": "tx_channel",
}


def _group_family(name: str) -> str:
    return _FACTOR_ALIASES.get(name.split(":", 1)[0].lower(), name.split(":", 1)[0].lower())


def _group_name_equivalent(left: str, right: str) -> bool:
    left_parts = left.split(":")
    right_parts = right.split(":")
    if _group_family(left) != _group_family(right) or len(left_parts) != len(right_parts):
        return False
    return left_parts[1:] == right_parts[1:]


def _mask_from_indices(value: Any, *, device: torch.device, n: int, name: str) -> Tensor:
    indices = _as_tensor(value, device=device, name=name).reshape(-1)
    if indices.is_floating_point() or indices.is_complex():
        if not torch.equal(indices, indices.round()):
            raise ValueError(f"{name} indices must be integers")
    indices = indices.long()
    if bool(((indices < 0) | (indices >= n)).any()):
        raise ValueError(f"{name} contains an out-of-range index")
    mask = torch.zeros(n, device=device, dtype=torch.bool)
    return mask.scatter(0, indices, True)


def _direct_group_mask(value: Any, *, device: torch.device, n: int, name: str) -> Tensor:
    tensor = _as_tensor(value, device=device, name=name)
    if tensor.dtype == torch.bool:
        return _as_bool_mask(tensor, device=device, n=n, name=name)
    if tensor.ndim == 1 and tensor.numel() != n:
        return _mask_from_indices(tensor, device=device, n=n, name=name)
    if tensor.ndim == 1 and tensor.numel() == n and tensor.is_floating_point():
        raise ValueError(f"{name} must be a boolean mask or index sequence")
    if tensor.ndim == 1 and tensor.numel() == n:
        return _mask_from_indices(tensor, device=device, n=n, name=name)
    return _mask_from_indices(tensor, device=device, n=n, name=name)


def _add_factor_groups(
    specs: dict[str, _GroupSpec],
    family: str,
    labels: Any,
    *,
    device: torch.device,
    n: int,
) -> None:
    label_tensor = _as_vector(labels, device=device, n=n, name=family)
    for value in torch.unique(label_tensor, sorted=True):
        scalar = _python_scalar(value)
        specs[f"{family}:{scalar}"] = _GroupSpec(label_tensor.eq(value))


def _add_pair_groups(
    specs: dict[str, _GroupSpec],
    family: str,
    left: Any,
    right: Any,
    *,
    device: torch.device,
    n: int,
) -> None:
    left_tensor = _as_vector(left, device=device, n=n, name=f"{family}_left")
    right_tensor = _as_vector(right, device=device, n=n, name=f"{family}_right")
    pairs = torch.stack((left_tensor, right_tensor), dim=1)
    for pair in torch.unique(pairs, dim=0, sorted=True):
        left_value = _python_scalar(pair[0])
        right_value = _python_scalar(pair[1])
        specs[f"{family}:{left_value}:{right_value}"] = _GroupSpec(
            (left_tensor == pair[0]) & (right_tensor == pair[1])
        )


def _normalise_group_specs(groups: Any, *, device: torch.device, n: int) -> dict[str, _GroupSpec]:
    specs: dict[str, _GroupSpec] = {}
    if groups is None:
        return specs
    if isinstance(groups, Mapping):
        entries: Iterable[tuple[Any, Any]] = groups.items()
    else:
        parsed: list[tuple[Any, Any]] = []
        for entry in groups:
            if isinstance(entry, Mapping):
                parsed.append((entry.get("name"), entry))
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) >= 2:
                parsed.append((entry[0], entry[1:] if len(entry) > 2 else entry[1]))
            else:
                raise TypeError("groups must be a mapping or named group entries")
        entries = parsed

    for raw_name, value in entries:
        if raw_name is None:
            raise ValueError("every HDRO group needs a name")
        name = str(raw_name)
        family = _group_family(name)
        if family in {"receiver", "day", "channel"} and ":" not in name:
            if isinstance(value, Mapping) and "labels" in value:
                _add_factor_groups(specs, family, value["labels"], device=device, n=n)
            elif isinstance(value, Tensor) and value.ndim == 1 and value.numel() == n and value.dtype != torch.bool:
                _add_factor_groups(specs, family, value, device=device, n=n)
            else:
                try:
                    tensor_value = torch.as_tensor(value, device=device)
                except (TypeError, ValueError):
                    tensor_value = None
                if tensor_value is not None and tensor_value.ndim == 1 and tensor_value.numel() == n and tensor_value.dtype != torch.bool:
                    _add_factor_groups(specs, family, tensor_value, device=device, n=n)
                else:
                    specs[name] = _GroupSpec(_direct_group_mask(value, device=device, n=n, name=name))
            continue
        if family in {"tx_receiver", "tx_day", "tx_channel"} and ":" not in name:
            pair = value
            if isinstance(value, Mapping):
                left = value.get("tx", value.get("tx_ids"))
                right_name = family.removeprefix("tx_")
                right = value.get(right_name, value.get(f"{right_name}_ids"))
                if left is not None and right is not None:
                    _add_pair_groups(specs, family, left, right, device=device, n=n)
                    continue
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2:
                try:
                    _add_pair_groups(specs, family, pair[0], pair[1], device=device, n=n)
                    continue
                except (TypeError, ValueError):
                    pass

        parent: str | Tensor | None = None
        actual_value = value
        if isinstance(value, HDROGroup):
            actual_value = value.mask
            parent = value.parent
        elif isinstance(value, Mapping):
            parent = value.get("parent")
            if "mask" in value:
                actual_value = value["mask"]
            elif "indices" in value:
                actual_value = value["indices"]
                actual_value = _mask_from_indices(actual_value, device=device, n=n, name=name)
            elif "labels" in value and ":" not in name:
                _add_factor_groups(specs, family, value["labels"], device=device, n=n)
                continue
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, Tensor)):
            if len(value) == 2 and isinstance(value[1], (str, Tensor)):
                actual_value = value[0]
                parent = value[1]
        if isinstance(parent, Tensor):
            parent = parent.to(device=device)
        specs[name] = _GroupSpec(
            _direct_group_mask(actual_value, device=device, n=n, name=name),
            parent=parent,
        )
    return specs


def _inferred_parent(name: str, specs: Mapping[str, _GroupSpec]) -> str | None:
    parts = name.split(":")
    family = _group_family(name)
    if len(parts) < 3 or family not in {"tx_receiver", "tx_day", "tx_channel"}:
        return None
    tx_value, factor_value = parts[-2], parts[-1]
    factor_family = family.removeprefix("tx_")
    candidates = (
        f"{factor_family}:{factor_value}",
        f"receiver:{factor_value}" if factor_family == "receiver" else "",
        f"tx:{tx_value}",
    )
    for candidate in candidates:
        if not candidate:
            continue
        for existing in specs:
            if _group_name_equivalent(candidate, existing):
                return existing
    return None


def _parent_risk(
    parent: str | Tensor | None,
    *,
    specs: Mapping[str, _GroupSpec],
    raw_risks: Mapping[str, Tensor],
    losses: Tensor,
    current_name: str,
) -> Tensor | None:
    if isinstance(parent, Tensor):
        parent_mask = parent.to(device=losses.device, dtype=torch.bool).reshape(-1)
        if parent_mask.numel() != losses.shape[0] or not bool(parent_mask.any()):
            return None
        return losses[parent_mask].mean()
    if parent is None:
        parent = _inferred_parent(current_name, specs)
    if parent is None:
        return None
    for existing_name, value in specs.items():
        if existing_name == parent or _group_name_equivalent(existing_name, str(parent)):
            return raw_risks.get(existing_name)
    return None


def hierarchical_dro_loss(
    per_sample_loss: Tensor,
    groups: Any,
    *,
    kappa: float = 8.0,
    tau: float = 0.25,
    min_group: int = 4,
) -> tuple[Tensor, HDROInfo]:
    """Aggregate raw group risks after small-child parent shrinkage."""

    if not isinstance(per_sample_loss, Tensor):
        raise TypeError("per_sample_loss must be a torch.Tensor")
    if per_sample_loss.ndim != 1:
        raise ValueError("per_sample_loss must have shape [N]")
    if not per_sample_loss.is_floating_point():
        per_sample_loss = per_sample_loss.float()
    kappa = _validate_nonnegative(kappa, name="kappa")
    tau = _validate_temperature(tau, name="tau")
    if isinstance(min_group, bool) or int(min_group) < 1:
        raise ValueError("min_group must be a positive integer")
    min_group = int(min_group)
    specs = _normalise_group_specs(
        groups,
        device=per_sample_loss.device,
        n=per_sample_loss.shape[0],
    )
    raw_risks: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    for name, spec in specs.items():
        mask = spec.mask.to(device=per_sample_loss.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != per_sample_loss.shape[0]:
            raise ValueError(f"group {name} has the wrong number of rows")
        count = int(mask.sum().item())
        if count:
            counts[name] = count
            raw_risks[name] = per_sample_loss[mask].mean()

    shrunk_risks: dict[str, Tensor] = {}
    parent_risks: dict[str, Tensor] = {}
    shrinkage: dict[str, float] = {}
    for name, raw in raw_risks.items():
        parent = _parent_risk(
            specs[name].parent,
            specs=specs,
            raw_risks=raw_risks,
            losses=per_sample_loss,
            current_name=name,
        )
        count = counts[name]
        if parent is not None and count < min_group:
            n_tensor = raw.new_tensor(float(count))
            k_tensor = raw.new_tensor(kappa)
            shrunk = (n_tensor * raw + k_tensor * parent) / (n_tensor + k_tensor).clamp_min(_EPS)
            parent_risks[name] = parent
            shrinkage[name] = float(kappa / (count + kappa)) if kappa else 0.0
        else:
            shrunk = raw
            shrinkage[name] = 0.0
        shrunk_risks[name] = shrunk

    info = HDROInfo(
        raw_risks=raw_risks,
        shrunk_risks=shrunk_risks,
        parent_risks=parent_risks,
        group_counts=counts,
        active_groups=tuple(raw_risks),
        shrinkage=shrinkage,
    )
    if not shrunk_risks:
        return _zero_from(per_sample_loss), info
    values = torch.stack(tuple(shrunk_risks.values()))
    return tau * torch.logsumexp(values / tau, dim=0), info


@dataclass(frozen=True)
class HCFDGLossInfo:
    """Component diagnostics returned by :func:`compose_hcfdg_loss`."""

    lodo: LODOInfo | ContentLODOInfo | None
    counterfactual: CounterfactualLossInfo | None
    hdro: HDROInfo | None
    active_components: frozenset[str]


@dataclass(frozen=True)
class HCFDGLossResult:
    total: Tensor
    id_loss: Tensor
    lodo_loss: Tensor
    cf_loss: Tensor
    hdro_loss: Tensor
    csd_loss: Tensor
    fac_loss: Tensor
    info: HCFDGLossInfo

    @property
    def counterfactual_loss(self) -> Tensor:
        return self.cf_loss

    @property
    def loss(self) -> Tensor:
        return self.total

    @property
    def total_loss(self) -> Tensor:
        return self.total

    @property
    def components(self) -> Mapping[str, Tensor]:
        return {
            "id": self.id_loss,
            "lodo": self.lodo_loss,
            "counterfactual": self.cf_loss,
            "hdro": self.hdro_loss,
            "csd": self.csd_loss,
            "fac": self.fac_loss,
            "total": self.total,
        }

    def __iter__(self):
        yield self.total
        yield self.info

    def __getitem__(self, key: int | str) -> Any:
        if key == 0:
            return self.total
        if key == 1:
            return self.info
        if not isinstance(key, str):
            raise IndexError(key)
        values = {
            "total": self.total,
            "id": self.id_loss,
            "id_loss": self.id_loss,
            "lodo": self.lodo_loss,
            "lodo_loss": self.lodo_loss,
            "cf": self.cf_loss,
            "cf_loss": self.cf_loss,
            "hdro": self.hdro_loss,
            "hdro_loss": self.hdro_loss,
            "csd": self.csd_loss,
            "csd_loss": self.csd_loss,
            "fac": self.fac_loss,
            "fac_loss": self.fac_loss,
        }
        try:
            return values[key]
        except KeyError as exc:
            raise KeyError(key) from exc


def _component_enabled(
    name: str,
    explicit: bool | None,
    *,
    config: Any,
    default: bool,
) -> bool:
    if explicit is not None:
        return bool(explicit)
    if config is not None:
        value = _get_field(config, f"use_{name}", default=None)
        if value is not None:
            return bool(value)
    return default


def _common_specific_loss(specific_logits: Any, *, labels: Any) -> Tensor:
    if not isinstance(specific_logits, Tensor) or labels is None:
        raise ValueError("CSD requires specific_logits and labels")
    labels_tensor = _as_vector(
        labels,
        device=specific_logits.device,
        n=specific_logits.shape[0],
        name="labels",
    ).long()
    return F.cross_entropy(specific_logits, labels_tensor)


def _factor_auxiliary_loss(
    output: Any,
    *,
    receiver: Any,
    day: Any,
    channel: Any,
    labels: Any,
    reference: Any,
) -> Tensor:
    pairs = (
        ("receiver", _get_field(output, "receiver_logits"), receiver),
        ("day", _get_field(output, "day_logits"), day),
        ("channel", _get_field(output, "channel_logits"), channel),
        (
            "conditional_receiver",
            _get_field(output, "conditional_receiver_logits"),
            receiver,
        ),
        ("tx_from_env", _get_field(output, "tx_from_env_logits"), labels),
    )
    if any(not isinstance(logits, Tensor) or target is None for _, logits, target in pairs):
        raise ValueError(
            "FAC requires logits and targets for receiver, day, channel, "
            "conditional_receiver, and tx_from_env"
        )
    terms = [
        _cross_entropy_if_available(logits, target, reference=reference)
        for _, logits, target in pairs
    ]
    return torch.stack(terms).mean()


def compose_hcfdg_loss(
    model_output: Any = None,
    labels: Any = None,
    domain: Any = None,
    *,
    output: Any = None,
    tx_labels: Any = None,
    receiver: Any = None,
    day: Any = None,
    channel: Any = None,
    env_meta: Any = None,
    content_keys: Any = None,
    query_domain: Any = None,
    support_mask: Any | None = None,
    query_mask: Any | None = None,
    groups: Any = None,
    per_sample_loss: Tensor | None = None,
    config: Any = None,
    use_lodo: bool | None = None,
    use_content_conditioning: bool | None = None,
    use_counterfactual: bool | None = None,
    use_hdro: bool | None = None,
    use_csd: bool | None = None,
    use_fac: bool | None = None,
    counterfactual_mode: str | None = None,
    cf_kwargs: Mapping[str, Any] | None = None,
    counterfactual_output: Any = None,
    cf_logits: Any = None,
    cf_z_id: Tensor | None = None,
    cf_env_logits: Any = None,
    target_env: Any = None,
    cf_h: Tensor | None = None,
    target_h: Tensor | None = None,
    z_id_for_cf: Tensor | None = None,
    id_loss: Any = None,
    lodo_loss: Any = None,
    cf_loss: Any = None,
    hdro_loss: Any = None,
    csd_loss: Any = None,
    fac_loss: Any = None,
) -> HCFDGLossResult:
    """Compose the six explicit HCF-DG objective components.

    ``output`` may be the typed ``HCFDGOutput`` from the model task or any
    object exposing the same field names.  Precomputed component arguments
    are accepted for trainer integrations that already computed a component;
    otherwise the component is derived from the output and labels here.
    """

    output = model_output if output is None else output
    labels = labels if labels is not None else tx_labels
    if receiver is None:
        receiver = _get_field(env_meta, "receiver")
    if day is None:
        day = _get_field(env_meta, "day")
    if channel is None:
        channel = _get_field(env_meta, "channel")
    if domain is None and receiver is not None:
        domain = receiver

    common_logits = output if isinstance(output, Tensor) else _get_field(output, "common_logits")
    z_id = _get_field(output, "z_id")
    reference = _first_tensor(common_logits, z_id, labels, domain, receiver, day, channel)
    if reference is None:
        reference = torch.tensor(0.0)

    if counterfactual_mode is None:
        counterfactual_mode = _get_field(config, "counterfactual_mode", default="off")
    cf_default = str(counterfactual_mode).strip().lower() not in {"", "off", "none"}
    use_lodo = _component_enabled(
        "lodo", use_lodo, config=config, default=True
    )
    use_content_conditioning = _component_enabled(
        "content_conditioning",
        use_content_conditioning,
        config=config,
        default=False,
    )
    use_counterfactual = _component_enabled(
        "counterfactual",
        use_counterfactual,
        config=config,
        default=cf_default,
    )
    use_hdro = _component_enabled(
        "hdro", use_hdro, config=config, default=False
    )
    use_csd = _component_enabled(
        "csd", use_csd, config=config, default=False
    )
    use_fac = _component_enabled(
        "fac", use_fac, config=config, default=False
    )

    if id_loss is None:
        id_value = _cross_entropy_if_available(common_logits, labels, reference=reference)
    else:
        id_value = _coerce_scalar_loss(id_loss, reference=reference)

    lodo_info: LODOInfo | ContentLODOInfo | None = None
    if not use_lodo:
        lodo_value = _zero_from(reference)
    elif lodo_loss is not None:
        lodo_value = _coerce_scalar_loss(lodo_loss, reference=reference)
    elif z_id is None or labels is None or domain is None:
        raise ValueError("LODO requires z_id, labels, and domain")
    elif query_domain is None and query_mask is None:
        raise ValueError("LODO requires query_domain or query_mask")
    elif use_content_conditioning and content_keys is None:
        raise ValueError("content-conditioned LODO requires content_keys")
    elif use_content_conditioning:
        lodo_value, lodo_info = content_conditioned_lodo_loss(
            z_id,
            labels,
            domain,
            content_keys,
            query_domain=query_domain,
            support_mask=support_mask,
            query_mask=query_mask,
        )
    else:
        lodo_value, lodo_info = lodo_prototype_loss(
            z_id,
            labels,
            domain,
            query_domain=query_domain,
            support_mask=support_mask,
            query_mask=query_mask,
        )

    cf_info: CounterfactualLossInfo | None = None
    if not use_counterfactual:
        cf_value = _zero_from(reference)
    elif cf_loss is not None:
        cf_value = _coerce_scalar_loss(cf_loss, reference=reference)
    else:
        cf_args = dict(cf_kwargs or {})
        explicit_cf = {
            "cf_logits": cf_logits,
            "cf_z_id": cf_z_id,
            "cf_env_logits": cf_env_logits,
            "target_env": target_env,
            "cf_h": cf_h,
            "target_h": target_h,
        }
        if z_id_for_cf is not None:
            explicit_cf["z_id"] = z_id_for_cf
        for key, value in explicit_cf.items():
            if value is not None:
                cf_args.setdefault(key, value)
        cf_source = counterfactual_output
        if cf_source is None:
            cf_source = output
        cf_result = counterfactual_losses(output=cf_source, labels=labels, **cf_args)
        cf_value = cf_result.total
        cf_info = cf_result.info

    hdro_info: HDROInfo | None = None
    if not use_hdro:
        hdro_value = _zero_from(reference)
    elif hdro_loss is not None:
        hdro_value = _coerce_scalar_loss(hdro_loss, reference=reference)
    else:
        if per_sample_loss is None:
            if isinstance(common_logits, Tensor) and labels is not None:
                labels_tensor = _as_vector(labels, device=common_logits.device, n=common_logits.shape[0], name="labels").long()
                per_sample_loss = F.cross_entropy(common_logits, labels_tensor, reduction="none")
        if per_sample_loss is None:
            raise ValueError("HDRO requires per_sample_loss or common_logits and labels")
        else:
            if groups is None:
                if receiver is None or day is None or channel is None:
                    raise ValueError("HDRO requires groups or receiver, day, and channel labels")
                if labels is None:
                    raise ValueError("HDRO requires labels for TX-crossed groups")
                groups = {
                    "receiver": receiver,
                    "day": day,
                    "channel": channel,
                    "tx_receiver": (labels, receiver),
                    "tx_day": (labels, day),
                    "tx_channel": (labels, channel),
                }
            if not groups:
                raise ValueError("HDRO requires a non-empty groups definition")
            hdro_value, hdro_info = hierarchical_dro_loss(per_sample_loss, groups)

    if not use_csd:
        csd_value = _zero_from(reference)
    elif csd_loss is not None:
        csd_value = _coerce_scalar_loss(csd_loss, reference=reference)
    else:
        csd_value = _common_specific_loss(
            _get_field(output, "specific_logits"),
            labels=labels,
        )

    if not use_fac:
        fac_value = _zero_from(reference)
    elif fac_loss is not None:
        fac_value = _coerce_scalar_loss(fac_loss, reference=reference)
    else:
        fac_value = _factor_auxiliary_loss(
            output,
            receiver=receiver,
            day=day,
            channel=channel,
            labels=labels,
            reference=reference,
        )

    total = (
        id_value
        + LODO_WEIGHT * lodo_value
        + COUNTERFACTUAL_WEIGHT * cf_value
        + HDRO_WEIGHT * hdro_value
        + CSD_WEIGHT * csd_value
        + FAC_WEIGHT * fac_value
    )
    active = frozenset(
        name
        for name, enabled in (
            ("id", True),
            ("lodo", use_lodo),
            ("counterfactual", use_counterfactual),
            ("hdro", use_hdro),
            ("csd", use_csd),
            ("fac", use_fac),
        )
        if enabled
    )
    info = HCFDGLossInfo(
        lodo=lodo_info,
        counterfactual=cf_info,
        hdro=hdro_info,
        active_components=active,
    )
    return HCFDGLossResult(
        total=total,
        id_loss=id_value,
        lodo_loss=lodo_value,
        cf_loss=cf_value,
        hdro_loss=hdro_value,
        csd_loss=csd_value,
        fac_loss=fac_value,
        info=info,
    )


def _coerce_scalar_loss(value: Any, *, reference: Any) -> Tensor:
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.to(device=_first_tensor(reference).device if _first_tensor(reference) is not None else value.device)
        return value.mean()
    reference_tensor = _first_tensor(reference)
    if reference_tensor is None:
        return torch.as_tensor(value, dtype=torch.get_default_dtype())
    return reference_tensor.new_tensor(float(value))


__all__ = [
    "COUNTERFACTUAL_WEIGHT",
    "CF_WEIGHT",
    "CSD_WEIGHT",
    "ContentLODOInfo",
    "CounterfactualLossInfo",
    "CounterfactualLossResult",
    "FAC_WEIGHT",
    "HCFDG_LOSS_WEIGHTS",
    "HDROGroup",
    "HDROInfo",
    "HDRO_WEIGHT",
    "HCFDGLossInfo",
    "HCFDGLossResult",
    "LODOInfo",
    "LODO_TEMPERATURE",
    "LODO_WEIGHT",
    "content_conditioned_lodo_loss",
    "compose_hcfdg_loss",
    "counterfactual_losses",
    "hierarchical_dro_loss",
    "lodo_prototype_loss",
]
