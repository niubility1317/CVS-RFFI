"""Support-only sparse encoder adaptation for frozen ADV3B02 decisions.

The existing checkpoint classifier is never replaced or trained.  Its labeled
support loss is differentiated through into a small allowlisted set of
identity-encoder parameters.  Query inference is exposed separately and is
strictly read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_ADAPTATION_STEPS = 40
MAX_TRAINABLE_FRACTION = 0.01
_REQUIRED_CONTEXT = (
    "protocol_schema",
    "phase2_data_status",
    "capsule_id",
    "split_id",
)
_FORBIDDEN_CONTEXT_TOKENS = (
    "source",
    "clean",
    "query",
    "truth",
    "role",
    "cache",
)
_CLASSIFIER_TOKENS = (
    ".cls_head.",
    ".classifier.",
    ".adv_head.",
    ".dom_head.",
    ".domain_head.",
)

_C1_NORM_AFFINE_NAMES = frozenset(
    f"id_backbone.{block}.norm.{affine}"
    for block in ("t1", "t2", "t3", "f1", "f2", "f3", "pa_b1", "pa_b2", "pa_b3")
    for affine in ("weight", "bias")
)
_C2_NORM_GATE_NAMES = _C1_NORM_AFFINE_NAMES | frozenset(
    {
        "id_backbone.freq_gate.conv.weight",
        "id_backbone.freq_gate.conv.bias",
        "id_backbone.pa_gate.net.weight",
        "id_backbone.pa_gate.net.bias",
    }
)
_C3_NORM_GATE_FPROJ_NAMES = _C2_NORM_GATE_NAMES | frozenset(
    {
        "id_backbone.f_proj.weight",
        "id_backbone.f_proj.bias",
    }
)
_CANDIDATE_PARAMETER_NAMES = {
    "c1_norm_affine": _C1_NORM_AFFINE_NAMES,
    "c2_norm_gates": _C2_NORM_GATE_NAMES,
    "c3_norm_gates_fproj": _C3_NORM_GATE_FPROJ_NAMES,
}


class SparseEncoderAdaptationError(ValueError):
    """Raised when the support-only adaptation contract is violated."""


@dataclass(frozen=True)
class SparseEncoderAdaptationConfig:
    candidate: str = "c1_norm_affine"
    steps: int = 20
    learning_rate: float = 5.0e-4
    feature_anchor_weight: float = 0.05
    parameter_anchor_weight: float = 1.0e-4
    gradient_clip: float = 1.0
    max_trainable_fraction: float = MAX_TRAINABLE_FRACTION


@dataclass(frozen=True)
class SparseEncoderAdaptationAudit:
    method_id: str
    candidate: str
    gradient_updates: int
    support_samples: int
    support_class_count: int
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    trainable_parameter_names: tuple[str, ...]
    classifier_parameters_changed: int
    loss_trace: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class EnrollmentSupport:
    iq: torch.Tensor
    class_indices: torch.Tensor
    rank_within_class: torch.Tensor
    tokens: tuple[str, ...]
    checkpoint_sha256: str
    scenario: str


def load_validated_enrollment_support(
    enrollment_root: str | Path,
    *,
    scenario: str,
    k_shot: int,
    context: Mapping[str, Any],
) -> EnrollmentSupport:
    """Load a nested-K prefix from an already VALIDATED_ONCE support package.

    The function opens only the enrollment manifest and the requested support
    NPZ.  It does not hash or revalidate the fixed received-IQ capsule and has
    no query, truth, source, or clean-data input surface.
    """

    _validate_context(context)
    root = Path(enrollment_root)
    if root.name != "enrollment_only" or not root.is_dir():
        raise SparseEncoderAdaptationError(
            "support root must be an existing enrollment_only directory"
        )
    manifest_path = root / "package_manifest.json"
    if not manifest_path.is_file():
        raise SparseEncoderAdaptationError("enrollment package manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, Mapping):
        raise SparseEncoderAdaptationError("enrollment manifest must be a mapping")
    if manifest.get("profile") != "enrollment_only" or str(
        manifest.get("stage", "")
    ).lower() not in {"stage2b", "stage2-b"}:
        raise SparseEncoderAdaptationError("enrollment manifest profile/stage drift")
    for key in ("protocol_schema", "phase2_data_status", "capsule_id", "split_id"):
        if key not in manifest or str(manifest[key]) != str(context[key]):
            raise SparseEncoderAdaptationError(
                f"enrollment manifest/context mismatch: {key}"
            )
    explicitly_forbidden = (
        "source_cache_path",
        "source_sample_path",
        "source_feature_path",
        "clean_sample_path",
        "clean_cache_path",
        "query_truth",
        "query_role",
        "query_labels",
        "scorer_path",
    )
    for key in explicitly_forbidden:
        if key in manifest and manifest[key] not in (None, False, "", 0, [], {}):
            raise SparseEncoderAdaptationError(
                f"forbidden enrollment manifest field: {key}"
            )
    for member in manifest.get("members", ()):  # member list is metadata only
        kind = str(member.get("kind", "")).lower()
        relative_path = str(member.get("relative_path", "")).lower()
        if any(
            token in kind or token in relative_path
            for token in ("query", "truth", "score", "scorer", "source", "clean")
        ):
            raise SparseEncoderAdaptationError(
                "forbidden non-support member in enrollment manifest"
            )
    scenarios = tuple(str(value) for value in manifest.get("target_channel_scenarios", ()))
    if str(scenario) not in scenarios:
        raise SparseEncoderAdaptationError("requested scenario is not in enrollment")
    maximum_k = int(manifest.get("support_pool_max_k", manifest.get("k_shot", 0)))
    requested_k = int(k_shot)
    if requested_k < 1 or requested_k > maximum_k:
        raise SparseEncoderAdaptationError("k_shot exceeds the validated support pool")
    checkpoint_sha256 = str(manifest.get("phase1_checkpoint_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise SparseEncoderAdaptationError("checkpoint SHA256 binding is missing")

    support_path = root / f"support_{scenario}.npz"
    if not support_path.is_file():
        raise SparseEncoderAdaptationError("requested support artifact is missing")
    with np.load(support_path, allow_pickle=False) as payload:
        required = {
            "support_leo_weak_iq",
            "support_class_indices",
            "support_rank_within_class",
            "support_tokens",
        }
        if not required.issubset(payload.files):
            raise SparseEncoderAdaptationError("support artifact fields are incomplete")
        if any(
            any(token in key.lower() for token in ("query", "truth", "role", "source", "clean"))
            for key in payload.files
        ):
            raise SparseEncoderAdaptationError("support artifact contains forbidden fields")
        iq = np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)
        class_indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
        ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
        tokens = np.asarray(payload["support_tokens"]).astype(str)
    if (
        iq.ndim != 3
        or iq.shape[1] != 2
        or class_indices.ndim != 1
        or ranks.ndim != 1
        or tokens.ndim != 1
        or not (len(iq) == len(class_indices) == len(ranks) == len(tokens))
        or not np.isfinite(iq).all()
    ):
        raise SparseEncoderAdaptationError("support artifact row alignment drift")
    selected: list[int] = []
    for class_index in sorted(np.unique(class_indices).tolist()):
        positions = np.flatnonzero(class_indices == int(class_index))
        ordered = positions[np.argsort(ranks[positions], kind="stable")]
        prefix = ordered[ranks[ordered] < requested_k]
        if len(prefix) != requested_k or set(ranks[prefix].tolist()) != set(
            range(requested_k)
        ):
            raise SparseEncoderAdaptationError(
                "validated support package does not contain the requested nested-K prefix"
            )
        selected.extend(int(value) for value in prefix)
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_tokens = tuple(str(value) for value in tokens[selected_array].tolist())
    if len(set(selected_tokens)) != len(selected_tokens):
        raise SparseEncoderAdaptationError("support physical tokens are not unique")
    return EnrollmentSupport(
        iq=torch.as_tensor(iq[selected_array].copy(), dtype=torch.float32),
        class_indices=torch.as_tensor(
            class_indices[selected_array].copy(), dtype=torch.long
        ),
        rank_within_class=torch.as_tensor(ranks[selected_array].copy(), dtype=torch.long),
        tokens=selected_tokens,
        checkpoint_sha256=checkpoint_sha256,
        scenario=str(scenario),
    )


def _validate_context(context: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED_CONTEXT if key not in context]
    if missing:
        raise SparseEncoderAdaptationError(
            f"missing Phase2 context fields: {missing}"
        )
    for raw_key in context:
        key = str(raw_key).strip().lower()
        if any(token in key for token in _FORBIDDEN_CONTEXT_TOKENS):
            raise SparseEncoderAdaptationError(
                f"forbidden Phase2 adaptation context field: {raw_key}"
            )
    if str(context["protocol_schema"]) != "p2_min_v1":
        raise SparseEncoderAdaptationError("protocol_schema must be p2_min_v1")
    if str(context["phase2_data_status"]) != "VALIDATED_ONCE":
        raise SparseEncoderAdaptationError(
            "phase2_data_status must be VALIDATED_ONCE"
        )
    for key in ("capsule_id", "split_id"):
        if not str(context[key]).strip():
            raise SparseEncoderAdaptationError(f"{key} must be nonempty")


def _canonical_parameter_name(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def _is_classifier_parameter(name: str) -> bool:
    lowered = f".{_canonical_parameter_name(name).lower()}"
    return any(token in lowered for token in _CLASSIFIER_TOKENS)


def _candidate_allows(candidate: str, name: str) -> bool:
    canonical = _canonical_parameter_name(name).lower()
    allowed = _CANDIDATE_PARAMETER_NAMES.get(str(candidate))
    if allowed is None:
        raise SparseEncoderAdaptationError(
            f"unknown sparse encoder candidate: {candidate}"
        )
    if not canonical.startswith("id_backbone.") or _is_classifier_parameter(name):
        return False
    return canonical in allowed


def _select_parameters(
    model: nn.Module,
    config: SparseEncoderAdaptationConfig,
) -> tuple[list[tuple[str, nn.Parameter]], int, float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expected = _CANDIDATE_PARAMETER_NAMES.get(str(config.candidate))
    if expected is None:
        raise SparseEncoderAdaptationError(
            f"unknown sparse encoder candidate: {config.candidate}"
        )
    available = {
        _canonical_parameter_name(name).lower()
        for name, _parameter in model.named_parameters()
    }
    missing = sorted(expected - available)
    if missing:
        raise SparseEncoderAdaptationError(
            "candidate is missing declared encoder parameters: " + ",".join(missing)
        )
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _candidate_allows(str(config.candidate), name)
    ]
    if not selected:
        raise SparseEncoderAdaptationError(
            f"candidate {config.candidate!r} selected no encoder parameters"
        )
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for _, parameter in selected))
    fraction = float(trainable / max(total, 1))
    configured_cap = float(config.max_trainable_fraction)
    if configured_cap <= 0.0 or configured_cap > MAX_TRAINABLE_FRACTION:
        raise SparseEncoderAdaptationError(
            "max_trainable_fraction cannot exceed the fixed 1% cap"
        )
    if fraction > configured_cap:
        raise SparseEncoderAdaptationError(
            f"selected encoder parameters exceed the 1% cap: {fraction:.6f}"
        )
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return selected, total, fraction


def _feature_and_logits(
    model: nn.Module, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "id_backbone") and callable(getattr(model, "_pick_z_id", None)):
        from cvsrffi.identity_only_forward import identity_only_feature_forward

        identity_only = identity_only_feature_forward(model, rows, "z_id")
        if identity_only is not None:
            return identity_only
    output = model(rows, y_tx=None, grl_lambda=1.0, return_aux=True)
    if not isinstance(output, Mapping):
        raise SparseEncoderAdaptationError("model output must be a mapping")
    features = output.get("z_id")
    logits = output.get("tx_logits", output.get("logits"))
    if not torch.is_tensor(features) or not torch.is_tensor(logits):
        raise SparseEncoderAdaptationError(
            "model output must contain tensor z_id and tx_logits/logits"
        )
    return features.float(), logits.float()


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise SparseEncoderAdaptationError("model has no parameters") from exc


def _validate_support(
    support_iq: torch.Tensor, support_labels: torch.Tensor
) -> None:
    if not torch.is_tensor(support_iq) or support_iq.ndim != 3:
        raise SparseEncoderAdaptationError("support_iq must be a [N,2,L] tensor")
    if support_iq.shape[0] < 1 or support_iq.shape[1] != 2:
        raise SparseEncoderAdaptationError("support_iq must contain [N,2,L] rows")
    if not torch.isfinite(support_iq).all():
        raise SparseEncoderAdaptationError("support_iq contains non-finite values")
    if not torch.is_tensor(support_labels) or support_labels.ndim != 1:
        raise SparseEncoderAdaptationError("support_labels must be a vector")
    if support_labels.shape[0] != support_iq.shape[0]:
        raise SparseEncoderAdaptationError("support IQ and labels must align")
    if support_labels.numel() < 1 or int(support_labels.min().item()) < 0:
        raise SparseEncoderAdaptationError("support labels must be nonnegative")


def adapt_on_target_support(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    context: Mapping[str, Any],
    config: SparseEncoderAdaptationConfig = SparseEncoderAdaptationConfig(),
) -> SparseEncoderAdaptationAudit:
    """Update only allowlisted identity-encoder parameters using target support.

    This API intentionally has no query argument.  The original checkpoint
    classifier remains frozen and supplies the differentiable decision loss.
    """

    _validate_context(context)
    _validate_support(support_iq, support_labels)
    steps = int(config.steps)
    if steps < 1 or steps > MAX_ADAPTATION_STEPS:
        raise SparseEncoderAdaptationError(
            f"adaptation steps must be in [1, {MAX_ADAPTATION_STEPS}]"
        )
    if float(config.learning_rate) <= 0.0:
        raise SparseEncoderAdaptationError("learning_rate must be positive")
    if float(config.feature_anchor_weight) < 0.0:
        raise SparseEncoderAdaptationError("feature_anchor_weight must be nonnegative")
    if float(config.parameter_anchor_weight) < 0.0:
        raise SparseEncoderAdaptationError("parameter_anchor_weight must be nonnegative")
    if float(config.gradient_clip) <= 0.0:
        raise SparseEncoderAdaptationError("gradient_clip must be positive")

    device = _model_device(model)
    rows = support_iq.detach().to(device=device, dtype=torch.float32)
    labels = support_labels.detach().to(device=device, dtype=torch.long)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected, total_parameters, trainable_fraction = _select_parameters(model, config)
    for _, parameter in selected:
        parameter.requires_grad_(False)
    with torch.enable_grad():
        reference_features, reference_logits = _feature_and_logits(model, rows)
    reference_features = reference_features.detach()
    if reference_logits.ndim != 2 or int(labels.max().item()) >= reference_logits.shape[1]:
        raise SparseEncoderAdaptationError(
            "support label is outside the frozen checkpoint classifier"
        )

    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    classifier_names = tuple(
        name for name, _ in model.named_parameters() if _is_classifier_parameter(name)
    )
    for _, parameter in selected:
        parameter.requires_grad_(True)
    selected_initial = {
        name: parameter.detach().clone() for name, parameter in selected
    }
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in selected],
        lr=float(config.learning_rate),
    )
    trace: list[Mapping[str, float]] = []
    completed_updates = 0
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            features, logits = _feature_and_logits(model, rows)
            classification_loss = F.cross_entropy(logits, labels)
            feature_anchor = (
                1.0
                - F.cosine_similarity(
                    features.float(), reference_features.float(), dim=1
                )
            ).mean()
            parameter_anchor_terms = [
                (parameter - selected_initial[name]).float().square().mean()
                for name, parameter in selected
            ]
            parameter_anchor = torch.stack(parameter_anchor_terms).mean()
            loss = (
                classification_loss
                + float(config.feature_anchor_weight) * feature_anchor
                + float(config.parameter_anchor_weight) * parameter_anchor
            )
            if not torch.isfinite(loss):
                raise SparseEncoderAdaptationError("non-finite support adaptation loss")
            support_gradients = torch.autograd.grad(
                classification_loss
                + float(config.feature_anchor_weight) * feature_anchor,
                [parameter for _, parameter in selected],
                retain_graph=True,
                allow_unused=True,
            )
            missing_support_gradients = [
                name
                for (name, _parameter), gradient in zip(selected, support_gradients)
                if gradient is None
            ]
            if missing_support_gradients:
                raise SparseEncoderAdaptationError(
                    "selected parameters received no support gradient: "
                    + ",".join(missing_support_gradients)
                )
            if any(
                not torch.isfinite(gradient).all()
                for gradient in support_gradients
                if gradient is not None
            ):
                raise SparseEncoderAdaptationError(
                    "support adaptation produced non-finite support gradients"
                )
            loss.backward()
            missing_gradients = [
                name for name, parameter in selected if parameter.grad is None
            ]
            if missing_gradients:
                raise SparseEncoderAdaptationError(
                    "selected parameters received no gradient: "
                    + ",".join(missing_gradients)
                )
            gradients = [parameter.grad for _, parameter in selected]
            if any(not torch.isfinite(grad).all() for grad in gradients):
                raise SparseEncoderAdaptationError(
                    "support adaptation produced non-finite gradients"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in selected],
                max_norm=float(config.gradient_clip),
            )
            optimizer.step()
            completed_updates += 1
            trace.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "classification_loss": float(
                        classification_loss.detach().cpu().item()
                    ),
                    "feature_anchor_loss": float(feature_anchor.detach().cpu().item()),
                    "parameter_anchor_loss": float(
                        parameter_anchor.detach().cpu().item()
                    ),
                    "gradient_norm": float(
                        torch.as_tensor(gradient_norm).detach().cpu().item()
                    ),
                }
            )
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

    selected_names = {name for name, _ in selected}
    unexpected_changes = [
        name
        for name, parameter in model.named_parameters()
        if name not in selected_names
        and not torch.equal(parameter.detach(), parameter_before[name])
    ]
    if unexpected_changes:
        raise SparseEncoderAdaptationError(
            f"non-allowlisted parameters changed: {unexpected_changes}"
        )
    classifier_changes = sum(
        not torch.equal(dict(model.named_parameters())[name].detach(), parameter_before[name])
        for name in classifier_names
    )
    if classifier_changes:
        raise SparseEncoderAdaptationError("frozen classifier parameters changed")

    return SparseEncoderAdaptationAudit(
        method_id="SOFESA_V1",
        candidate=str(config.candidate),
        gradient_updates=completed_updates,
        support_samples=int(rows.shape[0]),
        support_class_count=int(torch.unique(labels).numel()),
        trainable_parameters=int(sum(parameter.numel() for _, parameter in selected)),
        total_parameters=total_parameters,
        trainable_fraction=trainable_fraction,
        trainable_parameter_names=tuple(name for name, _ in selected),
        classifier_parameters_changed=int(classifier_changes),
        loss_trace=tuple(trace),
    )


def predict_query_read_only(
    model: nn.Module, received_iq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the unchanged checkpoint decision rule without updating state."""

    if model.training:
        raise SparseEncoderAdaptationError("read-only query requires model.eval()")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SparseEncoderAdaptationError(
            "read-only query requires every model parameter to be frozen"
        )
    if not torch.is_tensor(received_iq) or received_iq.ndim != 3:
        raise SparseEncoderAdaptationError("received_iq must be a [N,2,L] tensor")
    device = _model_device(model)
    rows = received_iq.detach().to(device=device, dtype=torch.float32)
    # Grad mode stays enabled deliberately: ADV3B02's eval/no-grad fast paths
    # populate internal caches.  With every parameter and input frozen this
    # builds no graph, while avoiding any query-time cache mutation.
    with torch.enable_grad():
        _features, logits = _feature_and_logits(model, rows)
    scores = logits.detach().clone()
    return scores.argmax(dim=1), scores


__all__ = [
    "MAX_ADAPTATION_STEPS",
    "MAX_TRAINABLE_FRACTION",
    "EnrollmentSupport",
    "SparseEncoderAdaptationAudit",
    "SparseEncoderAdaptationConfig",
    "SparseEncoderAdaptationError",
    "adapt_on_target_support",
    "load_validated_enrollment_support",
    "predict_query_read_only",
]
