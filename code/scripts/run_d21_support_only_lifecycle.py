#!/usr/bin/env python3
"""Run D21 prototype lifecycle on sealed LEO_weak enrollment support only.

The CLI intentionally has no query, truth, scorer, role, quota, or batch
assignment input.  Real execution delegates target package materialization and
the NumPy/Torch DLPack bridge to the existing D20/D19 runner, while Phase1
knowledge is accepted only through the jointly signed ADV3B02 runtime+v2
component deployment bundle.  A legacy target capsule lacking the outer-root,
seal, envelope, lineage, or runtime bindings is rejected for rebuild.  This
script never invents or substitutes a radius row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (CODE, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cvsrffi.phase1_adv3b02_deployment_bundle import (  # noqa: E402
    FORMAL_CONTEXT_SCHEMA,
    VerifiedADV3B02DeploymentBundle,
    load_formal_adv3b02_deployment_bundle,
)
from cvsrffi.phase1_center_lowrank_prototype_bundle import (  # noqa: E402
    CenterLowRankPrototypeComponent,
)
from cvsrffi.somph_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    finalize_somph_enrollment_authority_after_materialization,
    materialize_somph_enrollment_with_signed_authority,
)
from cvsrffi.stage2_prototype_lifecycle import (  # noqa: E402
    FEATURE_DIM,
    LifecycleConfig,
    PrototypeLifecycleState,
    fit_old_snapshot,
    register_new_classes,
    score_batch,
)
from run_d19_support_only_ciaf import (  # noqa: E402
    _numpy2_torch21_as_tensor_compatibility,
    _old_reuse,
    _overlay_index,
    _preopen_manifest,
    _require_post_materialization_authority,
    _tensor_from_numpy_dlpack,
)


MODE = "support_only_local_no_performance_claim"
SUPPORTED_K = (1, 5, 10, 20)
L0 = "L0_PURE_CENTROID"
L1 = "L1_RADIUS"
L2 = "L2_RADIUS_BOUNDARY"
CANDIDATE_ORDER = (L0, L1, L2)
_RECEIPT_TOKEN_SECRET = object()
TARGET_JOINT_BINDING_FIELDS = {
    "phase1_adv3b02_outer_content_root_sha256",
    "phase1_adv3b02_detached_seal_sha256",
    "phase1_adv3b02_signature_envelope_sha256",
    "phase1_checkpoint_lineage_sha256",
    "feature_runtime_sha256",
}


class D21RunnerError(ValueError):
    """Fail-closed D21 support-only runner error."""


@dataclass(frozen=True)
class _VerifiedSupportReceiptToken:
    before_capsule_root_sha256: str
    before_receipt_sha256: str
    after_capsule_root_sha256: str
    after_receipt_sha256: str
    sealed: bool
    _secret: object

    def __post_init__(self) -> None:
        values = (
            self.before_capsule_root_sha256,
            self.before_receipt_sha256,
            self.after_capsule_root_sha256,
            self.after_receipt_sha256,
        )
        if self._secret is not _RECEIPT_TOKEN_SECRET or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in values
        ):
            raise D21RunnerError("verified support receipt token drift")


def _synthetic_receipt_token(envelope_sha256: str) -> _VerifiedSupportReceiptToken:
    return _VerifiedSupportReceiptToken(
        before_capsule_root_sha256=hashlib.sha256(
            f"d21-synthetic-before-root:{envelope_sha256}".encode("ascii")
        ).hexdigest(),
        before_receipt_sha256=hashlib.sha256(
            f"d21-synthetic-before-receipt:{envelope_sha256}".encode("ascii")
        ).hexdigest(),
        after_capsule_root_sha256=hashlib.sha256(
            f"d21-synthetic-after-root:{envelope_sha256}".encode("ascii")
        ).hexdigest(),
        after_receipt_sha256=hashlib.sha256(
            f"d21-synthetic-after-receipt:{envelope_sha256}".encode("ascii")
        ).hexdigest(),
        sealed=False,
        _secret=_RECEIPT_TOKEN_SECRET,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise D21RunnerError(f"refusing to overwrite output: {path}")
    raw = _json_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def preregistered_candidates(radius_prior: float) -> dict[str, LifecycleConfig]:
    """Return the fixed L0/L1/L2 mechanism ladder."""

    if not math.isfinite(radius_prior) or not 0.0 <= radius_prior <= 2.0:
        raise D21RunnerError("component-derived radius prior must be finite in [0,2]")
    common = {"radius_prior": float(radius_prior)}
    return {
        L0: LifecycleConfig(radius_enabled=False, boundary_enabled=False, **common),
        L1: LifecycleConfig(radius_enabled=True, boundary_enabled=False, **common),
        L2: LifecycleConfig(radius_enabled=True, boundary_enabled=True, **common),
    }


def component_global_median_radius_prior(
    component: CenterLowRankPrototypeComponent,
    *,
    expected_old_class_count: int,
    formal_context: Mapping[str, Any] | None = None,
    require_joint_formal: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Dequantize every immutable radius row and take one deterministic median."""

    manifest = component.manifest
    if require_joint_formal and (
        not isinstance(formal_context, Mapping)
        or formal_context.get("schema") != FORMAL_CONTEXT_SCHEMA
        or formal_context.get("formal_phase2_eligible") is not True
        or formal_context.get("standalone_component_formal_phase2_eligible") is not False
        or formal_context.get("outer_signature_verified") is not True
        or formal_context.get("detached_seal_verified") is not True
        or formal_context.get("runtime_checkpoint_parity_verified") is not True
    ):
        raise D21RunnerError("joint formal Phase1 context drift")
    if (
        "missing" in str(manifest.get("radius_provenance", ""))
        or len(component.class_registry) != int(expected_old_class_count)
        or not component.domain_registry
    ):
        raise D21RunnerError("formal v2 component/radius/old-class binding drift")
    rows = np.stack(
        [component.radius_for_domain(handle) for handle in component.domain_registry]
    ).astype(np.float32)
    if (
        rows.shape != (len(component.domain_registry), expected_old_class_count)
        or not np.isfinite(rows).all()
        or bool(np.any(rows < 0.0))
        or bool(np.any(rows > 2.0))
    ):
        raise D21RunnerError("v2 component radius rows are invalid")
    prior = float(np.median(rows.reshape(-1)))
    if not math.isfinite(prior):
        raise D21RunnerError("v2 component radius median is non-finite")
    audit = {
        "schema": "cvs.phase2.d21.component_radius_binding.v1",
        "component_schema": str(manifest.get("schema", "")),
        "checkpoint_sha256": str(manifest.get("checkpoint_sha256", "")),
        "class_handle_binding_sha256": str(
            manifest.get("class_handle_binding_sha256", "")
        ),
        "deployment_bundle_root_sha256": str(
            manifest.get("deployment_bundle_root_sha256", "")
        ),
        "radius_provenance": str(manifest.get("radius_provenance", "")),
        "radius_definition": str(manifest.get("radius_definition", "")),
        "domain_count": len(component.domain_registry),
        "class_count": len(component.class_registry),
        "radius_value_count": int(rows.size),
        "global_median_radius_prior": prior,
        "median_rule": "numpy_median_over_all_dequantized_domain_class_radius_rows",
        "component_update_access": False,
        "synthetic_or_substitute_radius_used": False,
        "joint_formal_context_required": require_joint_formal,
        "joint_formal_context_verified": bool(require_joint_formal),
        "standalone_component_formal_claimed": False,
    }
    return prior, audit


def _support_metrics(
    state: PrototypeLifecycleState,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
) -> dict[str, Any]:
    labels = np.asarray([str(value) for value in support_labels])
    scores = np.asarray(score_batch(state, support_z_id), dtype=np.float32)
    index = {value: offset for offset, value in enumerate(state.classes)}
    try:
        targets = np.asarray([index[value] for value in labels], dtype=np.int64)
    except KeyError as exc:
        raise D21RunnerError("support label absent from lifecycle registry") from exc
    prediction = np.argmax(scores, axis=1)
    rival = np.array(scores, copy=True)
    rival[np.arange(len(scores)), targets] = -np.inf
    margin = scores[np.arange(len(scores)), targets] - np.max(rival, axis=1)
    by_class: dict[str, Any] = {}
    for class_index, class_handle in enumerate(state.classes):
        mask = targets == class_index
        if not np.any(mask):
            raise D21RunnerError("support result lacks one registered class")
        by_class[class_handle] = {
            "support_rows": int(np.sum(mask)),
            "accuracy": float(np.mean(prediction[mask] == class_index)),
            "mean_true_margin": float(np.mean(margin[mask])),
            "worst_true_margin": float(np.min(margin[mask])),
        }
    return {
        "support_rows": len(labels),
        "overall_accuracy": float(np.mean(prediction == targets)),
        "minimum_class_accuracy": min(row["accuracy"] for row in by_class.values()),
        "worst_true_margin": float(np.min(margin)),
        "per_class": by_class,
    }


def _state_record(state: PrototypeLifecycleState) -> dict[str, Any]:
    return {
        "schema": state.schema,
        "stage": state.stage,
        "classes": state.classes,
        "old_class_count": state.old_class_count,
        "k_shot": state.k_shot,
        "old_support_capsule_root_sha256": state.old_support_capsule_root_sha256,
        "old_support_content_sha256": state.old_support_content_sha256,
        "old_support_receipt_sha256": state.old_support_receipt_sha256,
        "current_support_capsule_root_sha256": (
            state.current_support_capsule_root_sha256
        ),
        "current_support_receipt_sha256": state.current_support_receipt_sha256,
        "prototypes": state.prototypes,
        "radii": state.radii,
        "radius_active": state.radius_active,
        "support_count_by_class": state.support_count_by_class,
        "old_prototype_snapshot_sha256": hashlib.sha256(
            state.old_prototype_snapshot.tobytes()
        ).hexdigest(),
        "old_radius_snapshot_sha256": hashlib.sha256(
            state.old_radius_snapshot.tobytes()
        ).hexdigest(),
        "center_policy": state.center_policy,
        "radius_policy": state.radius_policy,
        "boundaries": [
            {
                "new_class_index": item.new_class_index,
                "rival_class_index": item.rival_class_index,
                "feature_indices": item.feature_indices,
                "direction_values": item.direction_values,
                "midpoint_projection": item.midpoint_projection,
                "safe_threshold": item.safe_threshold,
            }
            for item in state.boundaries
        ],
    }


def _internal_target_score_lock(
    before: PrototypeLifecycleState,
    after: PrototypeLifecycleState,
    probes: np.ndarray,
) -> dict[str, Any]:
    locked = (
        np.array_equal(after.prototypes[: before.old_class_count], before.prototypes)
        and np.array_equal(after.radii[: before.old_class_count], before.radii)
        and np.array_equal(
            after.radius_active[: before.old_class_count], before.radius_active
        )
        and np.array_equal(after.old_prototype_snapshot, before.prototypes)
        and np.array_equal(after.old_radius_snapshot, before.radii)
        and np.array_equal(
            np.asarray(score_batch(after, probes))[:, : before.old_class_count],
            np.asarray(score_batch(before, probes)),
        )
    )
    if not locked:
        raise D21RunnerError("append-only old prototype/radius/score lock drift")
    return {
        "lock_kind": "internal-target-score-lock",
        "dali_integrated": False,
        "dali_lock_claimed": False,
        "old_prototype_bitwise_locked": True,
        "old_radius_bitwise_locked": True,
        "old_radius_active_mask_bitwise_locked": True,
        "old_score_columns_bitwise_locked": True,
        "probe_rows": int(len(probes)),
    }


def _select_candidate(candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = candidates[L0]["after_support_result"]
    baseline_acc = np.asarray(
        [row["accuracy"] for row in baseline["per_class"].values()], dtype=np.float64
    )
    baseline_margin = float(baseline["worst_true_margin"])
    eligible = []
    decisions = {}
    for complexity, candidate_id in enumerate(CANDIDATE_ORDER):
        result = candidates[candidate_id]["after_support_result"]
        accuracy = np.asarray(
            [row["accuracy"] for row in result["per_class"].values()], dtype=np.float64
        )
        margin = float(result["worst_true_margin"])
        noninferior = bool(
            np.all(accuracy + 1.0e-7 >= baseline_acc)
            and margin + 1.0e-7 >= baseline_margin
        )
        key = (
            float(np.min(accuracy)),
            float(result["overall_accuracy"]),
            margin,
            -complexity,
        )
        decisions[candidate_id] = {
            "noninferior_to_l0": noninferior,
            "ranking_key": key,
        }
        if noninferior:
            eligible.append((key, candidate_id))
    selected = max(eligible)[1] if eligible else L0
    return {
        "selected_candidate_id": selected,
        "fallback_to_l0": selected == L0,
        "selection_rule": (
            "all_class_support_accuracy_and_worst_true_margin_noninferior_to_L0_"
            "then_lexicographic_accuracy_margin_with_lower_complexity_tie_break"
        ),
        "candidate_decisions": decisions,
    }


def _evaluate_support_lifecycle(
    *,
    old_support_z_id: np.ndarray,
    old_support_labels: Sequence[str],
    new_support_z_id: np.ndarray,
    new_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    component: CenterLowRankPrototypeComponent,
    receipt_token: _VerifiedSupportReceiptToken,
    formal_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Internal evaluation requiring a module-issued receipt token."""

    if not isinstance(receipt_token, _VerifiedSupportReceiptToken):
        raise D21RunnerError("verified support receipt token required")

    old_classes = tuple(str(value) for value in old_classes)
    new_classes = tuple(str(value) for value in new_classes)
    radius_prior, component_audit = component_global_median_radius_prior(
        component,
        expected_old_class_count=len(old_classes),
        formal_context=formal_context,
        require_joint_formal=receipt_token.sealed,
    )
    candidate_results: dict[str, Any] = {}
    all_support = np.concatenate([old_support_z_id, new_support_z_id], axis=0)
    all_labels = tuple(old_support_labels) + tuple(new_support_labels)
    for candidate_id, config in preregistered_candidates(radius_prior).items():
        before = fit_old_snapshot(
            old_support_z_id,
            old_support_labels,
            old_classes,
            old_support_capsule_root_sha256=(
                receipt_token.before_capsule_root_sha256
            ),
            old_support_receipt_sha256=receipt_token.before_receipt_sha256,
            config=config,
        )
        after = register_new_classes(
            before,
            old_support_z_id,
            old_support_labels,
            new_support_z_id,
            new_support_labels,
            new_classes,
            old_support_capsule_root_sha256=(
                receipt_token.before_capsule_root_sha256
            ),
            old_support_receipt_sha256=receipt_token.before_receipt_sha256,
            after_registration_capsule_root_sha256=(
                receipt_token.after_capsule_root_sha256
            ),
            after_registration_receipt_sha256=receipt_token.after_receipt_sha256,
        )
        if before.k_shot not in SUPPORTED_K or after.k_shot != before.k_shot:
            raise D21RunnerError("D21 K-shot lifecycle drift")
        candidate_results[candidate_id] = {
            "config": {
                "radius_enabled": config.radius_enabled,
                "boundary_enabled": config.boundary_enabled,
                "radius_prior": config.radius_prior,
            },
            "before_state": _state_record(before),
            "after_state": _state_record(after),
            "before_resource": before.resource_audit(),
            "after_resource": after.resource_audit(),
            "before_support_result": _support_metrics(
                before, old_support_z_id, old_support_labels
            ),
            "after_support_result": _support_metrics(after, all_support, all_labels),
            "support_guard": after.support_audit.get("boundary_support_guard"),
            "internal_target_score_lock": _internal_target_score_lock(
                before, after, all_support
            ),
            "candidate_fallback": {
                "radius_off": not config.radius_enabled,
                "boundary_off": not config.boundary_enabled,
                "accepted_sparse_boundary_count": len(after.boundaries),
                "boundary_guard_fell_back": bool(
                    config.boundary_enabled
                    and after.support_audit["boundary_support_guard"][
                        "accepted_boundary_count"
                    ]
                    == 0
                ),
            },
        }
    selection = _select_candidate(candidate_results)
    return {
        "schema": "cvs.phase2.d21.support_only_lifecycle_result.v1",
        "status": (
            "SEALED_SUPPORT_ONLY_COMPLETE"
            if receipt_token.sealed
            else "SYNTHETIC_LOCAL_COMPLETE"
        ),
        "evidence_mode": (
            "sealed_enrollment_only" if receipt_token.sealed else "synthetic_local"
        ),
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "query_rows_opened": 0,
        "scorer_opened": False,
        "k_shot": candidate_results[L0]["before_state"]["k_shot"],
        "old_classes": old_classes,
        "new_classes": new_classes,
        "component_binding": component_audit,
        "old_support_capsule_root_sha256": (
            receipt_token.before_capsule_root_sha256
        ),
        "old_support_receipt_sha256": receipt_token.before_receipt_sha256,
        "after_registration_capsule_root_sha256": (
            receipt_token.after_capsule_root_sha256
        ),
        "after_registration_receipt_sha256": receipt_token.after_receipt_sha256,
        "candidate_order": CANDIDATE_ORDER,
        "candidates": candidate_results,
        "selection": selection,
    }


def evaluate_support_lifecycle(
    *,
    old_support_z_id: np.ndarray,
    old_support_labels: Sequence[str],
    new_support_z_id: np.ndarray,
    new_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    component: CenterLowRankPrototypeComponent,
) -> dict[str, Any]:
    """Public synthetic/local evaluator; it cannot mint sealed evidence."""

    envelope = hashlib.sha256(
        np.ascontiguousarray(
            np.concatenate([old_support_z_id, new_support_z_id], axis=0),
            dtype=np.float32,
        ).tobytes()
    ).hexdigest()
    token = _synthetic_receipt_token(envelope)
    return _evaluate_support_lifecycle(
        old_support_z_id=old_support_z_id,
        old_support_labels=old_support_labels,
        new_support_z_id=new_support_z_id,
        new_support_labels=new_support_labels,
        old_classes=old_classes,
        new_classes=new_classes,
        component=component,
        receipt_token=token,
        formal_context=None,
    )


def _manifest_binding(before: Mapping[str, Any], after: Mapping[str, Any]) -> int:
    k_shot = int(before.get("k_shot", -1))
    if (
        k_shot not in SUPPORTED_K
        or int(after.get("k_shot", -2)) != k_shot
        or before.get("receiver") != after.get("receiver")
        or int(before.get("seed", -1)) != int(after.get("seed", -2))
        or before.get("feature_runtime_sha256")
        != after.get("feature_runtime_sha256")
        or before.get("phase1_checkpoint_lineage_sha256")
        != after.get("phase1_checkpoint_lineage_sha256")
    ):
        raise D21RunnerError("before/after enrollment binding drift")
    return k_shot


def _require_target_joint_binding(
    surface: Mapping[str, Any],
    *,
    surface_name: str,
    expected_outer_content_root_sha256: str,
    expected_detached_seal_sha256: str,
    expected_signature_envelope_sha256: str,
    expected_checkpoint_lineage_sha256: str,
    expected_runtime_sha256: str,
) -> None:
    expected = {
        "phase1_adv3b02_outer_content_root_sha256": expected_outer_content_root_sha256,
        "phase1_adv3b02_detached_seal_sha256": expected_detached_seal_sha256,
        "phase1_adv3b02_signature_envelope_sha256": (
            expected_signature_envelope_sha256
        ),
        "phase1_checkpoint_lineage_sha256": expected_checkpoint_lineage_sha256,
        "feature_runtime_sha256": expected_runtime_sha256,
    }
    missing = TARGET_JOINT_BINDING_FIELDS - set(surface)
    if missing:
        raise D21RunnerError(
            f"{surface_name} lacks joint Phase1 binding fields; rebuild target capsule"
        )
    if any(str(surface.get(key, "")).lower() != value.lower() for key, value in expected.items()):
        raise D21RunnerError(f"{surface_name} joint Phase1 binding drift")


def _registered_handles(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list) or not rows:
        raise D21RunnerError("registered class manifest missing")
    ordered = sorted(rows, key=lambda row: int(row.get("class_index", -1)))
    indices = [int(row.get("class_index", -1)) for row in ordered]
    handles = tuple(str(row.get("class_handle", "")) for row in ordered)
    if indices != list(range(len(indices))) or len(set(handles)) != len(handles):
        raise D21RunnerError("registered class order drift")
    return handles


def _payload_rows(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    overlay: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    classes = _registered_handles(manifest)
    k_shot = int(manifest["k_shot"])
    indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    labels = np.asarray(classes)[indices].astype(str)
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    order = np.asarray(
        sorted(range(len(labels)), key=lambda i: (str(labels[i]), int(ranks[i]))),
        dtype=np.int64,
    )
    rows = {
        "iq": np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[order],
        "labels": labels[order],
        "ranks": ranks[order],
        "tokens": np.asarray(payload["support_tokens"]).astype(str)[order],
        "hashes": np.asarray(payload["support_post_channel_iq_sha256"])
        .astype(str)[order],
        "overlay_tokens": np.asarray(payload["support_overlay_tokens"])
        .astype(str)[order],
        "satellite_seeds": np.asarray(
            payload["support_satellite_seeds"], dtype=np.int64
        )[order],
    }
    unique, counts = np.unique(rows["labels"], return_counts=True)
    computed = np.asarray(
        [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows["iq"]]
    )
    if (
        rows["iq"].ndim != 3
        or rows["iq"].shape[1] != 2
        or not np.isfinite(rows["iq"]).all()
        or set(unique.tolist()) != set(classes)
        or set(counts.tolist()) != {k_shot}
        or not np.array_equal(computed, rows["hashes"])
    ):
        raise D21RunnerError(f"strict enrollment payload drift: {scenario}")
    for index, (token, parent) in enumerate(zip(rows["tokens"], rows["hashes"])):
        item = overlay.get((str(token), str(parent), scenario))
        if (
            item is None
            or str(item.get("overlay_token")) != str(rows["overlay_tokens"][index])
            or int(item.get("satellite_seed", -1))
            != int(rows["satellite_seeds"][index])
        ):
            raise D21RunnerError("support payload/overlay binding drift")
    return rows


def _extract_z_id(
    model: torch.nn.Module, device: torch.device, iq_rows: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    values = []
    hashes = []
    model.eval()
    with _numpy2_torch21_as_tensor_compatibility():
        for iq in np.asarray(iq_rows, dtype=np.float32):
            batch = _tensor_from_numpy_dlpack(
                np.ascontiguousarray(iq[None, ...]),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                output = model(batch)
            feature = output.get("features") if isinstance(output, dict) else output[0]
            if not torch.is_tensor(feature) or tuple(feature.shape) != (1, FEATURE_DIM):
                raise D21RunnerError("sealed runtime z_id output drift")
            row = np.asarray(feature.detach().float().cpu().tolist()[0], dtype=np.float32)
            if not np.isfinite(row).all():
                raise D21RunnerError("sealed runtime z_id is non-finite")
            values.append(row)
            hashes.append(hashlib.sha256(row.tobytes()).hexdigest())
    return np.stack(values), {
        "support_rows": len(values),
        "backbone_forwards": len(values),
        "one_forward_per_physical_support": True,
        "derived_views_per_support": 0,
        "dlpack_numpy_torch_bridge": True,
        "feature_sha256": hashes,
    }


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    before_formal_policy: Path,
    before_formal_policy_authorization: Path,
    before_signed_policy_authorization_envelope: Path,
    expected_before_signed_policy_authorization_envelope_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    after_formal_policy: Path,
    after_formal_policy_authorization: Path,
    after_signed_policy_authorization_envelope: Path,
    expected_after_signed_policy_authorization_envelope_sha256: str,
    joint_package_root: Path,
    joint_detached_seal: Path,
    expected_joint_detached_seal_sha256: str,
    joint_signature_envelope: Path,
    expected_joint_signature_envelope_sha256: str,
    expected_checkpoint_lineage_sha256: str,
    expected_runtime_sha256: str,
    expected_component_pre_sign_content_root_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_parity_receipt_sha256: str,
    expected_generation_lock_sha256: str,
    expected_method_lock_sha256: str,
    expected_generation_config_sha256: str,
    expected_generation_code_sha256: str,
    expected_outer_content_root_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = MODE,
) -> dict[str, Any]:
    if mode != MODE or output.exists():
        raise D21RunnerError("D21 requires a new local support-only output")
    before_preopen = _preopen_manifest(
        before_root, before_seal, expected_seal_sha256=expected_before_seal_sha256
    )
    after_preopen = _preopen_manifest(
        after_root, after_seal, expected_seal_sha256=expected_after_seal_sha256
    )
    _manifest_binding(before_preopen, after_preopen)
    for name, manifest in (("before target manifest", before_preopen), ("after target manifest", after_preopen)):
        _require_target_joint_binding(
            manifest,
            surface_name=name,
            expected_outer_content_root_sha256=expected_outer_content_root_sha256,
            expected_detached_seal_sha256=expected_joint_detached_seal_sha256,
            expected_signature_envelope_sha256=expected_joint_signature_envelope_sha256,
            expected_checkpoint_lineage_sha256=expected_checkpoint_lineage_sha256,
            expected_runtime_sha256=expected_runtime_sha256,
        )
    old_classes = _registered_handles(before_preopen)
    joint_bundle = load_formal_adv3b02_deployment_bundle(
        joint_package_root,
        detached_seal_path=joint_detached_seal,
        expected_detached_seal_sha256=expected_joint_detached_seal_sha256,
        signature_envelope_path=joint_signature_envelope,
        expected_signature_envelope_sha256=(
            expected_joint_signature_envelope_sha256
        ),
        expected_checkpoint_lineage_sha256=expected_checkpoint_lineage_sha256,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_component_pre_sign_content_root_sha256=(
            expected_component_pre_sign_content_root_sha256
        ),
        expected_class_handle_binding_sha256=expected_class_handle_binding_sha256,
        expected_parity_receipt_sha256=expected_parity_receipt_sha256,
        expected_generation_lock_sha256=expected_generation_lock_sha256,
        expected_method_lock_sha256=expected_method_lock_sha256,
        expected_generation_config_sha256=expected_generation_config_sha256,
        expected_generation_code_sha256=expected_generation_code_sha256,
        expected_outer_content_root_sha256=expected_outer_content_root_sha256,
    )
    if not isinstance(joint_bundle, VerifiedADV3B02DeploymentBundle):
        raise D21RunnerError("formal joint deployment bundle loader result drift")
    component = joint_bundle.component
    if tuple(component.class_registry) != old_classes:
        raise D21RunnerError("joint component/target old-class ordered binding drift")
    component_global_median_radius_prior(
        component,
        expected_old_class_count=len(old_classes),
        formal_context=joint_bundle.formal_phase2_context,
        require_joint_formal=True,
    )
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = joint_bundle.runtime.to(device)
    model.eval()
    materialize_kwargs = (
        (
            before_root,
            before_seal,
            expected_before_seal_sha256,
            before_formal_policy,
            before_formal_policy_authorization,
            before_signed_policy_authorization_envelope,
            expected_before_signed_policy_authorization_envelope_sha256,
        ),
        (
            after_root,
            after_seal,
            expected_after_seal_sha256,
            after_formal_policy,
            after_formal_policy_authorization,
            after_signed_policy_authorization_envelope,
            expected_after_signed_policy_authorization_envelope_sha256,
        ),
    )
    evidences = []
    for root, seal, seal_sha, policy, authorization, envelope, envelope_sha in materialize_kwargs:
        evidences.append(
            materialize_somph_enrollment_with_signed_authority(
                root,
                detached_seal_path=seal,
                expected_seal_sha256=seal_sha,
                formal_policy_path=policy,
                formal_policy_authorization_path=authorization,
                signed_policy_authorization_envelope_path=envelope,
                expected_signed_policy_authorization_envelope_sha256=envelope_sha,
            )
        )
    before_evidence, after_evidence = evidences
    before_authority = finalize_somph_enrollment_authority_after_materialization(
        before_evidence
    )
    after_authority = finalize_somph_enrollment_authority_after_materialization(
        after_evidence
    )
    _require_post_materialization_authority(before_authority, after_authority)
    for name, surface in (
        ("before materialized manifest", before_evidence.manifest),
        ("after materialized manifest", after_evidence.manifest),
        ("before materialization authority", before_authority),
        ("after materialization authority", after_authority),
    ):
        _require_target_joint_binding(
            surface,
            surface_name=name,
            expected_outer_content_root_sha256=expected_outer_content_root_sha256,
            expected_detached_seal_sha256=expected_joint_detached_seal_sha256,
            expected_signature_envelope_sha256=(
                expected_joint_signature_envelope_sha256
            ),
            expected_checkpoint_lineage_sha256=expected_checkpoint_lineage_sha256,
            expected_runtime_sha256=expected_runtime_sha256,
        )
    before_capsule_root = str(before_authority.get("package_root_sha256", ""))
    after_capsule_root = str(after_authority.get("package_root_sha256", ""))
    before_receipt_sha256 = str(
        before_authority.get("post_materialization_audit_sha256", "")
    )
    after_receipt_sha256 = str(
        after_authority.get("post_materialization_audit_sha256", "")
    )
    if (
        len(before_capsule_root) != 64
        or len(after_capsule_root) != 64
        or len(before_receipt_sha256) != 64
        or len(after_receipt_sha256) != 64
        or before_capsule_root != before_evidence.manifest.get("package_root_sha256")
        or after_capsule_root != after_evidence.manifest.get("package_root_sha256")
    ):
        raise D21RunnerError("sealed enrollment authority capsule root drift")
    receipt_token = _VerifiedSupportReceiptToken(
        before_capsule_root_sha256=before_capsule_root,
        before_receipt_sha256=before_receipt_sha256,
        after_capsule_root_sha256=after_capsule_root,
        after_receipt_sha256=after_receipt_sha256,
        sealed=True,
        _secret=_RECEIPT_TOKEN_SECRET,
    )
    k_shot = _manifest_binding(before_evidence.manifest, after_evidence.manifest)
    all_classes = _registered_handles(after_evidence.manifest)
    if all_classes[: len(old_classes)] != old_classes or len(all_classes) == len(old_classes):
        raise D21RunnerError("after registry must append real new classes")
    new_classes = all_classes[len(old_classes) :]
    before_overlay, _ = _overlay_index(before_root, before_evidence.manifest)
    after_overlay, _ = _overlay_index(after_root, after_evidence.manifest)
    by_scenario = {}
    extraction = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _payload_rows(
            before_evidence.materialized_payloads[scenario],
            before_evidence.manifest,
            before_overlay,
            scenario=scenario,
        )
        after_rows = _payload_rows(
            after_evidence.materialized_payloads[scenario],
            after_evidence.manifest,
            after_overlay,
            scenario=scenario,
        )
        _old_reuse(before_rows, after_rows)
        is_new = np.isin(after_rows["labels"], np.asarray(new_classes))
        old_z, old_extract = _extract_z_id(model, device, before_rows["iq"])
        new_z, new_extract = _extract_z_id(model, device, after_rows["iq"][is_new])
        by_scenario[scenario] = _evaluate_support_lifecycle(
            old_support_z_id=old_z,
            old_support_labels=before_rows["labels"],
            new_support_z_id=new_z,
            new_support_labels=after_rows["labels"][is_new],
            old_classes=old_classes,
            new_classes=new_classes,
            component=component,
            receipt_token=receipt_token,
            formal_context=joint_bundle.formal_phase2_context,
        )
        extraction[scenario] = {"before_old": old_extract, "after_new": new_extract}
    output.mkdir(parents=True, exist_ok=False)
    result_sha = _write_json_new(
        output / "lifecycle_results.json",
        {
            "schema": "cvs.phase2.d21.support_only_lifecycle_matrix.v1",
            "k_shot": k_shot,
            "by_scenario": by_scenario,
        },
    )
    support_sha = _write_json_new(
        output / "support_audit.json",
        {
            "schema": "cvs.phase2.d21.support_only_audit.v1",
            "query_opened": False,
            "scorer_opened": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "before_old_support_only": True,
            "after_new_support_only": True,
            "before_old_support_capsule_root_sha256": before_capsule_root,
            "before_old_support_receipt_sha256": before_receipt_sha256,
            "after_registration_support_capsule_root_sha256": after_capsule_root,
            "after_registration_support_receipt_sha256": after_receipt_sha256,
            "dali_integrated": False,
            "score_lock_claim": "internal-target-score-lock-only",
            "component_radius_substitution": False,
            "joint_phase1_formal_context": joint_bundle.formal_phase2_context,
            "joint_phase1_load_audit": joint_bundle.audit,
            "feature_extraction": extraction,
        },
    )
    resource_payload = {
        "schema": "cvs.phase2.d21.resource_matrix.v1",
        "component": component.resource_audit(),
        "joint_phase1_load_audit": joint_bundle.audit,
        "by_scenario": {
            scene: {
                candidate: row["after_resource"]
                for candidate, row in value["candidates"].items()
            }
            for scene, value in by_scenario.items()
        },
        "actual_artifact_bytes": {},
        "total_delivery_footprint_bytes": 0,
        "footprint_includes_commit": True,
    }
    commit = {
        "schema": "cvs.phase2.d21.support_only_commit.v1",
        "status": "SEALED_SUPPORT_ONLY_COMPLETE_NO_PERFORMANCE_CLAIM",
        "mode": mode,
        "k_shot": k_shot,
        "query_opened": False,
        "scorer_opened": False,
        "phase1_joint_outer_content_root_sha256": expected_outer_content_root_sha256,
        "phase1_joint_detached_seal_sha256": expected_joint_detached_seal_sha256,
        "phase1_joint_signature_envelope_sha256": (
            expected_joint_signature_envelope_sha256
        ),
        "artifacts": {
            "lifecycle_results.json": result_sha,
            "support_audit.json": support_sha,
            "resource_audit.json": "0" * 64,
        },
        "actual_artifact_bytes": {},
        "total_delivery_footprint_bytes": 0,
    }
    artifact_bytes = {
        "lifecycle_results.json": (output / "lifecycle_results.json").stat().st_size,
        "support_audit.json": (output / "support_audit.json").stat().st_size,
        "resource_audit.json": 0,
        "COMMIT.json": 0,
    }
    for _ in range(12):
        total = sum(artifact_bytes.values())
        resource_payload["actual_artifact_bytes"] = dict(artifact_bytes)
        resource_payload["total_delivery_footprint_bytes"] = total
        resource_size = len(_json_bytes(resource_payload))
        commit["actual_artifact_bytes"] = dict(artifact_bytes)
        commit["total_delivery_footprint_bytes"] = total
        commit_size = len(_json_bytes(commit))
        updated = dict(artifact_bytes)
        updated["resource_audit.json"] = resource_size
        updated["COMMIT.json"] = commit_size
        if updated == artifact_bytes:
            break
        artifact_bytes = updated
    else:
        raise D21RunnerError("artifact footprint fixed point did not converge")
    total = sum(artifact_bytes.values())
    resource_payload["actual_artifact_bytes"] = dict(artifact_bytes)
    resource_payload["total_delivery_footprint_bytes"] = total
    resource_sha = _write_json_new(output / "resource_audit.json", resource_payload)
    commit["artifacts"]["resource_audit.json"] = resource_sha
    commit["actual_artifact_bytes"] = dict(artifact_bytes)
    commit["total_delivery_footprint_bytes"] = total
    commit_sha = _write_json_new(output / "COMMIT.json", commit)
    actual = {name: (output / name).stat().st_size for name in artifact_bytes}
    if actual != artifact_bytes or sum(actual.values()) != total:
        raise D21RunnerError("actual artifact delivery footprint drift")
    return {**commit, "commit_sha256": commit_sha, "output": str(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for state in ("before", "after"):
        parser.add_argument(f"--{state}-root", type=Path, required=True)
        parser.add_argument(f"--{state}-seal", type=Path, required=True)
        parser.add_argument(f"--{state}-seal-sha256", required=True)
        parser.add_argument(f"--{state}-formal-policy", type=Path, required=True)
        parser.add_argument(
            f"--{state}-formal-policy-authorization", type=Path, required=True
        )
        parser.add_argument(
            f"--{state}-signed-policy-authorization-envelope",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{state}-signed-policy-authorization-envelope-sha256", required=True
        )
    parser.add_argument("--joint-package-root", type=Path, required=True)
    parser.add_argument("--joint-detached-seal", type=Path, required=True)
    parser.add_argument("--joint-detached-seal-sha256", required=True)
    parser.add_argument("--joint-signature-envelope", type=Path, required=True)
    parser.add_argument("--joint-signature-envelope-sha256", required=True)
    parser.add_argument("--checkpoint-lineage-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--component-pre-sign-content-root-sha256", required=True)
    parser.add_argument("--class-handle-binding-sha256", required=True)
    parser.add_argument("--parity-receipt-sha256", required=True)
    parser.add_argument("--generation-lock-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--generation-config-sha256", required=True)
    parser.add_argument("--generation-code-sha256", required=True)
    parser.add_argument("--outer-content-root-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", default=MODE, choices=[MODE])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        before_root=args.before_root,
        before_seal=args.before_seal,
        expected_before_seal_sha256=args.before_seal_sha256,
        before_formal_policy=args.before_formal_policy,
        before_formal_policy_authorization=args.before_formal_policy_authorization,
        before_signed_policy_authorization_envelope=(
            args.before_signed_policy_authorization_envelope
        ),
        expected_before_signed_policy_authorization_envelope_sha256=(
            args.before_signed_policy_authorization_envelope_sha256
        ),
        after_root=args.after_root,
        after_seal=args.after_seal,
        expected_after_seal_sha256=args.after_seal_sha256,
        after_formal_policy=args.after_formal_policy,
        after_formal_policy_authorization=args.after_formal_policy_authorization,
        after_signed_policy_authorization_envelope=(
            args.after_signed_policy_authorization_envelope
        ),
        expected_after_signed_policy_authorization_envelope_sha256=(
            args.after_signed_policy_authorization_envelope_sha256
        ),
        joint_package_root=args.joint_package_root,
        joint_detached_seal=args.joint_detached_seal,
        expected_joint_detached_seal_sha256=args.joint_detached_seal_sha256,
        joint_signature_envelope=args.joint_signature_envelope,
        expected_joint_signature_envelope_sha256=(
            args.joint_signature_envelope_sha256
        ),
        expected_checkpoint_lineage_sha256=args.checkpoint_lineage_sha256,
        expected_runtime_sha256=args.runtime_sha256,
        expected_component_pre_sign_content_root_sha256=(
            args.component_pre_sign_content_root_sha256
        ),
        expected_class_handle_binding_sha256=args.class_handle_binding_sha256,
        expected_parity_receipt_sha256=args.parity_receipt_sha256,
        expected_generation_lock_sha256=args.generation_lock_sha256,
        expected_method_lock_sha256=args.method_lock_sha256,
        expected_generation_config_sha256=args.generation_config_sha256,
        expected_generation_code_sha256=args.generation_code_sha256,
        expected_outer_content_root_sha256=args.outer_content_root_sha256,
        output=args.output,
        device_name=args.device,
        mode=args.mode,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
