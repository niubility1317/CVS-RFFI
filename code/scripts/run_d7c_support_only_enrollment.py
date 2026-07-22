#!/usr/bin/env python3
"""Lock D7a->D7c from strict K10 sealed enrollment support only.

This runner intentionally exposes no query, truth, prediction, score, or
scorer argument. It verifies one before and one after ``enrollment_only``
package, locks D7c at K10, and emits K1/K5 prototype-only rebuild proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.somph_diagnostic_bundle_loader import (  # noqa: E402
    load_verified_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS  # noqa: E402
from cvsrffi.stage2_class_conditional_iq_head import (  # noqa: E402
    OPERATORS,
    ClassConditionalIQHeadState,
    ValidatedOperatorFeatureArtifact,
    build_validated_operator_feature_artifact,
    fit_class_conditional_head,
    register_absent_classes,
)
from cvsrffi.stage2_class_conditional_local_boundary import (  # noqa: E402
    ClassConditionalLocalBoundaryState,
    extend_class_conditional_local_boundary,
    fit_class_conditional_local_boundary,
    lock_k10_class_conditional_local_boundary_strategy,
    rebuild_from_locked_k10_strategy,
)
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


class D7cSupportRunnerError(ValueError):
    """Raised when enrollment-only or immutable-output invariants drift."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _write_json_new(path: Path, payload: Any) -> str:
    raw = _canonical(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_text_new(path: Path, text: str) -> str:
    raw = text.encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _member(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    rows = [dict(row) for row in manifest["members"] if row.get("kind") == kind]
    if len(rows) != 1:
        raise D7cSupportRunnerError(f"enrollment member drift: {kind}")
    return rows[0]


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    registration_state: str,
) -> None:
    required_false = (
        "clean_sample_access",
        "clean_derived_signal_access",
        "phase2_clean_dataset_reachable",
        "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
        "phase2_source_sample_access",
        "phase2_source_derived_signal_access",
        "phase2_source_cache_access",
        "phase2_source_label_access",
        "phase2_source_replay",
        "phase2_additional_leo_channel_state_generation",
        "phase2_post_reception_view_counts_as_additional_physical_sample",
        "phase2_query_post_reception_view_fit_access",
        "phase2_query_role_oracle_access",
        "phase2_query_true_batch_class_count_access",
        "phase2_query_class_quota_access",
        "phase2_query_batch_global_assignment",
    )
    if (
        manifest.get("profile") != "enrollment_only"
        or manifest.get("registration_state") != registration_state
        or int(manifest.get("k_shot", -1)) != 10
        or manifest.get("phase2_sample_view_policy")
        != "leo_weak_only_no_clean_access"
        or manifest.get("phase2_physical_sample_observation_policy")
        != "single_leo_weak_observation_per_physical_sample"
        or manifest.get("phase2_pretrained_artifact_policy")
        != "sealed_phase1_checkpoint_only"
        or manifest.get("phase2_query_decision_policy")
        != "per_sample_all_registered_classes"
        or not bool(
            manifest.get(
                "phase2_post_reception_view_from_fixed_received_iq_only"
            )
        )
        or any(bool(manifest.get(field, True)) for field in required_false)
        or tuple(manifest.get("target_channel_scenarios", ()))
        != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise D7cSupportRunnerError("enrollment package protocol drift")
    kinds = {str(row.get("kind")) for row in manifest.get("members", ())}
    required_kinds = {
        "feature_runtime",
        "method_lock",
        "overlay_provenance",
        *{f"support:{scenario}" for scenario in FORMAL_LEO_WEAK_SCENARIOS},
    }
    forbidden = ("query", "truth", "prediction", "score", "scorer")
    if kinds != required_kinds or any(
        any(token in kind.lower() for token in forbidden) for kind in kinds
    ):
        raise D7cSupportRunnerError("enrollment package member allowlist drift")


def _load_enrollment(
    root: Path,
    seal: Path,
    *,
    registration_state: str,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    if root.name != "enrollment_only" or "enrollment" not in seal.name:
        raise D7cSupportRunnerError("runner accepts enrollment-only paths")
    payloads, manifest, preopen_audit = load_verified_somph_predictor_bundle(
        root,
        detached_seal_path=seal,
        expected_seal_sha256=_sha256_file(seal),
    )
    _validate_manifest(manifest, registration_state=registration_state)
    return payloads, manifest, preopen_audit


def _payload_rows(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    class_handles = np.asarray(
        [str(row["class_handle"]) for row in manifest["registered_classes"]]
    )
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    labels = class_handles[indices].astype(str)
    classes, counts = np.unique(labels, return_counts=True)
    expected = {
        str(row["class_handle"]) for row in manifest["registered_classes"]
    }
    if (
        set(classes.tolist()) != expected
        or set(counts.tolist()) != {10}
        or set(ranks.tolist()) != set(range(10))
    ):
        raise D7cSupportRunnerError(
            f"strict K10-only support reachability drift: {scenario}"
        )
    order = np.asarray(
        sorted(
            range(len(indices)),
            key=lambda index: (str(labels[index]), int(ranks[index])),
        ),
        dtype=np.int64,
    )
    rows = {
        "iq": np.asarray(
            payload["support_leo_weak_iq"], dtype=np.float32
        )[order],
        "labels": labels[order],
        "ranks": ranks[order],
        "tokens": np.asarray(payload["support_tokens"]).astype(str)[order],
        "overlay_tokens": np.asarray(
            payload["support_overlay_tokens"]
        ).astype(str)[order],
        "satellite_seeds": np.asarray(
            payload["support_satellite_seeds"], dtype=np.int64
        )[order],
        "hashes": np.asarray(
            payload["support_post_channel_iq_sha256"]
        ).astype(str)[order],
    }
    computed = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in rows["iq"]
        ]
    )
    if (
        rows["iq"].ndim != 3
        or rows["iq"].shape[1] != 2
        or not np.isfinite(rows["iq"]).all()
        or not np.array_equal(computed, rows["hashes"])
        or len(set(rows["tokens"].tolist())) != len(rows["tokens"])
        or len(set(rows["overlay_tokens"].tolist()))
        != len(rows["overlay_tokens"])
        or len(set(rows["hashes"].tolist())) != len(rows["hashes"])
        or any(not str(value).startswith("sid_") for value in rows["tokens"])
        or any(
            not str(value).startswith("oid_") for value in rows["overlay_tokens"]
        )
    ):
        raise D7cSupportRunnerError(
            f"sealed K10 support payload drift: {scenario}"
        )
    return rows


class _SamplewiseFeatureCache:
    """Bitwise-reuse one fixed operator view within one support-only run."""

    def __init__(
        self,
        extractor: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._extractor = extractor
        self._cache: dict[str, tuple[bytes, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, view: np.ndarray) -> np.ndarray:
        rows = np.asarray(view)
        if (
            rows.dtype != np.float32
            or rows.ndim != 3
            or len(rows) != 1
            or rows.shape[1] != 2
            or not np.isfinite(rows).all()
        ):
            raise D7cSupportRunnerError(
                "formal D7c feature extraction must be samplewise float32 IQ"
            )
        raw = np.ascontiguousarray(rows).tobytes()
        key = hashlib.sha256(raw).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            if cached[0] != raw:
                raise D7cSupportRunnerError(
                    "samplewise feature cache SHA collision"
                )
            self.hits += 1
            return cached[1]
        feature = np.ascontiguousarray(
            self._extractor(rows), dtype=np.float32
        )
        if (
            feature.ndim != 2
            or feature.shape[0] != 1
            or feature.shape[1] < 1
            or not np.isfinite(feature).all()
        ):
            raise D7cSupportRunnerError(
                "samplewise feature extractor output drift"
            )
        feature.setflags(write=False)
        self._cache[key] = (raw, feature)
        self.misses += 1
        return feature

    @property
    def entry_count(self) -> int:
        return len(self._cache)


def _feature_extractor(
    model: torch.nn.Module,
    device: torch.device,
) -> _SamplewiseFeatureCache:
    def extract(view: np.ndarray) -> np.ndarray:
        zid = forward_zid160(model, view, device=device, batch_size=1)
        return registered_feature(view, zid)

    return _SamplewiseFeatureCache(extract)


def _artifact(
    rows: Mapping[str, np.ndarray],
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    *,
    indices: np.ndarray | None = None,
) -> ValidatedOperatorFeatureArtifact:
    selected = (
        np.arange(len(rows["labels"]), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    return build_validated_operator_feature_artifact(
        rows["iq"][selected],
        feature_extractor=feature_extractor,
        physical_sample_ids=rows["tokens"][selected].tolist(),
        parent_received_iq_sha256=rows["hashes"][selected].tolist(),
        operator_ids=OPERATORS,
    )


def _state_metadata(
    state: ClassConditionalLocalBoundaryState,
) -> dict[str, Any]:
    base = state.base_state
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "class_operators": list(base.class_operators),
        "prototype_sha256": [
            _array_sha256(base.prototypes[index])
            for index in range(len(base.classes))
        ],
        "calibrations": [
            {
                "operator_id": value.operator_id,
                "center": value.center,
                "scale": value.scale,
            }
            for value in base.calibrations
        ],
        "rival_class_handles": [
            state.classes[int(index)] for index in state.rival_indices
        ],
        "beta": state.beta.tolist(),
        "feature_dim": base.feature_dim,
        "used_operators": list(base.used_operators),
        "old_class_count": state.old_class_count,
        "registration_generation": base.registration_generation,
        "strategy_locked_k": state.strategy_locked_k,
        "support_lineage": [
            {
                "class_handle": label,
                "physical_sample_id": token,
                "parent_received_iq_sha256": digest,
            }
            for label, token, digest in zip(
                state.support_labels,
                state.support_physical_sample_ids,
                state.support_parent_received_iq_sha256,
            )
        ],
        "support_binding_fingerprints": list(
            state.support_binding_fingerprints
        ),
        "d7a_selection_trace": base.selection_trace,
        "d7c_support_audit": state.support_audit,
        "resource": state.resource_audit(),
    }


def _write_state_new(
    output: Path,
    *,
    stem: str,
    state: ClassConditionalLocalBoundaryState,
) -> dict[str, str]:
    npz_path = output / f"{stem}.npz"
    with npz_path.open("xb") as handle:
        np.savez(
            handle,
            prototypes=state.base_state.prototypes,
            rival_indices=state.rival_indices,
            beta=state.beta,
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz_path)
    metadata_path = output / f"{stem}.json"
    return {
        "npz_sha256": _sha256_file(npz_path),
        "metadata_sha256": _write_json_new(
            metadata_path, _state_metadata(state)
        ),
    }


def _per_class_rows(
    state: ClassConditionalLocalBoundaryState,
    *,
    scenario: str,
    registration_state: str,
) -> list[dict[str, Any]]:
    selection = state.support_audit["selection"]
    if registration_state == "before":
        baseline = selection["baseline"]["per_class_accuracy"]
        final = selection["combined_final"]["per_class_accuracy"]
        selected = {
            row["class_handle"]: row
            for row in selection["per_class_selection"]
        }
    else:
        baseline = {
            **selection["baseline_old"]["per_class_accuracy"],
            **selection["baseline_new"]["per_class_accuracy"],
        }
        final = {
            **selection["combined_final_old"]["per_class_accuracy"],
            **selection["combined_final_new"]["per_class_accuracy"],
        }
        selected = {
            row["class_handle"]: row
            for row in selection["per_new_class_selection"]
        }
    result = []
    for index, handle in enumerate(state.classes):
        trace = selected.get(handle)
        result.append(
            {
                "scenario": scenario,
                "registration_state": registration_state,
                "class_handle": handle,
                "lifecycle": (
                    "old_locked"
                    if registration_state == "after"
                    and index < state.old_class_count
                    else "registered"
                ),
                "operator_id": state.base_state.class_operators[index],
                "rival_class_handle": state.classes[
                    int(state.rival_indices[index])
                ],
                "beta": float(state.beta[index]),
                "support_baseline_accuracy": float(baseline[handle]),
                "support_final_accuracy": float(final[handle]),
                "support_non_degradation_pass": (
                    float(final[handle]) >= float(baseline[handle])
                ),
                "d7c_selection_trace": trace,
            }
        )
    return result


def _nested_indices(rows: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(rows["ranks"]) < int(k))


def _nested_proof(
    locked: ClassConditionalLocalBoundaryState,
    rows: Mapping[str, np.ndarray],
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    *,
    scenario: str,
    registration_state: str,
    k: int,
    output: Path,
) -> dict[str, Any]:
    indices = _nested_indices(rows, k)
    artifact = _artifact(rows, feature_extractor, indices=indices)
    rebuilt = rebuild_from_locked_k10_strategy(
        locked,
        artifact,
        rows["labels"][indices].tolist(),
        expected_k=k,
    )
    hashes = _write_state_new(
        output,
        stem=f"state_{scenario}_{registration_state}_k{k}",
        state=rebuilt,
    )
    return {
        "scenario": scenario,
        "registration_state": registration_state,
        "k": k,
        "support_count": len(indices),
        "support_count_per_class": k,
        "k10_lineage_prefix_verified": True,
        "operator_policy_locked": (
            rebuilt.base_state.class_operators
            == locked.base_state.class_operators
        ),
        "calibrations_locked": (
            rebuilt.base_state.calibrations
            == locked.base_state.calibrations
        ),
        "rivals_bitwise_locked": np.array_equal(
            rebuilt.rival_indices, locked.rival_indices
        ),
        "beta_bitwise_locked": np.array_equal(rebuilt.beta, locked.beta),
        "only_prototypes_rebuilt": True,
        "formal_lower_k_package_opened": False,
        "claim_scope": "nested_prefix_rebuild_proof_not_formal_K_evaluation",
        "resource": rebuilt.resource_audit(),
        "state_sha256": hashes,
    }


def _old_lineage_equal(
    before_rows: Mapping[str, np.ndarray],
    after_rows: Mapping[str, np.ndarray],
    old_handles: set[str],
) -> bool:
    before = sorted(
        (str(label), str(token), str(digest))
        for label, token, digest in zip(
            before_rows["labels"],
            before_rows["tokens"],
            before_rows["hashes"],
        )
    )
    after = sorted(
        (str(label), str(token), str(digest))
        for label, token, digest in zip(
            after_rows["labels"],
            after_rows["tokens"],
            after_rows["hashes"],
        )
        if str(label) in old_handles
    )
    return before == after


def _support_summary(
    state: ClassConditionalLocalBoundaryState,
    *,
    registration_state: str,
) -> dict[str, Any]:
    selection = state.support_audit["selection"]
    if registration_state == "before":
        baseline = selection["baseline"]
        final = selection["combined_final"]
        return {
            "overall_baseline": baseline["overall_accuracy"],
            "overall_final": final["overall_accuracy"],
            "floor_baseline": baseline["min_class_accuracy"],
            "floor_final": final["min_class_accuracy"],
            "every_class_non_degradation_pass": all(
                final["per_class_accuracy"][label]
                >= baseline["per_class_accuracy"][label]
                for label in state.classes
            ),
        }
    baseline_old = selection["baseline_old"]
    final_old = selection["combined_final_old"]
    baseline_new = selection["baseline_new"]
    final_new = selection["combined_final_new"]
    return {
        "old_overall_baseline": baseline_old["overall_accuracy"],
        "old_overall_final": final_old["overall_accuracy"],
        "old_floor_baseline": baseline_old["min_class_accuracy"],
        "old_floor_final": final_old["min_class_accuracy"],
        "new_overall_baseline": baseline_new["overall_accuracy"],
        "new_overall_final": final_new["overall_accuracy"],
        "new_floor_baseline": baseline_new["min_class_accuracy"],
        "new_floor_final": final_new["min_class_accuracy"],
        "every_old_class_non_degradation_pass": all(
            final_old["per_class_accuracy"][label]
            >= baseline_old["per_class_accuracy"][label]
            for label in final_old["per_class_accuracy"]
        ),
        "every_new_class_non_degradation_pass": all(
            final_new["per_class_accuracy"][label]
            >= baseline_new["per_class_accuracy"][label]
            for label in final_new["per_class_accuracy"]
        ),
    }


def run(
    *,
    before_root: Path,
    before_seal: Path,
    after_root: Path,
    after_seal: Path,
    output: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    if output.exists():
        raise D7cSupportRunnerError("output already exists")
    before_payloads, before_manifest, before_preopen = _load_enrollment(
        before_root, before_seal, registration_state="before"
    )
    after_payloads, after_manifest, after_preopen = _load_enrollment(
        after_root, after_seal, registration_state="after"
    )
    if (
        before_manifest["receiver"] != after_manifest["receiver"]
        or int(before_manifest["seed"]) != int(after_manifest["seed"])
        or before_manifest["feature_runtime_sha256"]
        != after_manifest["feature_runtime_sha256"]
        or before_manifest["phase1_checkpoint_sha256"]
        != after_manifest["phase1_checkpoint_sha256"]
    ):
        raise D7cSupportRunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"])
        for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"])
        for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D7cSupportRunnerError("absent-class registration drift")
    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root,
        _member(before_manifest, "feature_runtime"),
        device=device,
    )
    feature_extractor = _feature_extractor(model, device)
    output.mkdir(parents=True)
    state_hashes: dict[str, Any] = {}
    nested_proofs: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    scenario_results: dict[str, Any] = {}
    prior_tokens: set[str] = set()
    prior_hashes: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _payload_rows(
            before_payloads[scenario],
            before_manifest,
            scenario=scenario,
        )
        after_rows = _payload_rows(
            after_payloads[scenario],
            after_manifest,
            scenario=scenario,
        )
        if not _old_lineage_equal(before_rows, after_rows, before_handles):
            raise D7cSupportRunnerError(
                "old support lineage changed across registration"
            )
        scenario_tokens = set(after_rows["tokens"].tolist())
        scenario_hashes = set(after_rows["hashes"].tolist())
        if (
            scenario_tokens.intersection(prior_tokens)
            or scenario_hashes.intersection(prior_hashes)
        ):
            raise D7cSupportRunnerError("cross-scenario support lineage reuse")
        prior_tokens.update(scenario_tokens)
        prior_hashes.update(scenario_hashes)
        before_artifact = _artifact(before_rows, feature_extractor)
        after_artifact = _artifact(after_rows, feature_extractor)
        before_base = fit_class_conditional_head(
            before_artifact, before_rows["labels"].tolist()
        )
        before_state = lock_k10_class_conditional_local_boundary_strategy(
            fit_class_conditional_local_boundary(
                before_base,
                before_artifact,
                before_rows["labels"].tolist(),
            )
        )
        after_base = register_absent_classes(
            before_base,
            after_artifact,
            after_rows["labels"].tolist(),
        )
        after_state = lock_k10_class_conditional_local_boundary_strategy(
            extend_class_conditional_local_boundary(
                before_state,
                after_base,
                after_artifact,
                after_rows["labels"].tolist(),
            )
        )
        for registration_state, state, rows in (
            ("before", before_state, before_rows),
            ("after", after_state, after_rows),
        ):
            state_hashes[f"{scenario}:{registration_state}:k10"] = (
                _write_state_new(
                    output,
                    stem=f"state_{scenario}_{registration_state}_k10",
                    state=state,
                )
            )
            per_class_rows.extend(
                _per_class_rows(
                    state,
                    scenario=scenario,
                    registration_state=registration_state,
                )
            )
            for k in (1, 5):
                nested_proofs.append(
                    _nested_proof(
                        state,
                        rows,
                        feature_extractor,
                        scenario=scenario,
                        registration_state=registration_state,
                        k=k,
                        output=output,
                    )
                )
        scenario_results[scenario] = {
            "before": {
                "registered_class_count": before_state.class_count,
                "support_count_per_class": 10,
                "support_metrics": _support_summary(
                    before_state, registration_state="before"
                ),
                "resource": before_state.resource_audit(),
            },
            "after": {
                "registered_class_count": after_state.class_count,
                "support_count_per_class": 10,
                "support_metrics": _support_summary(
                    after_state, registration_state="after"
                ),
                "resource": after_state.resource_audit(),
                "old_state_bitwise_locked": True,
                "old_support_lineage_and_feature_binding_verified": True,
            },
        }
    audit = {
        "schema": "cvs.phase2.d7c_support_only_runner_audit.v1",
        "diagnostic_only": True,
        "status": "SUPPORT_ONLY_D7C_LOCKED_NO_QUERY_OPEN",
        "claim_scope": (
            "support_selection_state_and_nested_rebuild_only_"
            "no_query_performance_claim"
        ),
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "new_class_count": (
            int(after_manifest["registered_class_count"])
            - int(before_manifest["registered_class_count"])
        ),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "opened_package_profiles": [
            "before:enrollment_only",
            "after:enrollment_only",
        ],
        "opened_member_kinds": sorted(
            {
                str(row["kind"]) for row in before_manifest["members"]
            }
            | {str(row["kind"]) for row in after_manifest["members"]}
        ),
        "query_package_opened": False,
        "query_truth_opened": False,
        "query_prediction_opened": False,
        "query_score_opened": False,
        "scorer_opened": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "additional_leo_channel_state_generation": False,
        "post_reception_views_count_as_additional_k": False,
        "operator_feature_extraction_batch_size": 1,
        "operator_feature_view_seed": 0,
        "operator_feature_binding_fields": [
            "physical_sample_id",
            "parent_received_iq_sha256",
            "operator_id",
            "view_seed",
            "feature_sha256",
        ],
        "feature_extractor_binding": {
            "sealed_feature_runtime_sha256": before_manifest[
                "feature_runtime_sha256"
            ],
            "phase1_checkpoint_sha256": before_manifest[
                "phase1_checkpoint_sha256"
            ],
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "d7a_module_sha256": _sha256_file(
                CODE / "cvsrffi" / "stage2_class_conditional_iq_head.py"
            ),
            "d7c_module_sha256": _sha256_file(
                CODE
                / "cvsrffi"
                / "stage2_class_conditional_local_boundary.py"
            ),
            "samplewise_fixed_view_cache": {
                "policy": (
                    "reuse_bitwise_feature_for_identical_fixed_operator_view_"
                    "within_same_support_only_process"
                ),
                "cache_key": "sha256_float32_operator_view_bytes",
                "cache_entries": feature_extractor.entry_count,
                "cache_hits": feature_extractor.hits,
                "cache_misses": feature_extractor.misses,
                "old_support_second_gpu_forward_required": False,
            },
        },
        "scenario_results": scenario_results,
        "per_class_support_rows": per_class_rows,
        "nested_k_rebuild_proofs": nested_proofs,
        "state_sha256": state_hashes,
        "before_package_root_sha256": before_manifest[
            "package_root_sha256"
        ],
        "after_package_root_sha256": after_manifest["package_root_sha256"],
        "before_seal_sha256": _sha256_file(before_seal),
        "after_seal_sha256": _sha256_file(after_seal),
        "preopen_audit": {
            "before": before_preopen,
            "after": after_preopen,
        },
        "lower_k_boundary": (
            "K1_K5_are_nested_prefix_prototype_rebuild_proofs_from_K10_"
            "support_not_formal_lower_K_package_or_performance_evidence"
        ),
    }
    audit_sha = _write_json_new(output / "support_audit.json", audit)
    report_lines = [
        "# D7c sealed enrollment support-only锁定",
        "",
        "状态：只打开before/after的`enrollment_only`包；未创建或打开query、truth、"
        "prediction、score或scorer。本artifact不包含query性能结论。",
        "",
        "D7a和D7c只从每个固定LEO_weak接收IQ逐样本生成三个预登记operator view；"
        "每个view绑定父IQ SHA、operator、固定view seed 0和feature SHA，view不增加K，"
        "不生成额外LEO信道状态。",
        "",
        "|场景|状态|类数|support overall baseline/final|"
        "support floor baseline/final|状态bytes|去重operator数|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before = scenario_results[scenario]["before"]
        after = scenario_results[scenario]["after"]
        for registration_state, row in (("before", before), ("after", after)):
            metrics = row["support_metrics"]
            if registration_state == "before":
                overall = (
                    f"{metrics['overall_baseline']:.4f}/"
                    f"{metrics['overall_final']:.4f}"
                )
                floor = (
                    f"{metrics['floor_baseline']:.4f}/"
                    f"{metrics['floor_final']:.4f}"
                )
            else:
                overall = (
                    f"old {metrics['old_overall_baseline']:.4f}/"
                    f"{metrics['old_overall_final']:.4f}; new "
                    f"{metrics['new_overall_baseline']:.4f}/"
                    f"{metrics['new_overall_final']:.4f}"
                )
                floor = (
                    f"old {metrics['old_floor_baseline']:.4f}/"
                    f"{metrics['old_floor_final']:.4f}; new "
                    f"{metrics['new_floor_baseline']:.4f}/"
                    f"{metrics['new_floor_final']:.4f}"
                )
            resource = row["resource"]
            report_lines.append(
                f"|`{scenario}`|{registration_state}|"
                f"{row['registered_class_count']}|{overall}|{floor}|"
                f"{resource['persistent_state_bytes']}|"
                f"{resource['used_operator_count']}|"
            )
    report_lines.extend(
        [
            "",
            "K1/K5仅用K10有序物理support lineage的前缀重建prototype；operator、"
            "calibration、rival和beta保持锁定。这些是support-only嵌套重建证明，"
            "不是独立K1/K5密封包或性能证据。",
            "",
            "逐类operator、rival、beta、support baseline/final、非退化判断和资源"
            "明细见`support_audit.json`。端到端MAC、时延和峰值显存仍为`None`，"
            "不得据head侧MAC形成正式Pareto声明。",
            "",
        ]
    )
    report_sha = _write_text_new(
        output / "report.md", "\n".join(report_lines)
    )
    commit = {
        "schema": "cvs.phase2.d7c_support_only_commit.v1",
        "diagnostic_only": True,
        "status": audit["status"],
        "support_audit_sha256": audit_sha,
        "report_sha256": report_sha,
        "state_sha256": state_hashes,
        "query_package_opened": False,
        "query_truth_opened": False,
        "query_prediction_opened": False,
        "query_score_opened": False,
        "scorer_opened": False,
        "independent_performance_claim": False,
    }
    commit_sha = _write_json_new(output / "COMMIT.json", commit)
    return {
        "status": commit["status"],
        "output": str(output),
        "commit_sha256": commit_sha,
        "support_audit_sha256": audit_sha,
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "nested_k_proof_count": len(nested_proofs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lock D7a->D7c from strict K10 before/after enrollment support"
        )
    )
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        before_root=args.before_root.resolve(),
        before_seal=args.before_seal.resolve(),
        after_root=args.after_root.resolve(),
        after_seal=args.after_seal.resolve(),
        output=args.output.resolve(),
        device_name=str(args.device),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
