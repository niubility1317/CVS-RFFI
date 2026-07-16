"""Run D17-SPRTDR on sealed strict-K10 enrollment packages only.

The runner has no query, scorer, truth, or formal-authority input.  Candidate
selection is support-only and requires every class-level gate in every outer
leave-two-out fold of every LEO_weak scenario.  Any failed positive arm causes
selection of the canonical true-zero state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (CODE, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS  # noqa: E402
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)
from cvsrffi.stage2_sprtdr import (  # noqa: E402
    ALPHA_GRID,
    SprtdrHyperparameters,
    SprtdrState,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
)
from run_d14_support_only_pairwise_fisher_guard import (  # noqa: E402
    _build_feature_artifact,
    _canonical,
    _feature_provenance,
    _load_enrollment,
    _member,
    _normalise,
    _payload_rows,
    _readonly,
    _sha256_file,
    _write_json_new,
    _write_jsonl_new,
    _write_text_new,
)


STATE_ARRAY_FIELDS = (
    "prototypes",
    "old_pairs",
    "old_dims",
    "old_mu_a",
    "old_var_a",
    "old_mu_b",
    "old_var_b",
    "old_mid",
    "old_gap",
    "old_alpha_pos",
    "old_alpha_neg",
    "new_rivals",
    "new_dims",
    "new_mu",
    "new_var",
    "rival_mu",
    "rival_var",
    "new_mid",
    "new_gap",
    "new_alpha_pos",
    "new_alpha_neg",
    "old_floor",
    "new_floor",
)


class D17RunnerError(ValueError):
    """Raised when the D17 support-only runner fails closed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _candidates(k_shot: int = 10) -> tuple[SprtdrHyperparameters, ...]:
    if int(k_shot) not in (1, 5, 10):
        raise D17RunnerError("D17 runner supports only exact K1/K5/K10")
    zero = SprtdrHyperparameters(
        candidate_id="d17_z0_true_zero_base",
        rank=0,
        margin_band=0.0,
        max_old_edges=0,
        max_new_rivals=0,
        force_zero=True,
    )
    if int(k_shot) == 1:
        return (zero,)
    return (
        zero,
        SprtdrHyperparameters(
            candidate_id="d17_sprtdr_mb002", margin_band=0.02
        ),
        SprtdrHyperparameters(
            candidate_id="d17_sprtdr_mb004", margin_band=0.04
        ),
    )


def _candidate_lock(
    candidates: tuple[SprtdrHyperparameters, ...], *, k_shot: int
) -> dict[str, Any]:
    rows = [
        {
            "candidate_id": value.candidate_id,
            "rank": value.rank,
            "margin_band": value.margin_band,
            "max_old_edges": value.max_old_edges,
            "max_new_rivals": value.max_new_rivals,
            "operator_id": value.operator_id,
            "force_zero": value.force_zero,
            "student_t_nu": 3,
            "activation_threshold": 0.5,
            "amplitude_grid": list(ALPHA_GRID),
        }
        for value in candidates
    ]
    return {
        "selection_scope": (
            "one_strict_k10_candidate_shared_by_all_three_scenarios"
            if int(k_shot) == 10
            else "nonselection_exact_k_support_audit"
        ),
        "k_shot": int(k_shot),
        "candidate_count": len(rows),
        "candidates": rows,
        "lock_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


def _validate_exact_k_rows(
    rows: Mapping[str, np.ndarray], *, k_shot: int, scenario: str
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    tokens = np.asarray(rows["tokens"]).astype(str)
    hashes = np.asarray(rows["hashes"]).astype(str)
    classes, counts = np.unique(labels, return_counts=True)
    valid = bool(
        int(k_shot) in (1, 5, 10)
        and len(classes) >= 2
        and set(counts.tolist()) == {int(k_shot)}
        and all(
            set(ranks[labels == label].tolist()) == set(range(int(k_shot)))
            for label in classes
        )
        and len(tokens) == len(set(tokens.tolist()))
        and len(hashes) == len(set(hashes.tolist()))
        and all(len(value) == 64 for value in hashes.tolist())
    )
    if not valid:
        raise D17RunnerError(f"exact physical K-shot drift: {scenario}")
    return {
        "scenario": scenario,
        "k_shot": int(k_shot),
        "class_count": int(len(classes)),
        "physical_support_count": int(len(tokens)),
        "unique_physical_sample_id_count": int(len(set(tokens.tolist()))),
        "unique_parent_received_iq_sha256_count": int(
            len(set(hashes.tolist()))
        ),
        "exact_k_pass": True,
    }


def _old_reuse_lock(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]
) -> None:
    old_classes = set(before["labels"].tolist())

    def keyed(rows: Mapping[str, np.ndarray]):
        return {
            (str(rows["labels"][index]), int(rows["ranks"][index])): (
                str(rows["tokens"][index]), str(rows["hashes"][index])
            )
            for index in range(len(rows["labels"]))
            if str(rows["labels"][index]) in old_classes
        }

    if keyed(before) != keyed(after):
        raise D17RunnerError("before/after old physical support exact-reuse drift")


def _cross_scenario_disjointness(
    by_scenario: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    token_sets = {
        scenario: set(np.asarray(rows["tokens"]).astype(str).tolist())
        for scenario, rows in by_scenario.items()
    }
    hash_sets = {
        scenario: set(np.asarray(rows["hashes"]).astype(str).tolist())
        for scenario, rows in by_scenario.items()
    }
    pairs = []
    scenarios = tuple(FORMAL_LEO_WEAK_SCENARIOS)
    for left_index, left in enumerate(scenarios):
        for right in scenarios[left_index + 1 :]:
            token_overlap = token_sets[left] & token_sets[right]
            hash_overlap = hash_sets[left] & hash_sets[right]
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "physical_sample_id_overlap_count": len(token_overlap),
                    "parent_received_iq_sha256_overlap_count": len(hash_overlap),
                    "pass": not token_overlap and not hash_overlap,
                }
            )
    if not all(row["pass"] for row in pairs):
        raise D17RunnerError(
            "physical support or received-IQ parent reused across LEO scenarios"
        )
    return {
        "policy": "pairwise_disjoint_after_enrollment_union",
        "pairs": pairs,
        "all_pairwise_disjoint": True,
    }


def _manifest_binding(
    before_manifest: Mapping[str, Any], after_manifest: Mapping[str, Any]
) -> None:
    if (
        before_manifest["receiver"] != after_manifest["receiver"]
        or int(before_manifest["seed"]) != int(after_manifest["seed"])
        or before_manifest["feature_runtime_sha256"]
        != after_manifest["feature_runtime_sha256"]
        or before_manifest["phase1_checkpoint_sha256"]
        != after_manifest["phase1_checkpoint_sha256"]
    ):
        raise D17RunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"])
        for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"])
        for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D17RunnerError("new-class registration set drift")


def _all_true(values: Mapping[str, bool]) -> bool:
    return bool(values) and all(bool(value) for value in values.values())


def _floor_role_gate(
    fold: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    handles = sorted(set(fold.get("floor_handles", {}).get(role, ())))
    rows: dict[str, Any] = {}
    strict = False
    nondegraded = True
    for handle in handles:
        if role == "old":
            candidate = float(fold["after_old"]["per_class_accuracy"][handle])
            baseline = float(
                fold["base_after_old"]["per_class_accuracy"][handle]
            )
            before = float(fold["before_old"]["per_class_accuracy"][handle])
            okay = candidate + 1.0e-12 >= max(baseline, before)
            rows[handle] = {
                "before": before,
                "z0_after": baseline,
                "candidate_after": candidate,
                "nondegraded": okay,
                "strict_gain_vs_z0": candidate > baseline + 1.0e-12,
            }
        else:
            candidate = float(fold["after_new"]["per_class_accuracy"][handle])
            baseline = float(
                fold["base_after_new"]["per_class_accuracy"][handle]
            )
            okay = candidate + 1.0e-12 >= baseline
            rows[handle] = {
                "z0_after": baseline,
                "candidate_after": candidate,
                "nondegraded": okay,
                "strict_gain_vs_z0": candidate > baseline + 1.0e-12,
            }
        nondegraded = nondegraded and okay
        strict = strict or bool(rows[handle]["strict_gain_vs_z0"])
    return {
        "role": role,
        "handles": handles,
        "per_class": rows,
        "nonempty": bool(handles),
        "nondegraded": bool(handles and nondegraded),
        "strict_gain_vs_z0": bool(handles and strict),
    }


def _enabled_edge_count(fold: Mapping[str, Any]) -> int:
    old_edges = len(fold.get("old_pairs", ()))
    new_edges = sum(
        int(value) >= 0
        for rivals in fold.get("new_rivals", ())
        for value in rivals
    )
    return int(old_edges + new_edges)


def _scenario_gate(
    result: Mapping[str, Any], *, force_zero: bool
) -> dict[str, Any]:
    fold_rows = []
    for fold in result["folds"]:
        vs_z0 = fold["candidate_vs_z0_per_class_non_degraded"]
        old_after_vs_before = {
            handle: (
                float(fold["after_old"]["per_class_accuracy"][handle])
                + 1.0e-12
                >= float(fold["before_old"]["per_class_accuracy"][handle])
            )
            for handle in fold["before_old"]["per_class_accuracy"]
        }
        old_floor = _floor_role_gate(fold, role="old")
        new_floor = _floor_role_gate(fold, role="new")
        enabled_edge_count = _enabled_edge_count(fold)
        decision_forgetting_pass = bool(
            _all_true(old_after_vs_before)
            and float(fold["old_forgetting"]) <= 1.0e-12
        )
        gate = bool(
            not force_zero
            and _all_true(vs_z0["before_old"])
            and _all_true(vs_z0["after_old"])
            and _all_true(vs_z0["after_new"])
            and decision_forgetting_pass
            and float(fold["joint"]["overall_accuracy"]) + 1.0e-12
            >= float(fold["base_joint"]["overall_accuracy"])
            and float(fold["H_old_new"]) + 1.0e-12
            >= float(fold["base_H_old_new"])
            and bool(fold["old_score_bitwise_locked"])
            and old_floor["nondegraded"]
            and old_floor["strict_gain_vs_z0"]
            and new_floor["nondegraded"]
            and new_floor["strict_gain_vs_z0"]
            and enabled_edge_count > 0
        )
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "gate_pass": gate,
                "enabled_edge_count": enabled_edge_count,
                "before_old_per_class_nondegraded_vs_z0": _all_true(
                    vs_z0["before_old"]
                ),
                "after_old_per_class_nondegraded_vs_z0": _all_true(
                    vs_z0["after_old"]
                ),
                "after_new_per_class_nondegraded_vs_z0": _all_true(
                    vs_z0["after_new"]
                ),
                "old_decision_after_registration_nondegraded_per_class": (
                    _all_true(old_after_vs_before)
                ),
                "old_decision_forgetting": float(fold["old_forgetting"]),
                "old_decision_forgetting_pass": decision_forgetting_pass,
                "old_score_columns_bitwise_locked": bool(
                    fold["old_score_bitwise_locked"]
                ),
                "joint_nondegraded_vs_z0": bool(
                    float(fold["joint"]["overall_accuracy"]) + 1.0e-12
                    >= float(fold["base_joint"]["overall_accuracy"])
                ),
                "H_nondegraded_vs_z0": bool(
                    float(fold["H_old_new"]) + 1.0e-12
                    >= float(fold["base_H_old_new"])
                ),
                "old_floor": old_floor,
                "new_floor": new_floor,
            }
        )
    return {
        "all_folds_gate_pass": bool(
            fold_rows and all(row["gate_pass"] for row in fold_rows)
        ),
        "old_floor_strict_gain_in_every_fold": bool(
            fold_rows
            and all(row["old_floor"]["strict_gain_vs_z0"] for row in fold_rows)
        ),
        "new_floor_strict_gain_in_every_fold": bool(
            fold_rows
            and all(row["new_floor"]["strict_gain_vs_z0"] for row in fold_rows)
        ),
        "folds": fold_rows,
    }


def _aggregate_floor(result: Mapping[str, Any]) -> dict[str, Any]:
    counts = {"old": {}, "new": {}}
    for fold in result["folds"]:
        for role in ("old", "new"):
            for handle in set(fold.get("floor_handles", {}).get(role, ())):
                counts[role][handle] = counts[role].get(handle, 0) + 1

    def block(role: str, key: str, base_key: str) -> dict[str, Any]:
        current = result[key]["per_class_accuracy"]
        baseline = result[base_key]["per_class_accuracy"]
        handles = sorted(handle for handle in counts[role] if handle in current)
        return {
            "handles": handles,
            "selection_fold_count": {
                handle: counts[role][handle] for handle in handles
            },
            "candidate_per_class_accuracy": {
                handle: current[handle] for handle in handles
            },
            "z0_per_class_accuracy": {
                handle: baseline[handle] for handle in handles
            },
            "candidate_min_accuracy": (
                min(current[handle] for handle in handles) if handles else None
            ),
            "z0_min_accuracy": (
                min(baseline[handle] for handle in handles) if handles else None
            ),
        }

    return {
        "old_floor": block("old", "after_old", "base_after_old"),
        "new_floor": block("new", "after_new", "base_after_new"),
    }


def _candidate_summary(
    candidate: SprtdrHyperparameters,
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gates = {
        scenario: _scenario_gate(
            evaluations[scenario], force_zero=candidate.force_zero
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    floors = {
        scenario: _aggregate_floor(evaluations[scenario])
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    old_floor = [
        value["old_floor"]["candidate_min_accuracy"]
        for value in floors.values()
        if value["old_floor"]["candidate_min_accuracy"] is not None
    ]
    new_floor = [
        value["new_floor"]["candidate_min_accuracy"]
        for value in floors.values()
        if value["new_floor"]["candidate_min_accuracy"] is not None
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "force_zero": candidate.force_zero,
        "margin_band": candidate.margin_band,
        "all_scenario_all_fold_gate_pass": bool(
            not candidate.force_zero
            and all(value["all_folds_gate_pass"] for value in gates.values())
        ),
        "scenario_gates": gates,
        "floors": floors,
        "worst_old_floor": min(old_floor) if old_floor else None,
        "worst_new_floor": min(new_floor) if new_floor else None,
        "mean_H_old_new": float(
            np.mean([value["H_old_new"] for value in evaluations.values()])
        ),
        "mean_joint_accuracy": float(
            np.mean(
                [value["joint"]["overall_accuracy"] for value in evaluations.values()]
            )
        ),
        "worst_old_decision_forgetting": float(
            max(value["old_forgetting"] for value in evaluations.values())
        ),
    }


def _select_candidate(
    rows: list[Mapping[str, Any]],
    candidates: tuple[SprtdrHyperparameters, ...],
) -> tuple[str, bool]:
    passing = [
        row
        for row in rows
        if bool(row["all_scenario_all_fold_gate_pass"])
        and not bool(row["force_zero"])
    ]
    passing.sort(
        key=lambda row: (
            min(float(row["worst_old_floor"]), float(row["worst_new_floor"])),
            -float(row["worst_old_decision_forgetting"]),
            float(row["mean_H_old_new"]),
            float(row["mean_joint_accuracy"]),
            -float(row["margin_band"]),
        ),
        reverse=True,
    )
    selected = (
        str(passing[0]["candidate_id"])
        if passing
        else "d17_z0_true_zero_base"
    )
    hp = next(value for value in candidates if value.candidate_id == selected)
    return selected, bool(passing and not hp.force_zero)


def _state_arrays(state: SprtdrState) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(getattr(state, name))
        for name in STATE_ARRAY_FIELDS
    }


def _state_metadata(state: SprtdrState) -> dict[str, Any]:
    hp = state.hyperparameters
    return {
        "schema": state.schema,
        "candidate_id": state.candidate_id,
        "classes": list(state.classes),
        "hyperparameters": {
            "candidate_id": hp.candidate_id,
            "rank": hp.rank,
            "margin_band": hp.margin_band,
            "max_old_edges": hp.max_old_edges,
            "max_new_rivals": hp.max_new_rivals,
            "operator_id": hp.operator_id,
            "force_zero": hp.force_zero,
        },
        "feature_dim": state.feature_dim,
        "k_shot": state.k_shot,
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "resource": dict(state.resource),
        "support_feature_artifact_sha256": (
            state.support_feature_artifact_sha256
        ),
        "support_selection_sha256": state.support_selection_sha256,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "feature_code_sha256": state.feature_code_sha256,
        "sealed_phase1_checkpoint_sha256": (
            state.sealed_phase1_checkpoint_sha256
        ),
        "operator_id": state.operator_id,
        "view_seed": state.view_seed,
        "state_content_sha256": state.state_content_sha256,
    }


def _load_state(
    state_dir: Path, *, expected_commit_sha256: str
) -> tuple[SprtdrState, dict[str, Any]]:
    expected_names = {"state.npz", "metadata.json", "COMMIT"}
    if (
        not state_dir.is_dir()
        or state_dir.is_symlink()
        or {value.name for value in state_dir.iterdir()} != expected_names
        or any(
            value.is_symlink() or not value.is_file()
            for value in state_dir.iterdir()
        )
    ):
        raise D17RunnerError("sealed state member allowlist drift")
    commit_path = state_dir / "COMMIT"
    if (
        len(expected_commit_sha256) != 64
        or _sha256_file(commit_path) != expected_commit_sha256
    ):
        raise D17RunnerError("sealed state COMMIT SHA mismatch")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if (
        commit.get("schema") != "cvs.phase2.d17_state_commit.v1"
        or set(commit.get("members", {})) != {"state.npz", "metadata.json"}
    ):
        raise D17RunnerError("sealed state COMMIT schema drift")
    for name in ("state.npz", "metadata.json"):
        path = state_dir / name
        member = commit["members"][name]
        if (
            member.get("sha256") != _sha256_file(path)
            or int(member.get("bytes", -1)) != path.stat().st_size
        ):
            raise D17RunnerError("sealed state member hash/size mismatch")
    metadata = json.loads(
        (state_dir / "metadata.json").read_text(encoding="utf-8")
    )
    with np.load(state_dir / "state.npz", allow_pickle=False) as loaded:
        if set(loaded.files) != set(STATE_ARRAY_FIELDS):
            raise D17RunnerError("sealed state array allowlist drift")
        arrays = {
            name: np.ascontiguousarray(loaded[name])
            for name in STATE_ARRAY_FIELDS
        }
    hp = SprtdrHyperparameters(**metadata["hyperparameters"])
    state = SprtdrState(
        schema=metadata["schema"],
        candidate_id=metadata["candidate_id"],
        classes=tuple(metadata["classes"]),
        hyperparameters=hp,
        feature_dim=int(metadata["feature_dim"]),
        k_shot=int(metadata["k_shot"]),
        old_class_count=int(metadata["old_class_count"]),
        registration_generation=int(metadata["registration_generation"]),
        resource=dict(metadata["resource"]),
        support_feature_artifact_sha256=(
            metadata["support_feature_artifact_sha256"]
        ),
        support_selection_sha256=metadata["support_selection_sha256"],
        sealed_runtime_sha256=metadata["sealed_runtime_sha256"],
        feature_code_sha256=metadata["feature_code_sha256"],
        sealed_phase1_checkpoint_sha256=(
            metadata["sealed_phase1_checkpoint_sha256"]
        ),
        operator_id=metadata["operator_id"],
        view_seed=int(metadata["view_seed"]),
        state_content_sha256=metadata["state_content_sha256"],
        **arrays,
    )
    if (
        state.state_content_sha256 != commit.get("state_content_sha256")
        or state.state_content_sha256 != metadata["state_content_sha256"]
    ):
        raise D17RunnerError("sealed state semantic SHA drift")
    return state, commit


def _write_state_roundtrip(
    state_dir: Path, *, state: SprtdrState
) -> dict[str, Any]:
    if state_dir.exists():
        raise D17RunnerError("state output path already exists")
    state_dir.mkdir(parents=True, exist_ok=False)
    npz_path = state_dir / "state.npz"
    with npz_path.open("xb") as handle:
        np.savez_compressed(handle, **_state_arrays(state))
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz_path)
    metadata_path = state_dir / "metadata.json"
    metadata_sha256 = _write_json_new(
        metadata_path, _state_metadata(state)
    )
    commit = {
        "schema": "cvs.phase2.d17_state_commit.v1",
        "serialization": "canonical_npz_no_pickle",
        "state_content_sha256": state.state_content_sha256,
        "members": {
            "state.npz": {
                "sha256": _sha256_file(npz_path),
                "bytes": npz_path.stat().st_size,
            },
            "metadata.json": {
                "sha256": metadata_sha256,
                "bytes": metadata_path.stat().st_size,
            },
        },
    }
    commit_path = state_dir / "COMMIT"
    commit_sha256 = _write_json_new(commit_path, commit)
    rebuilt, loaded_commit = _load_state(
        state_dir, expected_commit_sha256=commit_sha256
    )
    original_arrays = _state_arrays(state)
    rebuilt_arrays = _state_arrays(rebuilt)
    if any(
        not np.array_equal(original_arrays[name], rebuilt_arrays[name])
        for name in STATE_ARRAY_FIELDS
    ):
        raise D17RunnerError("sealed state array roundtrip mismatch")
    probe = np.random.default_rng(20260717).normal(
        size=(7, state.feature_dim)
    ).astype(np.float32)
    original_scores = _score_numpy(probe, state)
    rebuilt_scores = _score_numpy(probe, rebuilt)
    if (
        rebuilt.state_content_sha256 != state.state_content_sha256
        or not np.array_equal(original_scores, rebuilt_scores)
    ):
        raise D17RunnerError("sealed state score roundtrip mismatch")
    total_bytes = sum(
        (state_dir / name).stat().st_size
        for name in ("state.npz", "metadata.json", "COMMIT")
    )
    return {
        "state_directory": str(state_dir),
        "serialization": loaded_commit["serialization"],
        "state_content_sha256": state.state_content_sha256,
        "state_npz_sha256": commit["members"]["state.npz"]["sha256"],
        "state_npz_bytes": commit["members"]["state.npz"]["bytes"],
        "metadata_sha256": metadata_sha256,
        "metadata_bytes": commit["members"]["metadata.json"]["bytes"],
        "commit_sha256": commit_sha256,
        "commit_bytes": commit_path.stat().st_size,
        "serialized_state_total_bytes": total_bytes,
        "serialized_state_under_50kib": total_bytes < 50 * 1024,
        "serialized_state_under_256kib": total_bytes < 256 * 1024,
        "member_allowlist_verified": True,
        "semantic_state_validation_verified": True,
        "state_sha_roundtrip_verified": True,
        "fixed_probe_score_bitwise_verified": True,
    }


def _state_edge_audit(state) -> dict[str, Any]:
    old_edges = int(len(state.old_pairs))
    new_edges = int(np.sum(state.new_rivals >= 0))
    return {
        "state_content_sha256": state.state_content_sha256,
        "support_feature_artifact_sha256": (
            state.support_feature_artifact_sha256
        ),
        "support_selection_sha256": state.support_selection_sha256,
        "registration_generation": int(state.registration_generation),
        "old_class_count": int(state.old_class_count),
        "registered_class_count": int(len(state.classes)),
        "old_edge_count": old_edges,
        "new_rival_edge_count": new_edges,
        "enabled_edge_count": old_edges + new_edges,
        "resource": dict(state.resource),
        "under_50kib_estimated_state": bool(
            int(state.resource["estimated_serialized_state_bytes"]) < 50 * 1024
        ),
    }


def _measure_pareto(
    state, probe_feature: np.ndarray, support_features: np.ndarray, *, repeats: int = 200
) -> dict[str, Any]:
    support = _normalise(support_features)
    query = _normalise(probe_feature)
    for _ in range(10):
        _score_numpy(probe_feature, state)
        _ = query @ support.T
    start = time.perf_counter()
    for _ in range(repeats):
        _score_numpy(probe_feature, state)
    d17_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        _ = query @ support.T
    qknn_ms = (time.perf_counter() - start) * 1000.0 / repeats
    d17_macs = int(state.resource["head_mac_upper_bound_per_query"])
    qknn_macs = int(state.resource["identity_qknn_mac_per_query"])
    state_bytes = int(state.resource["estimated_serialized_state_bytes"])
    qknn_bytes = int(support_features.nbytes)
    return {
        "benchmark_input": "one_enrollment_support_row_resource_probe_no_query_open",
        "repeats": repeats,
        "sprtdr_head_latency_ms": d17_ms,
        "identity_single_qknn_latency_ms": qknn_ms,
        "latency_delta_percent": 100.0 * (d17_ms / qknn_ms - 1.0),
        "sprtdr_head_upper_bound_macs": d17_macs,
        "identity_single_qknn_exact_macs": qknn_macs,
        "mac_delta_percent": 100.0 * (d17_macs / qknn_macs - 1.0),
        "sprtdr_estimated_serialized_state_bytes": state_bytes,
        "identity_single_qknn_state_bytes": qknn_bytes,
        "state_delta_percent": 100.0 * (state_bytes / qknn_bytes - 1.0),
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "dense_query_graph": False,
        "backbone_forwards_per_physical_sample": 1,
        "fft_branches_per_physical_sample": 0,
    }


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = "development_select",
) -> dict[str, Any]:
    if mode != "development_select":
        raise D17RunnerError("D17 runner is development_select only")
    if (
        len(expected_before_seal_sha256) != 64
        or len(expected_after_seal_sha256) != 64
    ):
        raise D17RunnerError("external expected enrollment seal SHA required")
    if output.exists():
        raise D17RunnerError("output path already exists")

    before_payloads, before_manifest, before_preopen = _load_enrollment(
        before_root,
        before_seal,
        registration_state="before",
        expected_seal_sha256=expected_before_seal_sha256,
    )
    after_payloads, after_manifest, after_preopen = _load_enrollment(
        after_root,
        after_seal,
        registration_state="after",
        expected_seal_sha256=expected_after_seal_sha256,
    )
    _manifest_binding(before_manifest, after_manifest)
    k_shot = int(before_manifest["k_shot"])
    if k_shot != 10 or int(after_manifest["k_shot"]) != 10:
        raise D17RunnerError("development_select requires sealed strict K10")
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root, _member(before_manifest, "feature_runtime"), device=device
    )
    output.mkdir(parents=True)

    module_path = CODE / "cvsrffi" / "stage2_sprtdr.py"
    runner_path = Path(__file__).resolve()
    d14_path = CODE / "scripts" / "run_d14_support_only_pairwise_fisher_guard.py"
    artifact_path = CODE / "cvsrffi" / "stage2_joint_residual_logit_head.py"
    feature_path = CODE / "cvsrffi" / "stage2_diag_cosine_exploration.py"
    code_hashes = {
        "d17_module_sha256": _sha256_file(module_path),
        "d17_runner_sha256": _sha256_file(runner_path),
        "reused_d14_loader_runner_sha256": _sha256_file(d14_path),
        "artifact_provider_sha256": _sha256_file(artifact_path),
        "registered_feature_module_sha256": _sha256_file(feature_path),
    }
    feature_code_sha256 = hashlib.sha256(
        _canonical(
            {
                **code_hashes,
                "operator_id": "base",
                "feature_path": "reused_d14_physical_batch1_fixed_received_iq",
            }
        )
    ).hexdigest()
    runtime_sha256 = str(before_manifest["feature_runtime_sha256"])
    checkpoint_sha256 = str(before_manifest["phase1_checkpoint_sha256"])

    tracemalloc.start()
    run_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    contexts: dict[str, dict[str, Any]] = {}
    after_rows_by_scenario = {}
    exact_k_audit = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _payload_rows(
            before_payloads[scenario], before_manifest, scenario=scenario
        )
        after_rows = _payload_rows(
            after_payloads[scenario], after_manifest, scenario=scenario
        )
        exact_k_audit[scenario] = {
            "before": _validate_exact_k_rows(
                before_rows, k_shot=10, scenario=scenario
            ),
            "after": _validate_exact_k_rows(
                after_rows, k_shot=10, scenario=scenario
            ),
        }
        _old_reuse_lock(before_rows, after_rows)
        after_rows_by_scenario[scenario] = after_rows
        feature_start = time.perf_counter()
        before_artifact = _build_feature_artifact(
            model,
            device,
            before_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
        old_feature_by_token = {
            token: before_artifact.features[index]
            for index, token in enumerate(before_artifact.physical_sample_ids)
        }
        after_artifact = _build_feature_artifact(
            model,
            device,
            after_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            checkpoint_sha256=checkpoint_sha256,
            reuse_by_token=old_feature_by_token,
        )
        contexts[scenario] = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_artifact": before_artifact,
            "after_artifact": after_artifact,
            "feature_seconds": time.perf_counter() - feature_start,
            "before_provenance": _feature_provenance(
                before_rows, before_artifact
            ),
            "after_provenance": _feature_provenance(after_rows, after_artifact),
        }
    disjointness = _cross_scenario_disjointness(after_rows_by_scenario)

    candidates = _candidates(10)
    lock = _candidate_lock(candidates, k_shot=10)
    trace: list[dict[str, Any]] = [
        {
            "phase": "candidate_lock",
            "hyperparameter_lock_sha256": lock["lock_sha256"],
            "candidate_count": len(candidates),
        }
    ]
    evaluations: dict[str, dict[str, Any]] = {}
    candidate_rows = []
    for candidate in candidates:
        evaluations[candidate.candidate_id] = {}
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            context = contexts[scenario]
            evaluation, module_trace = evaluate_joint_leave_two_out(
                context["before_artifact"],
                context["before_rows"]["labels"],
                context["before_rows"]["ranks"],
                context["after_artifact"],
                context["after_rows"]["labels"],
                context["after_rows"]["ranks"],
                hyperparameters=candidate,
            )
            evaluations[candidate.candidate_id][scenario] = evaluation
            trace.extend(
                {
                    "candidate_id": candidate.candidate_id,
                    "scenario": scenario,
                    "hyperparameter_lock_sha256": lock["lock_sha256"],
                    **_jsonable(row),
                }
                for row in module_trace
            )
        summary = _candidate_summary(
            candidate, evaluations[candidate.candidate_id]
        )
        candidate_rows.append(summary)
        trace.append(
            {"phase": "candidate_three_scenario_summary", **_jsonable(summary)}
        )

    selected_id, support_candidate_pass = _select_candidate(
        candidate_rows, candidates
    )
    selected_hp = next(
        value for value in candidates if value.candidate_id == selected_id
    )
    selected_row = next(
        value for value in candidate_rows if value["candidate_id"] == selected_id
    )
    scenario_results = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        context = contexts[scenario]
        fit_start = time.perf_counter()
        fitted = fit_before_after_locked(
            context["before_artifact"],
            context["before_rows"]["labels"],
            context["before_rows"]["ranks"],
            context["after_artifact"],
            context["after_rows"]["labels"],
            context["after_rows"]["ranks"],
            k_shot=10,
            hyperparameters=selected_hp,
        )
        trace.extend(
            {
                "phase": "selected_full_support_fit",
                "candidate_id": selected_id,
                "scenario": scenario,
                **_jsonable(row),
            }
            for row in fitted.trace
        )
        evaluation = evaluations[selected_id][scenario]
        state_serialization = {
            "before": _write_state_roundtrip(
                output / "states" / scenario / "before",
                state=fitted.before_state,
            ),
            "after": _write_state_roundtrip(
                output / "states" / scenario / "after",
                state=fitted.after_state,
            ),
        }
        scenario_results[scenario] = {
            "joint_leave_two_out": evaluation,
            "all_fold_gate": selected_row["scenario_gates"][scenario],
            "floor": selected_row["floors"][scenario],
            "before_state_audit": _state_edge_audit(fitted.before_state),
            "after_state_audit": _state_edge_audit(fitted.after_state),
            "state_serialization": state_serialization,
            "pareto_vs_identity_single_qknn": _measure_pareto(
                fitted.after_state,
                context["after_artifact"].features[:1],
                context["after_artifact"].features,
            ),
            "before_feature_provenance": context["before_provenance"],
            "after_feature_provenance": context["after_provenance"],
            "measured": {
                "feature_extraction_seconds": context["feature_seconds"],
                "selected_full_fit_seconds": time.perf_counter() - fit_start,
            },
        }

    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = (
        "SUPPORT_ONLY_D17_DEVELOPMENT_SELECTED_NO_QUERY_OPEN"
        if support_candidate_pass
        else "SUPPORT_ONLY_D17_DEVELOPMENT_TRUE_Z0_NO_QUERY_OPEN"
    )
    training_log_sha256 = _write_jsonl_new(output / "training_log.jsonl", trace)
    audit = {
        "schema": "cvs.phase2.d17_support_only_audit.v1",
        "status": status,
        "claim_scope": "development_diagnostic_support_only_no_query_claim",
        "authority": "development_diagnostic_only",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "promotion_ready_for_query": False,
        "runner_mode": mode,
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "registration_states": ["before", "after"],
        "view_policy": "one_fixed_received_iq_base_view_no_new_channel_state",
        "exact_k_audit": exact_k_audit,
        "unified_hyperparameter_selection": {
            "selected_candidate_id": selected_id,
            "same_candidate_all_scenarios": True,
            "support_candidate_gate_pass": support_candidate_pass,
            "all_scenario_all_fold_all_class_gate_required": True,
            "old_and_new_floor_strict_gain_every_fold_required": True,
            "true_zero_fallback_policy": (
                "if_any_scene_fold_class_or_floor_gate_fails_select_canonical_true_zero"
            ),
            "candidate_rows": candidate_rows,
        },
        "hyperparameter_lock": lock,
        "scenario_results": scenario_results,
        "selected_state_serialization": {
            "state_count": 2 * len(FORMAL_LEO_WEAK_SCENARIOS),
            "serialized_state_total_bytes_all_scenarios": sum(
                int(block["serialized_state_total_bytes"])
                for value in scenario_results.values()
                for block in value["state_serialization"].values()
            ),
            "all_states_under_50kib": all(
                bool(block["serialized_state_under_50kib"])
                for value in scenario_results.values()
                for block in value["state_serialization"].values()
            ),
            "all_state_sha_and_score_roundtrips_verified": all(
                bool(block["state_sha_roundtrip_verified"])
                and bool(block["fixed_probe_score_bitwise_verified"])
                for value in scenario_results.values()
                for block in value["state_serialization"].values()
            ),
        },
        "cross_scenario_support_disjointness": disjointness,
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
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "runtime_authorization": {
            "feature_extraction_mode": (
                "reused_d14_internal_actual_iq_sha_physical_batch1"
            ),
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
            "combined_feature_code_sha256": feature_code_sha256,
            **code_hashes,
        },
        "training_log_sha256": training_log_sha256,
        "measured_run_resource": {
            "wall_seconds": time.perf_counter() - run_start,
            "peak_python_tracemalloc_bytes": int(peak_python),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            ),
            "device": str(device),
        },
        "before_package_root_sha256": before_manifest["package_root_sha256"],
        "after_package_root_sha256": after_manifest["package_root_sha256"],
        "before_seal_sha256": _sha256_file(before_seal),
        "after_seal_sha256": _sha256_file(after_seal),
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
        "preopen_audit": {"before": before_preopen, "after": after_preopen},
    }
    audit = _jsonable(audit)
    audit_sha256 = _write_json_new(output / "support_audit.json", audit)

    lines = [
        "# D17-SPRTDR strict-K10 enrollment-only开发审计",
        "",
        f"状态：`{status}`。authority固定为`development_diagnostic_only`；只打开sealed before/after support，未开放query、truth或scorer。",
        "",
        f"三场景统一选择`{selected_id}`；candidate lock SHA为`{lock['lock_sha256']}`。",
        "",
        "|场景|Before old/floor|After old/floor|Z0 new/floor|D17 new/floor|joint/H|old决策遗忘|旧score锁|全fold门|",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        value = scenario_results[scenario]
        joint = value["joint_leave_two_out"]
        floor = value["floor"]
        old_floor = floor["old_floor"]["candidate_min_accuracy"]
        new_floor = floor["new_floor"]["candidate_min_accuracy"]
        z0_new_floor = floor["new_floor"]["z0_min_accuracy"]
        lines.append(
            f"|`{scenario}`|"
            f"{joint['before_old']['overall_accuracy']:.4f}/"
            f"{joint['before_old']['min_class_accuracy']:.4f}|"
            f"{joint['after_old']['overall_accuracy']:.4f}/"
            f"{old_floor if old_floor is not None else joint['after_old']['min_class_accuracy']:.4f}|"
            f"{joint['base_after_new']['overall_accuracy']:.4f}/"
            f"{z0_new_floor if z0_new_floor is not None else joint['base_after_new']['min_class_accuracy']:.4f}|"
            f"{joint['after_new']['overall_accuracy']:.4f}/"
            f"{new_floor if new_floor is not None else joint['after_new']['min_class_accuracy']:.4f}|"
            f"{joint['joint']['overall_accuracy']:.4f}/"
            f"{joint['H_old_new']:.4f}|"
            f"{joint['old_forgetting']:.4f}|"
            f"{joint['old_score_bitwise_locked']}|"
            f"{value['all_fold_gate']['all_folds_gate_pass']}|"
        )
    lines.extend(
        [
            "",
            "硬门逐scene×fold×class执行：注册后old决策不得低于同fold注册前old-only决策，old/new均不得低于Z0，joint/H不得下降，并且每fold至少一个old floor和一个new floor分别严格优于Z0。旧score列锁与旧类决策遗忘分开报告。任一门失败即保存canonical true Z0。",
            "",
        ]
    )
    report_sha256 = _write_text_new(output / "report.md", "\n".join(lines))
    receipt = {
        "schema": "cvs.phase2.d17_support_only_receipt.v1",
        "status": status,
        "authority": "development_diagnostic_only",
        "formal_launch_authority": False,
        "promotion_ready_for_query": False,
        "support_audit_sha256": audit_sha256,
        "training_log_sha256": training_log_sha256,
        "report_sha256": report_sha256,
        "query_opened": False,
        "selected_candidate_id": selected_id,
        "support_candidate_gate_pass": support_candidate_pass,
        "hyperparameter_lock_sha256": lock["lock_sha256"],
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
        "sealed_runtime_sha256": runtime_sha256,
        "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
        "combined_feature_code_sha256": feature_code_sha256,
        **code_hashes,
    }
    receipt_sha256 = _write_json_new(output / "RECEIPT.json", receipt)
    return {"receipt_sha256": receipt_sha256, **receipt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mode", choices=("development_select",), required=True
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                before_root=args.before_root,
                before_seal=args.before_seal,
                expected_before_seal_sha256=args.before_seal_sha256,
                after_root=args.after_root,
                after_seal=args.after_seal,
                expected_after_seal_sha256=args.after_seal_sha256,
                output=args.output,
                device_name=args.device,
                mode=args.mode,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
