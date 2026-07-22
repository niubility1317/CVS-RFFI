#!/usr/bin/env python3
"""Lock D10 from strict K10 sealed enrollment support only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

_D9_RUNNER_PATH = Path(__file__).with_name(
    "run_d9_support_only_enrollment.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_d10_d9_runner_common", _D9_RUNNER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)

from cvsrffi.somph_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
)
from cvsrffi.stage2_blind_receiver_operator_bank import (  # noqa: E402
    FFT_ENVELOPE_EQ,
    FFT_ENVELOPE_SHRINK,
    FFT_GAIN_MAX,
    FFT_GAIN_MIN,
    OPERATORS,
    WL_IQ_CIRCULARIZE,
    WL_RHO_MAGNITUDE_CAP,
    BlindReceiverOperatorBankState,
    apply_received_iq_operator,
    build_operator_feature_provenance,
    extend_blind_receiver_operator_bank,
    fit_blind_receiver_operator_bank,
    rebuild_locked_blind_receiver_prototypes,
)
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


class D10SupportRunnerError(ValueError):
    """Raised when D10 support-only execution drifts."""


def _extract_operator_features(
    model: torch.nn.Module,
    device: torch.device,
    iq: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {}
    for operator in OPERATORS:
        view = apply_received_iq_operator(iq, operator)
        zid = forward_zid160(model, view, device=device, batch_size=64)
        result[operator] = registered_feature(view, zid)
    return result


def _state_metadata(
    state: BlindReceiverOperatorBankState,
) -> dict[str, Any]:
    class_rows = []
    for class_index, handle in enumerate(state.classes):
        components = []
        for slot in range(state.operator_indices.shape[1]):
            weight = float(state.weights[class_index, slot])
            operator_index = int(state.operator_indices[class_index, slot])
            if weight > 0.0:
                components.append(
                    {
                        "operator_id": OPERATORS[operator_index],
                        "weight": weight,
                        "prototype_sha256": common._array_sha256(
                            state.prototypes[class_index, slot]
                        ),
                    }
                )
        class_rows.append(
            {"class_handle": handle, "components": components}
        )
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "class_rows": class_rows,
        "calibrations": [
            {
                "operator_id": value.operator_id,
                "center": value.center,
                "scale": value.scale,
            }
            for value in state.calibrations
        ],
        "feature_dim": state.feature_dim,
        "used_operators": list(state.used_operators),
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "current_k": state.current_k,
        "selection_lock_k": state.selection_lock_k,
        "selection_lock_sha256": state.selection_lock_sha256,
        "support_lineage": [
            {
                "class_handle": label,
                "physical_sample_id": token,
                "parent_received_iq_sha256": digest,
            }
            for label, token, digest in state.support_lineage
        ],
        "strict_accuracy_gate": dict(state.strict_accuracy_gate),
        "resource": state.resource_audit(),
    }


def _write_state_new(
    output: Path,
    *,
    stem: str,
    state: BlindReceiverOperatorBankState,
) -> dict[str, str]:
    npz_path = output / f"{stem}.npz"
    with npz_path.open("xb") as handle:
        np.savez(
            handle,
            operator_indices=state.operator_indices,
            weights=state.weights,
            prototypes=state.prototypes,
        )
        handle.flush()
    common._readonly(npz_path)
    metadata_sha = common._write_json_new(
        output / f"{stem}.json", _state_metadata(state)
    )
    return {
        "npz_sha256": common._sha256_file(npz_path),
        "metadata_sha256": metadata_sha,
    }


def _selection_rows(
    state: BlindReceiverOperatorBankState,
    *,
    scenario: str,
    registration_state: str,
) -> list[dict[str, Any]]:
    selection = state.support_audit["selection"]
    key = (
        "per_class_selection"
        if registration_state == "before"
        else "per_new_class_selection"
    )
    by_class = {
        row["class_handle"]: row for row in selection.get(key, ())
    }
    rows = []
    for class_index, handle in enumerate(state.classes):
        trace = by_class.get(handle)
        rows.append(
            {
                "scenario": scenario,
                "registration_state": registration_state,
                "class_handle": handle,
                "lifecycle": (
                    "old_locked"
                    if registration_state == "after"
                    and class_index < state.old_class_count
                    else "registered"
                ),
                "components": [
                    {
                        "operator_id": OPERATORS[
                            int(state.operator_indices[class_index, slot])
                        ],
                        "weight": float(
                            state.weights[class_index, slot]
                        ),
                    }
                    for slot in range(state.operator_indices.shape[1])
                    if float(state.weights[class_index, slot]) > 0.0
                ],
                "selected_candidate_id": (
                    None
                    if trace is None
                    else trace["selected_candidate_id"]
                ),
                "candidate_evidence": (
                    [] if trace is None else trace["candidate_evidence"]
                ),
            }
        )
    return rows


def _nested_proof(
    locked: BlindReceiverOperatorBankState,
    rows: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    *,
    scenario: str,
    registration_state: str,
    k: int,
    output: Path,
) -> dict[str, Any]:
    indices = common._nested_indices(rows, k)
    hashes = tuple(rows["hashes"][indices].tolist())
    rebuilt = rebuild_locked_blind_receiver_prototypes(
        locked,
        {
            operator: values[indices]
            for operator, values in features.items()
        },
        build_operator_feature_provenance(hashes, view_seed=0),
        rows["labels"][indices].tolist(),
        physical_sample_ids=rows["tokens"][indices].tolist(),
        parent_received_iq_sha256=hashes,
    )
    state_hashes = _write_state_new(
        output,
        stem=f"state_{scenario}_{registration_state}_k{k}",
        state=rebuilt,
    )
    return {
        "scenario": scenario,
        "registration_state": registration_state,
        "k": k,
        "support_count": len(indices),
        "selection_lock_sha256": rebuilt.selection_lock_sha256,
        "operator_indices_bitwise_locked": (
            rebuilt.operator_indices is locked.operator_indices
        ),
        "weights_bitwise_locked": rebuilt.weights is locked.weights,
        "calibrations_locked": (
            rebuilt.inner.calibrations == locked.inner.calibrations
        ),
        "only_prototypes_rebuilt": True,
        "k10_lineage_prefix_verified": True,
        "state_sha256": state_hashes,
    }


def _old_lineage_reuse(
    before_rows: Mapping[str, np.ndarray],
    after_rows: Mapping[str, np.ndarray],
    before_features: Mapping[str, np.ndarray],
    after_features: dict[str, np.ndarray],
    old_handles: set[str],
) -> None:
    prior = {
        str(token): (
            str(label),
            str(digest),
            np.asarray(iq),
            {
                operator: before_features[operator][index]
                for operator in OPERATORS
            },
        )
        for index, (token, label, digest, iq) in enumerate(
            zip(
                before_rows["tokens"],
                before_rows["labels"],
                before_rows["hashes"],
                before_rows["iq"],
            )
        )
    }
    for index, (token, label, digest, iq) in enumerate(
        zip(
            after_rows["tokens"],
            after_rows["labels"],
            after_rows["hashes"],
            after_rows["iq"],
        )
    ):
        if str(label) not in old_handles:
            continue
        value = prior.get(str(token))
        if (
            value is None
            or value[0] != str(label)
            or value[1] != str(digest)
            or not np.array_equal(value[2], iq)
        ):
            raise D10SupportRunnerError(
                "D10 old support lineage/IQ changed"
            )
        for operator in OPERATORS:
            after_features[operator][index] = value[3][operator]


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
        raise D10SupportRunnerError("output already exists")
    before_payloads, before_manifest, before_preopen = (
        common._load_enrollment(
            before_root, before_seal, registration_state="before"
        )
    )
    after_payloads, after_manifest, after_preopen = common._load_enrollment(
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
        raise D10SupportRunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"])
        for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"])
        for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D10SupportRunnerError("absent-class registration drift")
    device = torch.device(
        "cuda:0" if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root,
        common._member(before_manifest, "feature_runtime"),
        device=device,
    )
    output.mkdir(parents=True)
    results: dict[str, Any] = {}
    state_hashes: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    nested_proofs: list[dict[str, Any]] = []
    prior_tokens: set[str] = set()
    prior_hashes: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = common._payload_rows(
            before_payloads[scenario],
            before_manifest,
            scenario=scenario,
        )
        after_rows = common._payload_rows(
            after_payloads[scenario],
            after_manifest,
            scenario=scenario,
        )
        scenario_tokens = set(before_rows["tokens"]) | set(
            after_rows["tokens"]
        )
        scenario_hashes = set(before_rows["hashes"]) | set(
            after_rows["hashes"]
        )
        if scenario_tokens & prior_tokens or scenario_hashes & prior_hashes:
            raise D10SupportRunnerError(
                "cross-scenario support lineage reuse"
            )
        prior_tokens.update(scenario_tokens)
        prior_hashes.update(scenario_hashes)
        before_features = _extract_operator_features(
            model, device, before_rows["iq"]
        )
        after_features = _extract_operator_features(
            model, device, after_rows["iq"]
        )
        _old_lineage_reuse(
            before_rows,
            after_rows,
            before_features,
            after_features,
            before_handles,
        )
        before_state = fit_blind_receiver_operator_bank(
            before_features,
            build_operator_feature_provenance(
                before_rows["hashes"].tolist(), view_seed=0
            ),
            before_rows["labels"].tolist(),
            physical_sample_ids=before_rows["tokens"].tolist(),
            parent_received_iq_sha256=before_rows["hashes"].tolist(),
            base_resource_audit={
                "persistent_state_bytes": 0,
                "estimated_head_macs_per_query": 0,
            },
            received_iq_length=before_rows["iq"].shape[-1],
        )
        after_state = extend_blind_receiver_operator_bank(
            before_state,
            after_features,
            build_operator_feature_provenance(
                after_rows["hashes"].tolist(), view_seed=0
            ),
            after_rows["labels"].tolist(),
            physical_sample_ids=after_rows["tokens"].tolist(),
            parent_received_iq_sha256=after_rows["hashes"].tolist(),
        )
        for name, state, rows, features in (
            ("before", before_state, before_rows, before_features),
            ("after", after_state, after_rows, after_features),
        ):
            state_hashes[f"{scenario}:{name}:k10"] = _write_state_new(
                output,
                stem=f"state_{scenario}_{name}_k10",
                state=state,
            )
            for k in (1, 5):
                nested_proofs.append(
                    _nested_proof(
                        state,
                        rows,
                        features,
                        scenario=scenario,
                        registration_state=name,
                        k=k,
                        output=output,
                    )
                )
            selection_rows.extend(
                _selection_rows(
                    state,
                    scenario=scenario,
                    registration_state=name,
                )
            )
        results[scenario] = {
            "before": {
                "registered_class_count": before_state.class_count,
                "selection": before_state.support_audit["selection"],
                "strict_accuracy_gate": dict(
                    before_state.strict_accuracy_gate
                ),
                "resource": before_state.resource_audit(),
            },
            "after": {
                "registered_class_count": after_state.class_count,
                "selection": after_state.support_audit["selection"],
                "strict_accuracy_gate": dict(
                    after_state.strict_accuracy_gate
                ),
                "resource": after_state.resource_audit(),
                "old_state_bitwise_locked": True,
                "old_support_lineage_verified": True,
            },
        }
    any_strict_improvement = any(
        bool(results[scenario][state]["strict_accuracy_gate"]["pass"])
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
        for state in ("before", "after")
    )
    audit = {
        "schema": "cvs.phase2.d10_support_only_runner_audit.v1",
        "status": "SUPPORT_ONLY_D10_LOCKED_NO_QUERY_OPEN",
        "claim_scope": (
            "support_selection_and_state_only_no_query_performance_claim"
        ),
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "new_class_count": len(after_handles - before_handles),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "operator_bank": list(OPERATORS),
        "operator_config": {
            "wl_rho_magnitude_cap": WL_RHO_MAGNITUDE_CAP,
            "fft_envelope_shrink": FFT_ENVELOPE_SHRINK,
            "fft_gain_clip": [FFT_GAIN_MIN, FFT_GAIN_MAX],
            "cfo_estimation": False,
            "cfo_frequency_shift": False,
            "cfo_derotation": False,
            "fft_phase_preserved_binwise": True,
        },
        "d9_sparse_selection_engine_reused": True,
        "any_strict_support_accuracy_improvement": (
            any_strict_improvement
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
        "scenario_results": results,
        "per_class_candidate_rows": selection_rows,
        "nested_k_rebuild_proofs": nested_proofs,
        "state_sha256": state_hashes,
        "before_package_root_sha256": before_manifest[
            "package_root_sha256"
        ],
        "after_package_root_sha256": after_manifest[
            "package_root_sha256"
        ],
        "before_seal_sha256": common._sha256_file(before_seal),
        "after_seal_sha256": common._sha256_file(after_seal),
        "preopen_audit": {
            "before": before_preopen,
            "after": after_preopen,
        },
    }
    audit_sha = common._write_json_new(
        output / "support_audit.json", audit
    )
    lines = [
        "# D10 sealed enrollment support-only锁定",
        "",
        "只打开strict K10 before/after enrollment support；未打开query、truth、"
        "prediction、score或scorer。本artifact不包含query性能结论。",
        "",
        "|场景|状态|类数|support overall base→final|support floor base→final|"
        "strict改善门|最终operator|状态bytes|",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for state_name in ("before", "after"):
            row = results[scenario][state_name]
            selection = row["selection"]
            baseline = selection[
                "baseline"
                if state_name == "before"
                else "baseline_new"
            ]
            final = selection[
                "combined_final"
                if state_name == "before"
                else "combined_final_new"
            ]
            resource = row["resource"]
            lines.append(
                f"|`{scenario}`|{state_name}|"
                f"{row['registered_class_count']}|"
                f"{baseline['overall_accuracy']:.4f}→"
                f"{final['overall_accuracy']:.4f}|"
                f"{baseline['min_class_accuracy']:.4f}→"
                f"{final['min_class_accuracy']:.4f}|"
                f"{row['strict_accuracy_gate']['pass']}|"
                f"{','.join(resource['used_operators'])}|"
                f"{resource['combined_persistent_state_bytes']}|"
            )
    lines.extend(
        [
            "",
            "D10复用D9逐类最多2-view稀疏融合与非退化门；若最终总体、floor、"
            "每类准确率未全部非退化，或所有准确率均与base相同，则自动回退base。",
            "K1/K5只按K10有序lineage前缀重建prototype。",
            "",
        ]
    )
    report_sha = common._write_text_new(
        output / "report.md", "\n".join(lines)
    )
    commit = {
        "schema": "cvs.phase2.d10_support_only_commit.v1",
        "status": audit["status"],
        "support_audit_sha256": audit_sha,
        "report_sha256": report_sha,
        "state_sha256": state_hashes,
        "any_strict_support_accuracy_improvement": (
            any_strict_improvement
        ),
        "query_package_opened": False,
        "query_truth_opened": False,
        "scorer_opened": False,
        "independent_performance_claim": False,
    }
    commit_sha = common._write_json_new(
        output / "COMMIT.json", commit
    )
    return {
        "status": commit["status"],
        "output": str(output),
        "commit_sha256": commit_sha,
        "support_audit_sha256": audit_sha,
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "nested_k_proof_count": len(nested_proofs),
        "any_strict_support_accuracy_improvement": (
            any_strict_improvement
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lock D10 from strict K10 sealed enrollment support only"
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
