"""Truth-last staging from an already VALIDATED_ONCE canonical Phase2 cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class FastTrustStagingError(ValueError):
    """Raised when a frozen receiver/scenario slice cannot be staged exactly."""


def _string_vector(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise FastTrustStagingError(f"cache slice is missing {key}")
    value = np.asarray(arrays[key])
    if value.ndim != 1 or value.dtype.kind == "O":
        raise FastTrustStagingError(f"cache {key} must be a non-object vector")
    return value.astype(str)


def _write_npz_new(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **payload)


def stage_receiver_arrays(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    *,
    receiver: str,
    output_root: str | Path,
    truth_root: str | Path,
    class_names: Sequence[str],
    k_shot: int,
    token_salt: str,
) -> dict[str, Any]:
    """Write truth-free support/query payloads and physically separate truth."""

    names = [str(value) for value in class_names]
    if not names or len(set(names)) != len(names) or any(not value for value in names):
        raise FastTrustStagingError("class_names must be nonempty and unique")
    requested_k = int(k_shot)
    if requested_k < 1:
        raise FastTrustStagingError("k_shot must be positive")
    if not str(receiver).strip() or not str(token_salt):
        raise FastTrustStagingError("receiver and token_salt must be nonempty")
    predictor_root = Path(output_root)
    isolated_truth_root = Path(truth_root)
    if predictor_root.resolve() == isolated_truth_root.resolve():
        raise FastTrustStagingError("predictor and truth roots must be separate")

    scenario_receipts: list[dict[str, Any]] = []
    for scenario, arrays in arrays_by_scenario.items():
        iq = np.asarray(arrays.get("leo_weak_iq"))
        tx_ids = _string_vector(arrays, "tx_ids")
        rx_ids = _string_vector(arrays, "rx_ids")
        split_roles = _string_vector(arrays, "split_roles")
        split_ranks = np.asarray(arrays.get("split_ranks"), dtype=np.int64)
        physical_ids = _string_vector(arrays, "canonical_physical_sample_ids")
        row_count = int(tx_ids.shape[0])
        if (
            iq.ndim != 3
            or iq.shape[0] != row_count
            or iq.shape[1] != 2
            or not np.issubdtype(iq.dtype, np.number)
            or not np.isfinite(iq).all()
            or any(value.shape[0] != row_count for value in (rx_ids, split_roles, split_ranks, physical_ids))
        ):
            raise FastTrustStagingError(f"{scenario} cache arrays are not row-aligned")

        support_indices: list[int] = []
        query_indices: list[int] = []
        support_labels: list[int] = []
        query_labels: list[int] = []
        for class_id, class_name in enumerate(names):
            class_receiver = (tx_ids == class_name) & (rx_ids == str(receiver))
            support = np.flatnonzero(class_receiver & (split_roles == "support"))
            support = support[np.argsort(split_ranks[support], kind="stable")]
            if support.shape[0] != requested_k or split_ranks[support].tolist() != list(range(requested_k)):
                raise FastTrustStagingError(
                    f"{scenario}/{receiver}/{class_name} is not exact K-shot={requested_k}"
                )
            query = np.flatnonzero(class_receiver & (split_roles == "query"))
            query = query[np.argsort(split_ranks[query], kind="stable")]
            if query.shape[0] < 1:
                raise FastTrustStagingError(
                    f"{scenario}/{receiver}/{class_name} has no query rows"
                )
            support_indices.extend(support.tolist())
            query_indices.extend(query.tolist())
            support_labels.extend([class_id] * len(support))
            query_labels.extend([class_id] * len(query))

        support_idx = np.asarray(support_indices, dtype=np.int64)
        query_idx = np.asarray(query_indices, dtype=np.int64)
        support_physical = physical_ids[support_idx].tolist()
        query_physical = physical_ids[query_idx].tolist()
        if (
            len(set(support_physical)) != len(support_physical)
            or len(set(query_physical)) != len(query_physical)
            or set(support_physical) & set(query_physical)
        ):
            raise FastTrustStagingError(
                f"{scenario}/{receiver} support/query physical IDs are not disjoint"
            )
        query_tokens = np.asarray(
            [
                "q_"
                + hashlib.sha256(
                    f"{token_salt}|{scenario}|{receiver}|{physical_id}".encode("utf-8")
                ).hexdigest()
                for physical_id in query_physical
            ]
        )
        if len(set(query_tokens.tolist())) != len(query_tokens):
            raise FastTrustStagingError(f"{scenario}/{receiver} query token collision")

        marker = f"{scenario}_rx{receiver}_k{requested_k}"
        support_path = predictor_root / f"support_{marker}.npz"
        query_path = predictor_root / f"query_{marker}.npz"
        audit_path = predictor_root / f"support_{marker}.audit.json"
        truth_path = isolated_truth_root / f"truth_{marker}.json"
        if any(path.exists() for path in (support_path, query_path, audit_path, truth_path)):
            raise FastTrustStagingError(f"staging output already exists for {marker}")
        _write_npz_new(
            support_path,
            received_iq=np.ascontiguousarray(iq[support_idx], dtype=np.float32),
            support_labels=np.asarray(support_labels, dtype=np.int64),
        )
        _write_npz_new(
            query_path,
            received_iq=np.ascontiguousarray(iq[query_idx], dtype=np.float32),
            query_ids=query_tokens,
        )
        audit = {
            "schema": "cvs.stage2.target_row_export.v1",
            "mode": "support_only_no_query_smoke",
            "k_shot": requested_k,
            "support_input_rows": len(support_indices),
            "support_output_rows": len(support_indices),
            "support_class_count": len(names),
            "support_class_ids": list(range(len(names))),
            "support_per_class_counts": {
                str(class_id): requested_k for class_id in range(len(names))
            },
            "support_selected_ids": support_physical,
            "support_ids_preserved": True,
            "query_input_opened": False,
            "query_input_rows": 0,
            "query_output_rows": 0,
            "query_ids": [],
            "query_ids_preserved": False,
            "query_truth_opened": False,
            "query_role_opened": False,
        }
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        truth_payload = {
            "schema": "cvs.phase2.fasttrust.query_truth.v1",
            "receiver": str(receiver),
            "scenario": str(scenario),
            "rows": [
                {"query_token": token, "true_class_index": int(class_id)}
                for token, class_id in zip(
                    query_tokens.tolist(), query_labels, strict=True
                )
            ],
        }
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(
            json.dumps(truth_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        scenario_receipts.append(
            {
                "scenario": str(scenario),
                "support_rows": len(support_indices),
                "query_rows": len(query_indices),
                "support_path": str(support_path),
                "query_path": str(query_path),
                "support_audit_path": str(audit_path),
                "truth_path": str(truth_path),
            }
        )

    return {
        "status": "STAGED",
        "receiver": str(receiver),
        "k_shot": requested_k,
        "class_count": len(names),
        "support_query_physical_disjoint": True,
        "query_truth_in_predictor": False,
        "scenarios": scenario_receipts,
    }


__all__ = ["FastTrustStagingError", "stage_receiver_arrays"]
