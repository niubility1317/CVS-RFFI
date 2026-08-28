#!/usr/bin/env python
"""Stage one verified canonical K20 support pool without query artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np

from cvsrffi.leo_weak_cache import load_verified_leo_weak_cache_set
from cvsrffi.stage2_predictor_bundle import SUPPORT_NPZ_MEMBERS, SUPPORT_SCHEMA
from scripts.build_cvs_stage2_support_prototypes import (
    EXPECTED_CAPSULE_ID,
    EXPECTED_K_SHOT,
    EXPECTED_RECEIVER,
    EXPECTED_SCENE,
    EXPECTED_SPLIT_ID,
    _validate_config,
    _validate_row_binding,
)


EXPECTED_PROFILE_ID = "SRC5_MAXP2"
EXPECTED_QUERY_POLICY = "BALANCED_4DAY_CORE"
EXPECTED_REGISTERED_TX_IDS = (
    "14-10",
    "14-7",
    "20-15",
    "20-19",
    "6-15",
    "8-20",
    "11-1",
    "7-11",
    "10-11",
    "10-7",
    "11-4",
    "11-7",
    "15-1",
    "16-16",
    "2-19",
    "20-12",
    "20-7",
    "3-13",
    "5-5",
    "6-1",
    "7-10",
    "8-18",
    "8-3",
    "13-3",
    "4-11",
    "3-18",
)
EXPECTED_CLASS_IDS = tuple(range(len(EXPECTED_REGISTERED_TX_IDS)))
_CACHE_SCOPE = "stage2_canonical_registered"
_ALLOWED_ROLES = {"target_old", "target_new"}


class CanonicalSupportStagingError(ValueError):
    """Raised when the frozen canonical support row drifts."""


def _validate_output_binding(path: Path) -> None:
    required_markers = (EXPECTED_SCENE, f"rx{EXPECTED_RECEIVER}", "k20")
    if any(marker not in path.name for marker in required_markers):
        raise CanonicalSupportStagingError(
            "support pool output is not bound to the frozen scene/receiver/K row"
        )
    if path.suffix.lower() != ".npz":
        raise CanonicalSupportStagingError("support pool output must be NPZ")
    if path.exists():
        raise CanonicalSupportStagingError(
            f"support pool output already exists: {path}"
        )


def _validate_parent_manifest(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    expected = {
        "protocol_schema": "p2_min_v1",
        "profile_id": EXPECTED_PROFILE_ID,
        "query_policy": EXPECTED_QUERY_POLICY,
        "k": EXPECTED_K_SHOT,
        "capsule_id": EXPECTED_CAPSULE_ID,
        "split_id": EXPECTED_SPLIT_ID,
    }
    failed = [field for field, value in expected.items() if manifest.get(field) != value]
    if failed:
        raise CanonicalSupportStagingError(
            f"verified parent cache-set field mismatch: {failed}"
        )
    if audit.get("phase2_single_observation_compliant") is not True:
        raise CanonicalSupportStagingError(
            "verified parent cache-set is not single-observation compliant"
        )


def _ordered_support_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    scene: str,
    receiver: str,
) -> np.ndarray:
    tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
    rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
    scenarios = np.asarray(arrays["sat_scenarios"]).astype(str)
    split_roles = np.asarray(arrays["split_roles"]).astype(str)
    split_ranks = np.asarray(arrays["split_ranks"], dtype=np.int64)
    dataset_roles = np.asarray(arrays["dataset_role"]).astype(str)
    if set(scenarios.tolist()) != {scene}:
        raise CanonicalSupportStagingError("cache scene rows drift from requested scene")
    support_mask = (rx_ids == receiver) & (split_roles == "support")
    selected_tx = set(tx_ids[support_mask].tolist())
    if selected_tx != set(EXPECTED_REGISTERED_TX_IDS):
        raise CanonicalSupportStagingError("support class registry is incomplete")

    ordered: list[int] = []
    for class_id, tx_id in enumerate(EXPECTED_REGISTERED_TX_IDS):
        positions = np.flatnonzero(support_mask & (tx_ids == tx_id)).astype(np.int64)
        expected_role = "target_old" if class_id < 6 else "target_new"
        if positions.size != EXPECTED_K_SHOT:
            raise CanonicalSupportStagingError(
                f"support class {tx_id} is not exact K-shot={EXPECTED_K_SHOT}"
            )
        if set(dataset_roles[positions].tolist()) != {expected_role}:
            raise CanonicalSupportStagingError(
                f"support class {tx_id} dataset role drift"
            )
        ranks = split_ranks[positions]
        if sorted(ranks.tolist()) != list(range(EXPECTED_K_SHOT)):
            raise CanonicalSupportStagingError(
                f"support class {tx_id} rank prefix drift"
            )
        ordered.extend(positions[np.argsort(ranks, kind="stable")].tolist())
    result = np.asarray(ordered, dtype=np.int64)
    if result.size != len(EXPECTED_REGISTERED_TX_IDS) * EXPECTED_K_SHOT:
        raise CanonicalSupportStagingError("support row count drift")
    if not np.all(rx_ids[result] == receiver):
        raise CanonicalSupportStagingError("support receiver drift")
    if not np.all(split_roles[result] == "support"):
        raise CanonicalSupportStagingError("non-support row selected")
    return result


def _write_support_pool_new(
    path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> None:
    canonical_ids = np.asarray(arrays["canonical_physical_sample_ids"]).astype(str)[
        indices
    ]
    if (
        any(not value for value in canonical_ids.tolist())
        or len(set(canonical_ids.tolist())) != len(canonical_ids)
    ):
        raise CanonicalSupportStagingError(
            "selected canonical physical IDs must be nonempty and unique"
        )
    manifest = {
        "schema": SUPPORT_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": EXPECTED_CAPSULE_ID,
        "split_id": EXPECTED_SPLIT_ID,
        "profile_id": EXPECTED_PROFILE_ID,
        "scene": EXPECTED_SCENE,
        "receiver": EXPECTED_RECEIVER,
        "registered_class_count": len(EXPECTED_REGISTERED_TX_IDS),
        "support_pool_max_k": EXPECTED_K_SHOT,
        "registered_tx_ids": list(EXPECTED_REGISTERED_TX_IDS),
        "token_scheme": "canonical_physical_sample_id_v1",
    }
    class_indices = np.repeat(
        np.asarray(EXPECTED_CLASS_IDS, dtype=np.int64), EXPECTED_K_SHOT
    )
    ranks = np.tile(
        np.arange(EXPECTED_K_SHOT, dtype=np.int64),
        len(EXPECTED_REGISTERED_TX_IDS),
    )
    payload = {
        "support_pool_leo_weak_iq": np.ascontiguousarray(
            np.asarray(arrays["leo_weak_iq"], dtype=np.float32)[indices]
        ),
        "support_pool_class_indices": class_indices,
        "support_pool_rank_within_class": ranks,
        "support_pool_tokens": canonical_ids,
        "support_pool_overlay_tokens": np.asarray(arrays["overlay_ids"])[indices],
        "support_pool_satellite_seeds": np.asarray(
            arrays["satellite_seeds"], dtype=np.int64
        )[indices],
        "support_pool_post_channel_iq_sha256": np.asarray(
            arrays["post_channel_iq_sha256"]
        )[indices],
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if tuple(payload) != SUPPORT_NPZ_MEMBERS:
        raise CanonicalSupportStagingError("support pool member order drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **payload)


def stage_support_pool(
    config: Mapping[str, Any],
    *,
    cache_set_path: str | Path,
    output_path: str | Path,
    scene: str,
    receiver: str,
) -> dict[str, Any]:
    """Select the frozen canonical support row and write one support-only NPZ."""

    resolved = _validate_config(config)
    _validate_row_binding(resolved, scene=scene, receiver=receiver)
    destination = Path(output_path)
    _validate_output_binding(destination)
    arrays_by_scenario, manifest, cache_audit = load_verified_leo_weak_cache_set(
        cache_set_path,
        expected_scope=_CACHE_SCOPE,
        allowed_roles=_ALLOWED_ROLES,
    )
    _validate_parent_manifest(manifest, cache_audit)
    arrays = arrays_by_scenario[EXPECTED_SCENE]
    indices = _ordered_support_indices(
        arrays,
        scene=EXPECTED_SCENE,
        receiver=EXPECTED_RECEIVER,
    )
    canonical_ids = np.asarray(arrays["canonical_physical_sample_ids"]).astype(str)[
        indices
    ]
    if len(set(canonical_ids.tolist())) != len(canonical_ids):
        raise CanonicalSupportStagingError(
            "selected canonical physical IDs must be unique"
        )
    _write_support_pool_new(destination, arrays=arrays, indices=indices)
    return {
        "status": "CANONICAL_SUPPORT_ONLY_STAGING_PASS",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": EXPECTED_CAPSULE_ID,
        "split_id": EXPECTED_SPLIT_ID,
        "scene": EXPECTED_SCENE,
        "receiver": EXPECTED_RECEIVER,
        "k_shot": EXPECTED_K_SHOT,
        "registered_tx_ids": list(EXPECTED_REGISTERED_TX_IDS),
        "class_ids": list(EXPECTED_CLASS_IDS),
        "support_rows": int(indices.size),
        "canonical_physical_ids_unique": True,
        "output_members": list(SUPPORT_NPZ_MEMBERS),
        "output_path": str(destination),
        "query_artifact_opened": False,
        "query_truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--receiver", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    audit = stage_support_pool(
        config,
        cache_set_path=args.cache_set,
        output_path=args.output,
        scene=args.scene,
        receiver=args.receiver,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CanonicalSupportStagingError", "main", "stage_support_pool"]
