"""Build an M2.7 Phase32 side cache from one sealed Stage2-C package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_ablation_feature_builder import (
    _load_package,
    _registered_handles,
)
from cvsrffi.stage2_ablation_feature_cache import load_feature_cache
from cvsrffi.stage2_m27_phase_side_cache import (
    phase_coherence32,
    publish_phase_side_cache,
)


class M27PhaseBuilderError(ValueError):
    """Raised when the sealed Stage2-C IQ does not bind the base cache."""


def build_phase_side_cache_from_sealed_stage2c(
    *,
    base_feature_cache_payload: str | Path,
    base_feature_cache_manifest: str | Path,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    after_package_root: str | Path,
    after_seal_path: str | Path,
    after_seal_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Extract deterministic Phase32 views without opening scorer truth."""

    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    support, query, package_manifest, _audit = _load_package(
        after_package_root,
        after_seal_path,
        after_seal_sha256,
    )
    base_manifest = base["manifest"]
    old_classes = tuple(str(item) for item in base["old_classes"])
    new_classes = tuple(str(item) for item in base["new_classes"])
    all_classes = old_classes + new_classes
    package_classes = _registered_handles(package_manifest)
    if (
        package_manifest.get("stage") != "stage2c"
        or str(package_manifest.get("receiver")) != str(base_manifest["receiver"])
        or int(package_manifest.get("new_class_count", -1)) != len(new_classes)
        or str(package_manifest.get("package_root_sha256", "")).lower()
        != str(base_manifest["package_root_sha256"]).lower()
        or str(after_seal_sha256).lower()
        != str(base_manifest["package_seal_sha256"]).lower()
        or package_classes != all_classes
        or base_manifest.get("phase2_data_status") != "VALIDATED_ONCE"
        or int(base_manifest["k_shot"]) not in {1, 2, 5, 10}
        or set(support) != set(FORMAL_LEO_WEAK_SCENARIOS)
        or set(query) != set(FORMAL_LEO_WEAK_SCENARIOS)
    ):
        raise M27PhaseBuilderError("sealed Stage2-C package/base-cache identity drift")

    k_shot = int(base_manifest["k_shot"])
    scenario_payloads: dict[str, dict[str, np.ndarray]] = {}
    support_iq_count = 0
    query_iq_count = 0
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        support_payload = support[scenario]
        ranks = np.asarray(
            support_payload["support_pool_rank_within_class"], dtype=np.int64
        )
        class_indices = np.asarray(
            support_payload["support_pool_class_indices"], dtype=np.int64
        )
        support_iq = np.asarray(
            support_payload["support_pool_leo_weak_iq"], dtype=np.float32
        )
        mask = ranks < k_shot
        selected_indices = class_indices[mask]
        if (
            ranks.ndim != 1
            or ranks.shape != class_indices.shape
            or len(support_iq) != len(ranks)
            or int(np.sum(mask)) != len(all_classes) * k_shot
            or any(
                int(np.sum(selected_indices == index)) != k_shot
                for index in range(len(all_classes))
            )
        ):
            raise M27PhaseBuilderError(f"{scenario} support assignment drift")
        selected_iq = support_iq[mask]
        support_phase = phase_coherence32(selected_iq)
        support_labels = np.asarray(all_classes, dtype=str)[selected_indices]
        old_mask = selected_indices < len(old_classes)

        query_payload = query[scenario]
        query_iq = np.asarray(query_payload["query_leo_weak_iq"], dtype=np.float32)
        query_tokens = np.asarray(query_payload["query_tokens"]).astype(str)
        base_tokens = np.asarray(
            base["scenario_payloads"][scenario]["query_tokens"]
        ).astype(str)
        if (
            query_iq.ndim != 3
            or query_tokens.ndim != 1
            or len(query_iq) != len(query_tokens)
            or len(query_iq) == 0
            or not np.array_equal(query_tokens, base_tokens)
        ):
            raise M27PhaseBuilderError(f"{scenario} query-token/IQ binding drift")
        query_phase = phase_coherence32(query_iq)
        scenario_payloads[scenario] = {
            "old_support_phase32": support_phase[old_mask],
            "old_support_labels": support_labels[old_mask],
            "new_support_phase32": support_phase[~old_mask],
            "new_support_labels": support_labels[~old_mask],
            "query_phase32": query_phase,
            "query_tokens": query_tokens,
        }
        support_iq_count += int(len(selected_iq))
        query_iq_count += int(len(query_iq))

    destination = Path(output_root).absolute()
    receipt = publish_phase_side_cache(
        destination / "phase32.npz",
        destination / "phase32.manifest.json",
        base_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
        capsule_id=str(base_manifest["capsule_id"]),
        split_id=str(base_manifest["split_id"]),
        receiver=str(base_manifest["receiver"]),
        method_seed=int(base_manifest["method_seed"]),
        k_shot=k_shot,
        old_classes=old_classes,
        new_classes=new_classes,
        scenario_payloads=scenario_payloads,
    )
    return {
        **receipt,
        "receiver": str(base_manifest["receiver"]),
        "method_seed": int(base_manifest["method_seed"]),
        "k_shot": k_shot,
        "new_class_count": len(new_classes),
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "support_received_iq_view_count": support_iq_count,
        "query_received_iq_view_count": query_iq_count,
        "query_truth_opened": False,
        "query_role_opened": False,
        "raw_dataset_opened": False,
        "source_or_clean_sample_opened": False,
        "phase2_data_revalidated": False,
    }


__all__ = [
    "M27PhaseBuilderError",
    "build_phase_side_cache_from_sealed_stage2c",
]
