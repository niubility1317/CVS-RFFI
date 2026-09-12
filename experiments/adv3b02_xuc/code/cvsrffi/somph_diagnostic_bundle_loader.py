"""Explicit diagnostic-only SOMP-H package readers.

These readers intentionally have no formal launch or metric-claim authority.
They preserve historical analysis workflows while keeping unsigned IQ/head
access outside the Phase2 runtime code closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi import somph_predictor_bundle as bundle


DIAGNOSTIC_ONLY = True
DIAGNOSTIC_STATUS = "UNVERIFIED_UNDER_CURRENT_PROTOCOL_DIAGNOSTIC_ONLY"


def preflight_somph_predictor_bundle(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, seal, audit, _provenance = bundle._preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    audit = dict(audit)
    audit.update(
        {
            "status": DIAGNOSTIC_STATUS,
            "control_state": DIAGNOSTIC_STATUS,
            "formal_launch_authority": False,
            "formal_metric_claim_allowed": False,
            "diagnostic_only": True,
        }
    )
    return manifest, seal, audit


def load_verified_somph_predictor_bundle(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    manifest, _seal, audit, provenance = bundle._preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    root = Path(package_root)
    by_kind = {item["kind"]: item for item in manifest["members"]}
    prefix = (
        "support:"
        if manifest["profile"] == bundle.ENROLLMENT_ONLY
        else "query:"
    )
    payloads: dict[str, dict[str, np.ndarray]] = {}
    observed_sample_tokens: set[str] = set()
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        arrays, embedded = bundle._materialize_iq(
            root, by_kind[prefix + scenario]
        )
        if manifest["profile"] == bundle.ENROLLMENT_ONLY:
            bundle._validate_support_payload(
                arrays,
                embedded,
                manifest=manifest,
                scenario=scenario,
                provenance=provenance[scenario],
            )
            sample_tokens = set(
                np.asarray(arrays["support_tokens"]).astype(str).tolist()
            )
        else:
            bundle._validate_query_payload(
                arrays,
                embedded,
                manifest=manifest,
                scenario=scenario,
                provenance=provenance[scenario],
            )
            sample_tokens = set(
                np.asarray(arrays["query_tokens"]).astype(str).tolist()
            )
        if observed_sample_tokens & sample_tokens:
            raise bundle.PredictorPackageError(
                "SOMP-H physical sample-token reuse across LEO_weak scenarios"
            )
        observed_sample_tokens.update(sample_tokens)
        payloads[scenario] = arrays
    return payloads, manifest, {
        **audit,
        "status": DIAGNOSTIC_STATUS,
        "control_state": DIAGNOSTIC_STATUS,
        "iq_payload_materialized": True,
        "materialized_scenarios": list(bundle.FORMAL_LEO_WEAK_SCENARIOS),
        "sample_level_overlay_provenance_crosscheck": "PASS",
        "cross_scenario_physical_sample_token_disjointness_check": "PASS",
        "per_scenario_unified_support_pool_check": (
            "PASS"
            if manifest["profile"] == bundle.ENROLLMENT_ONLY
            else "NOT_APPLICABLE"
        ),
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "diagnostic_only": True,
    }


def load_verified_somph_head_capsule(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    manifest, _seal, _audit, _provenance = bundle._preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    if manifest["profile"] != bundle.APPLY_ONLY:
        raise bundle.PredictorPackageError(
            "SOMP-H diagnostic head capsule is reachable only from apply_only"
        )
    by_kind = {item["kind"]: item for item in manifest["members"]}
    arrays, binding, binding_sha256 = bundle._load_head_capsule_member(
        Path(package_root), by_kind["head_capsule"]
    )
    bundle._validate_enrollment_binding(binding, manifest=manifest)
    return arrays, binding, binding_sha256
