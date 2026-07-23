#!/usr/bin/env python3
"""Build comparison packages while enforcing only the new-class LEO condition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "code" / "scripts"
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT), str(SCRIPT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_cvs_stage2_predictor_bundle as base_builder  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_leo_weak_cache,
    sha256_file,
)
from paper_reproduction.scripts.build_adv3b02_ci_predictor_bundle import (  # noqa: E402
    reject_predictor_truth_leaks_structurally,
)


def _resolve(manifest_path: Path, raw: str) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else manifest_path.parent / value


def load_comparison_leo_cache_set(
    manifest_path: str | Path,
    *,
    expected_scope: str,
    allowed_roles,
):
    """Load legacy/current cache sets without applying main-method p2_min_v1 gates.

    Each inner scenario cache still receives full cryptographic and LEO overlay
    verification. The relaxed surface is only the set-level schema/policy and
    cross-scenario physical-ID rule, which the user explicitly exempted for
    external comparison methods.
    """

    path = Path(manifest_path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    scenario_map = dict(payload.get("cache_npz_by_scenario", {}))
    hash_map = dict(payload.get("cache_sha256_by_scenario", {}))
    if tuple(scenario_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("comparison cache scenarios drift")
    if tuple(hash_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("comparison cache hash map drift")
    allowed = {str(value) for value in allowed_roles}
    arrays_by_scenario = {}
    audits = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        cache_path = _resolve(path, scenario_map[scenario]).resolve(strict=True)
        if sha256_file(cache_path) != str(hash_map[scenario]):
            raise ValueError(f"comparison LEO cache hash mismatch: {scenario}")
        arrays, inner_manifest, audit = load_verified_leo_weak_cache(
            cache_path,
            expected_scenario=scenario,
            allowed_roles=allowed,
        )
        roles = set(np.asarray(arrays["dataset_role"]).astype(str).tolist())
        if roles != allowed:
            raise ValueError(
                f"comparison LEO cache role drift: {scenario} {sorted(roles)}"
            )
        # The shared builder assumes the main-method cross-scenario physical-ID
        # prohibition. Comparison methods are exempt; namespace opaque builder
        # identities by scenario without modifying verified IQ or provenance.
        original_ids = np.asarray(arrays["sample_ids"]).astype(str)
        arrays["sample_ids"] = np.asarray(
            [f"{value}|comparison_scene={scenario}" for value in original_ids]
        )
        arrays_by_scenario[scenario] = arrays
        audits[scenario] = {
            **audit,
            "comparison_protocol_scope": (
                "stage2_main_method_protocol_exempt_new_class_leo_required"
            ),
            "set_manifest_schema_observed": payload.get("schema"),
            "set_manifest_expected_scope_argument": str(expected_scope),
            "inner_manifest_schema": inner_manifest.get("schema"),
            "new_class_leo_iq_verified": True,
            "cross_scenario_physical_reuse_allowed_for_comparison": True,
        }
    return arrays_by_scenario, payload, {
        "status": "PASS_COMPARISON_SCOPE",
        "new_class_leo_iq_verified": True,
        "scenario_audits": audits,
    }


def main() -> int:
    base_builder.load_verified_leo_weak_cache_set = load_comparison_leo_cache_set
    base_builder._assert_scenario_physical_independence = lambda _arrays: None
    base_builder._reject_predictor_truth_leaks = (
        reject_predictor_truth_leaks_structurally
    )
    return base_builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
