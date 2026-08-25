from __future__ import annotations

import pytest

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_m29_d92 import IDENTITY_ONLY
from cvsrffi.stage2_m29_row_executor import execute_m29_row
from cvsrffi.stage2_m29_row_executor import _resolved_protocol_schema
from cvsrffi.stage2_ablation_feature_cache import FEATURE_CACHE_MANIFEST_SCHEMA, FEATURE_CACHE_SCHEMA
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT


def test_row_executor_rejects_non_p2_min_v1_before_output(tmp_path) -> None:
    cache = {
        "manifest": {
            "receiver": "3-19",
            "method_seed": 7282101,
            "protocol_schema": "legacy",
            "phase2_data_status": "VALIDATED_ONCE",
        },
        "old_classes": tuple(f"old{i}" for i in range(6)),
        "new_classes": tuple(f"new{i}" for i in range(5)),
        "scenario_payloads": {name: {} for name in FORMAL_LEO_WEAK_SCENARIOS},
    }
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="protocol identity drift"):
        execute_m29_row(
            arm=IDENTITY_ONLY,
            row_id="bad-schema",
            receiver="3-19",
            base_cache=cache,
            output_root=output,
            seed=7282101,
            bundle=None,
            base_cache_bytes=0,
        )
    assert not output.exists()


def test_exact_legacy_feature_cache_v2_contract_resolves_to_p2_min_v1() -> None:
    manifest = {
        "schema": FEATURE_CACHE_MANIFEST_SCHEMA,
        "feature_cache_schema": FEATURE_CACHE_SCHEMA,
        **PHASE2_FULL_CONTRACT,
    }
    assert _resolved_protocol_schema(manifest) == "p2_min_v1"
    changed = dict(manifest)
    changed["phase2_source_sample_access"] = True
    assert _resolved_protocol_schema(changed) == ""
