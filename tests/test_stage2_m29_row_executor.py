from __future__ import annotations

import numpy as np
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


def test_row_executor_consumes_direct_compiler_quantization_audit(tmp_path) -> None:
    rng = np.random.default_rng(2902)
    old_classes = tuple(f"old{i}" for i in range(6))
    new_classes = tuple(f"new{i}" for i in range(5))

    def features(rows: int) -> np.ndarray:
        identity = rng.normal(size=(rows, 160))
        identity /= np.linalg.norm(identity, axis=1, keepdims=True)
        return np.concatenate((identity, rng.normal(size=(rows, 96))), axis=1).astype(np.float32)

    scenario_payload = {
        "old_support_features": features(6),
        "old_support_labels": np.asarray(old_classes),
        "new_support_features": features(5),
        "new_support_labels": np.asarray(new_classes),
        "query_features": features(4),
        "query_tokens": np.asarray([f"q{i}" for i in range(4)]),
    }
    cache = {
        "manifest": {
            "receiver": "3-19",
            "method_seed": 7282101,
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "k_shot": 1,
            "package_root_sha256": "a" * 64,
            "package_seal_sha256": "b" * 64,
        },
        "old_classes": old_classes,
        "new_classes": new_classes,
        "scenario_payloads": {
            name: scenario_payload for name in FORMAL_LEO_WEAK_SCENARIOS
        },
    }
    receipt = execute_m29_row(
        arm=IDENTITY_ONLY,
        row_id="compiler-quantization",
        receiver="3-19",
        base_cache=cache,
        output_root=tmp_path / "row",
        seed=7282101,
        bundle=None,
        base_cache_bytes=123,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["quantization"]["max_logit_abs_error"] >= 0.0
    assert receipt["behavior"]["full_block_weights"] == {
        "full": 1.0,
        "block3": 0.0,
    }
