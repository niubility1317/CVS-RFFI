from __future__ import annotations

import importlib.util
import json
import numpy as np
import pytest

from cvsrffi.stage2_m24_row_executor import (
    M24RowExecutionError,
    d1_overlay_from_base_cache,
    execute_m24_row,
)
from cvsrffi.stage2_m26_spectral_anchor import build_phase1_spectral_anchor
from cvsrffi.stage2_m26_td_src256 import T5, m26_arm_config_hash
from test_stage2_m24_integration import _caches


def test_m26_lifecycle_scripts_are_available() -> None:
    for name in (
        "scripts.build_m26_phase1_spectral_anchor",
        "scripts.run_m26_td_src256_matrix",
        "scripts.score_m26_td_src256_matrix",
        "scripts.summarize_m26_td_src256_matrix",
    ):
        assert importlib.util.find_spec(name) is not None


def test_m26_paired_score_filename_matches_shared_summary_contract() -> None:
    from scripts import score_m26_td_src256_matrix as scorer

    assert scorer.PAIRED_SCORE_FILENAME == "paired_vs_r0.json"


def test_m26_summary_collects_domain_shift_and_application_diagnostics(tmp_path) -> None:
    from scripts import summarize_m26_td_src256_matrix as summarizer

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "scenario_audit": {
                    "leo_clear_weak": {
                        "selected_strength": 0.02,
                        "fallback_reason": "SUPPORT_LOO_ACCEPTED",
                        "domain_state_digest": "a" * 64,
                        "before_registration_fit": {"domain_state_digest": "a" * 64},
                        "query_application": {
                            "query_count": 10,
                            "gated_query_fraction": 0.4,
                            "adjusted_query_count": 4,
                            "max_logit_abs_delta": 0.02,
                        },
                        "domain_state": {
                            "identity_reliability": 0.75,
                            "envelope_reliability": 0.5,
                            "geometry_reliability": 0.25,
                            "identity_shift_norm": 0.1,
                            "envelope_shift_norm": 0.2,
                            "geometry_shift_norm": 0.3,
                            "identity_loo_gain": [0.1, -0.1],
                            "envelope_loo_gain": [0.2, 0.1],
                            "geometry_loo_gain": [-0.2, 0.0],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    matrix = {
        "entries": [
            {
                "arm": "M26-T5-ID-MGD96-TD",
                "receiver": "3-19",
                "method_seed": 7282101,
                "k_shot": 5,
                "new_class_count": 20,
                "receipt_path": str(receipt_path),
            }
        ]
    }
    result = summarizer._build_m26_diagnostics(matrix)
    overall = result["overall"][0]
    assert overall["metrics"]["geometry_shift_norm"]["pooled_query_weighted_mean"] == 0.3
    assert overall["metrics"]["adjusted_query_fraction"]["pooled_query_weighted_mean"] == 0.4
    assert overall["metrics"]["before_after_domain_digest_equal"]["pooled_query_weighted_mean"] == 1.0
    assert result["fallback_reason_counts"][0]["scenario_fit_count"] == 1


def test_m26_runner_freezes_screen_and_full125_matrix_sizes() -> None:
    from scripts import run_m26_td_src256_matrix as runner

    screen = runner.matrix_spec("screen")
    full = runner.matrix_spec("full125")
    assert runner.EVIDENCE_ARMS == (runner.D1, *runner.M26_ARMS)
    assert screen["paired_input_identity_count"] == 4
    assert screen["expected_method_rows"] == 24
    assert full["paired_input_identity_count"] == 125
    assert full["expected_method_rows"] == 750
    assert len(full["receivers"]) == 5
    assert len(full["seeds"]) == 5
    assert len(full["conditions"]) == 5


def test_m26_t5_row_is_truth_unopened_and_uses_checkpoint_bound_anchor(tmp_path) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {"package_root_sha256": "3" * 64, "package_seal_sha256": "4" * 64}
    )
    compact = d1_overlay_from_base_cache(base)
    first = next(iter(compact["scenario_payloads"].values()))
    old_classes = tuple(str(item) for item in base["old_classes"])
    source = np.asarray(first["old_support_blocks"], dtype=np.float32)[:, :256]
    labels = np.asarray(first["old_support_labels"]).astype(str)
    anchor, _audit = build_phase1_spectral_anchor(
        source,
        labels,
        class_registry=old_classes,
        checkpoint_sha256="d" * 64,
        dataset_roles=np.asarray(["source"] * len(source)),
    )
    receipt = execute_m24_row(
        arm=T5,
        row_id="synthetic_m26_t5_k2",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        output_root=tmp_path / "t5",
        seed=7282101,
        source_anchor=anchor,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["query_truth_opened"] is False
    assert receipt["fit_query_rows_used"] == 0
    assert receipt["arm"] == T5
    assert receipt["source_anchor"]["checkpoint_sha256"] == "d" * 64
    assert receipt["source_anchor"]["component_id"] == anchor.component_id
    assert receipt["candidate_lock_sha256"] == m26_arm_config_hash(
        T5, anchor.component_id
    )
    assert all(
        audit["domain_state"]["query_rows_used"] == 0
        and audit["rf32_consumed"] is False
        and audit["query_application"]["query_count"] > 0
        and 0.0 <= audit["query_application"]["gated_query_fraction"] <= 1.0
        and audit["query_application"]["max_logit_abs_delta"]
        <= audit["selected_strength"] + 1.0e-6
        for audit in receipt["scenario_audit"].values()
    )

    wrong_anchor, _audit = build_phase1_spectral_anchor(
        source,
        labels,
        class_registry=tuple(reversed(old_classes)),
        checkpoint_sha256="d" * 64,
        dataset_roles=np.asarray(["source"] * len(source)),
    )
    rejected_output = tmp_path / "wrong_anchor"
    with pytest.raises(M24RowExecutionError, match="source anchor"):
        execute_m24_row(
            arm=T5,
            row_id="synthetic_m26_wrong_anchor",
            receiver="3-19",
            base_cache=base,
            overlay_cache=compact,
            output_root=rejected_output,
            seed=7282101,
            source_anchor=wrong_anchor,
        )
    assert not rejected_output.exists()
