import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from audit_phase1_ccoi_pa_m21 import (  # noqa: E402
    AGGREGATE_ARTIFACTS,
    build_arg_parser,
    build_not_run_gate_payload,
    run,
    validate_output_root,
)


def _args(tmp_path, output_name="new"):
    return build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / output_name),
            "--checkpoint",
            "base.pt",
            "--sidecar",
            "sidecar.pth",
            "--wisig_pkl",
            "wisig.pkl",
        ]
    )


def test_parser_freezes_source_only_roles_and_two_stage_names(tmp_path):
    args = _args(tmp_path)

    assert args.train_role == "L_s"
    assert args.gate_role == "V_cal"
    assert args.fit_role == "V_select_fit"
    assert args.audit_role == "V_audit_retro"
    assert args.stage_a == "M2.1A_THETA_TRANSFER_AUDIT"
    assert args.stage_b == "M2.1B_TRUTH_BLIND_EXPERT_GATE"
    assert args.target_or_query_access is False


def test_runner_refuses_existing_output_and_target_query_access(tmp_path):
    args = _args(tmp_path, "existing")
    Path(args.output_dir).mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        validate_output_root(args)

    blocked = _args(tmp_path, "blocked")
    blocked.target_or_query_access = True
    with pytest.raises(ValueError, match="target/query"):
        run(blocked)


def test_stage_b_not_run_is_scientific_closure_not_technical_failure():
    payload = build_not_run_gate_payload("A_PARTIAL")

    assert payload["status"] == "NOT_RUN_A_GATE"
    assert payload["stage_a_status"] == "A_PARTIAL"
    assert payload["technical_failure"] is False
    assert payload["target_or_query_access"] is False


def test_synthetic_smoke_closes_exactly_fourteen_aggregate_artifacts(tmp_path):
    args = _args(tmp_path, "smoke")
    args.synthetic_smoke = True

    assert run(args) == 0
    output = Path(args.output_dir)
    assert {path.name for path in output.iterdir()} == set(AGGREGATE_ARTIFACTS)
    assert len(AGGREGATE_ARTIFACTS) == 14
    decision = json.loads((output / "decision_manifest.json").read_text(encoding="utf-8"))
    assert decision["target_or_query_access"] is False
    assert decision["sample_level_state_persisted"] is False
    assert decision["artifact_count"] == 14
