import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cvsrffi.ecrs_config import (add_ecrs_revision_arguments, ecrs_model_config,
                                experiment_matrix, should_validate_source,
                                validate_ecrs_revision_args)

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/launch_phase1_adv3b02_ecrs_v1r_v2.py"
spec = importlib.util.spec_from_file_location("ecrs_launcher", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def parse(*tokens):
    parser = add_ecrs_revision_arguments(argparse.ArgumentParser())
    return validate_ecrs_revision_args(parser.parse_args(tokens))


def test_defaults_preserve_v1_and_revision_raw_ce():
    assert parse().ecrs_raw_ce_weight == .30
    assert not parse().ecrs_source_screen_only
    args = parse("--ecrs_version", "v2")
    assert args.ecrs_raw_ce_weight == 0
    assert args.ecrs_source_screen_only
    assert args.ecrs_resp_ce_enabled
    assert not args.ecrs_cross_rx_enabled
    assert not args.ecrs_u_pair_enabled
    assert args.ecrs_fusion_mode == "off"
    config = ecrs_model_config(args)
    assert config["response_dim"] == 64
    assert "lr_clock" not in config
    assert config["reference_mode"] == "estimated_reference"


@pytest.mark.parametrize("tokens", [
    ("--ecrs_fixed_rho", "nan"), ("--ecrs_fixed_rho", "inf"),
    ("--ecrs_rho_cap", ".26"), ("--ecrs_ema_decay", "1"),
    ("--ecrs_source_val_interval", "0"),
    ("--ecrs_cross_rx_enabled",),
    ("--ecrs_version", "v2", "--ecrs_compute_only"),
    ("--ecrs_version", "v2", "--ecrs_reference_mode", "protocol_reference"),
    ("--ecrs_version", "v2", "--ecrs_update_resp_from_fusion"),
    ("--ecrs_version", "v2", "--ecrs_gate_calibration_enabled"),
])
def test_invalid_configs_fail(tokens):
    with pytest.raises(ValueError):
        parse(*tokens)


def test_exact_validation_schedule():
    got = {e for e in range(1, 201) if should_validate_source(e, 200)}
    expected = set(range(5, 201, 5)) | {1, 39, 40, 41, 79, 80, 81, 89, 90, 91, 199, 200}
    assert got == expected
    assert len(got) == 48
    assert not should_validate_source(0, 200)
    assert not should_validate_source(201, 200)


def test_matrix_deltas_are_independent():
    rows = experiment_matrix()
    def delta(a, b):
        return {k for k in rows[a].keys() | rows[b].keys() if rows[a].get(k) != rows[b].get(k)}
    assert delta("B3", "B4") == {"ecrs_anchor_mode"}
    assert delta("B4", "B5-X") == {"ecrs_cross_rx_enabled"}
    assert delta("B4", "B5-U") == {"ecrs_u_pair_enabled"}
    assert delta("B5-XU", "B6") == {"ecrs_fusion_mode"}
    for row, key in (("S-LR", "ecrs_lr_clock"), ("S-LEO", "ecrs_leo_ce_mode"),
                     ("S-BATCH", "ecrs_sampler_mode")):
        assert delta("B5-XU", row) == {key}
    assert rows["B1"]["ecrs_compute_only"]
    assert not rows["B2-V1"]["ecrs_gate_calibration_enabled"]


def test_dry_run_has_no_outputs_and_no_target_selection(tmp_path):
    root = LAUNCHER.parents[2]
    run_root = tmp_path / "unique"
    result = subprocess.run([sys.executable, str(LAUNCHER), "--run-root", str(run_root),
                             "--rows", "B0,B2-V1,B3,B4,B5-X,B5-U,B5-XU,B6,S-LR,S-LEO,S-BATCH"],
                            check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert not run_root.exists()
    assert plan["selection_domain"] == "source_only"
    assert plan["baseline_route"] == "matched_ECRS_V1_train_py"
    assert not plan["final_target_evaluation"]
    assert plan["protocol"]["labeled"] == 6300
    assert len(plan["source_validation_epochs"]) == 48
    for row in plan["commands"]:
        assert row["config"]["exp_group"] == "s4_stagewise_full_dual"
        assert "--ecrs_source_screen_only" in row["argv"]
        assert row["argv"][0] == sys.executable
        assert row["config"]["ecrs_fusion_mode"] in ("off", "fixed")
    run_root.mkdir()
    with pytest.raises(FileExistsError):
        launcher.build_plan(root, run_root, tmp_path / "ManySig.pkl", ["B0"])


@pytest.mark.parametrize("row", ["B7"])
def test_unavailable_rows_never_alias(row, tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        launcher.build_plan(LAUNCHER.parents[2], tmp_path / "run", tmp_path / "ManySig.pkl", [row])


def test_training_cannot_turn_on_target_selection(tmp_path):
    with pytest.raises(ValueError, match="frozen-checkpoint"):
        launcher.build_plan(LAUNCHER.parents[2], tmp_path / "run", tmp_path / "ManySig.pkl",
                            ["B0"], final_target_eval=True)


def test_bridge_rows_expand_to_distinct_estimators(tmp_path):
    plan = launcher.build_plan(LAUNCHER.parents[2], tmp_path / "run", tmp_path / "ManySig.pkl",
                               ["B3a", "B3b", "B3c"])
    configs = [row["config"] for row in plan["commands"]]
    assert [c["ecrs_estimator_variant"] for c in configs] == [
        "legacy28_old_reference", "legacy28_estimated_reference", "compact8"]
    assert all(c["ecrs_version"] == "v2" and c["ecrs_anchor_mode"] == "real8" for c in configs)
    assert all(c["lambda_ecrs_cycle"] == 0 and c["ecrs_resp_ce_enabled"] for c in configs)
