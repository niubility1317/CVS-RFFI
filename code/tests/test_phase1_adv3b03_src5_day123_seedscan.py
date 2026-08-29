from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "code"
    / "scripts"
    / "launch_phase1_adv3b03_src5_day123_seed16_e200_20260829.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location("phase1_adv3b03_day123", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value_after(command: list[str], flag: str) -> str:
    position = command.index(flag)
    return command[position + 1]


def test_plan_packs_two_unique_adv3b03_seeds_per_gpu():
    launcher = _load_launcher()

    rows = launcher.build_plan()

    assert len(rows) == 16
    assert [row.seed for row in rows] == list(range(713101, 713117))
    assert len({row.candidate_id for row in rows}) == 16
    assert all("ADV3B03_MU10_ALPHA20_E200" in row.candidate_id for row in rows)
    assert {gpu: sum(row.gpu == gpu for row in rows) for gpu in range(8)} == {
        gpu: 2 for gpu in range(8)
    }


def test_train_command_is_source_only_day123_and_never_mentions_phase2():
    launcher = _load_launcher()
    row = launcher.build_plan()[0]

    command = launcher.build_train_command(
        row,
        root=Path("/srv/cvs"),
        code_root=Path("/srv/release"),
        python=Path("/opt/conda/envs/CVS-RFFI/bin/python"),
        runs_root=Path("/srv/cvs/runs/formal"),
        wisig_pkl=Path("/srv/cvs/Dataset_WigSig/ManySig.pkl"),
        epochs=200,
    )

    assert _value_after(command, "--wisig_train_days") == "1,2,3"
    assert _value_after(command, "--wisig_train_rxs") == "1,3,4,6,8"
    assert _value_after(command, "--wisig_test_days") == ""
    assert _value_after(command, "--wisig_test_rxs") == ""
    assert _value_after(command, "--phase1_source_only_eval") == "true"
    assert _value_after(command, "--phase1_external_final_eval") == "true"
    assert _value_after(command, "--phase1_source_role_protocol") == "l_s_u_s_v_cal_v_select"
    assert _value_after(command, "--labeled_ratio") == "0.07"
    assert _value_after(command, "--unlabeled_ratio") == "0.63"
    assert _value_after(command, "--source_cal_ratio") == "0.15"
    assert _value_after(command, "--source_select_ratio") == "0.15"
    assert "--from_scratch" in command
    assert _value_after(command, "--from_scratch") == "true"
    assert not any("phase2" in token.lower() for token in command)
    assert not any("muse" in token.lower() for token in command)
    assert not any("fasttrust" in token.lower() for token in command)


def test_train_command_freezes_adv3b03_and_b02_concat_satellite_semantics():
    launcher = _load_launcher()
    row = launcher.build_plan()[0]

    command = launcher.build_train_command(
        row,
        root=Path("/srv/cvs"),
        code_root=Path("/srv/release"),
        python=Path("/opt/conda/envs/CVS-RFFI/bin/python"),
        runs_root=Path("/srv/cvs/runs/formal"),
        wisig_pkl=Path("/srv/cvs/Dataset_WigSig/ManySig.pkl"),
        epochs=200,
    )

    assert _value_after(command, "--lambda_proxy_unknown") == "0.0050"
    assert _value_after(command, "--proxy_unknown_core_quantile") == "0.85"
    assert _value_after(command, "--proxy_unknown_accept_quantile") == "0.80"
    assert _value_after(command, "--proxy_unknown_core_accept_weight") == "0.35"
    assert _value_after(command, "--proxy_unknown_vaccept_cvar_alpha") == "0.20"
    assert _value_after(command, "--proxy_unknown_unknown_margin") == "0.10"
    assert "--use_concat_sat_channel_aug" in command
    assert "--concat_sat_ce_only" in command
    assert _value_after(command, "--concat_sat_ce_weight") == "1.0"
    assert _value_after(command, "--sat_cons_start_epoch") == "80"
    assert _value_after(command, "--lambda_sat_cls") == "0.68"
    assert _value_after(command, "--lambda_sat_cons") == "0"
    assert _value_after(command, "--lambda_zid_channel_invariance") == "0"
    assert _value_after(command, "--sat_train_scenarios") == (
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )
    assert _value_after(command, "--sat_view_schedule") == (
        "1@0.30:leo_clear_weak;"
        "41@0.60:leo_low_elev_weak,leo_rain_weak;"
        "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )


def test_eval_command_uses_final_checkpoint_and_all_required_phase1_scenarios():
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    candidate_root = Path("/srv/cvs/runs/formal") / row.candidate_id

    command = launcher.build_eval_command(
        row,
        code_root=Path("/srv/release"),
        python=Path("/opt/conda/envs/CVS-RFFI/bin/python"),
        candidate_root=candidate_root,
        eval_batch_size=512,
    )

    assert _value_after(command, "--ckpt") == str(candidate_root / "final_ssdg.pth")
    assert _value_after(command, "--eval_on") == "source_v_select"
    assert _value_after(command, "--group_loader") == "source_v_select"
    assert _value_after(command, "--scenarios") == (
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )
    assert "--strict_reconstruction" in command


def test_train_command_binds_the_requested_immutable_run_id():
    launcher = _load_launcher()
    row = launcher.build_plan([713101])[0]

    command = launcher.build_train_command(
        row,
        root=Path("/srv/cvs"),
        code_root=Path("/srv/release"),
        python=Path("/opt/conda/envs/CVS-RFFI/bin/python"),
        runs_root=Path("/srv/cvs/runs/smoke-r1"),
        wisig_pkl=Path("/srv/cvs/Dataset_WigSig/ManySig.pkl"),
        epochs=1,
        run_id="phase1_adv3b03_src5_day123_smoke_e1_r1",
    )

    assert _value_after(command, "--run_id") == "phase1_adv3b03_src5_day123_smoke_e1_r1"
    assert _value_after(command, "--epochs") == "1"
    assert _value_after(command, "--label_epochs") == "1"
    assert _value_after(command, "--pseudo_epochs") == "0"


def test_train_command_is_accepted_by_the_real_ssdg_parser():
    launcher = _load_launcher()
    row = launcher.build_plan([713101])[0]
    command = launcher.build_train_command(
        row,
        root=Path("/srv/cvs"),
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        runs_root=Path("/srv/cvs/runs/formal"),
        wisig_pkl=Path("/srv/cvs/Dataset_WigSig/ManySig.pkl"),
        epochs=200,
    )
    train_path = REPO_ROOT / "code" / "SSDG" / "train_ssdg.py"
    spec = importlib.util.spec_from_file_location("phase1_real_train_ssdg", train_path)
    assert spec is not None and spec.loader is not None
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    parsed = train_module.build_arg_parser().parse_args(command[3:])

    assert parsed.wisig_train_days == "1,2,3"
    assert parsed.wisig_train_rxs == "1,3,4,6,8"
    assert parsed.phase1_source_only_eval is True
    assert parsed.phase1_external_final_eval is True
    assert parsed.concat_sat_ce_only is True
    assert parsed.proxy_unknown_unknown_margin == 0.10


def test_non_muse_phase1_can_delegate_only_the_final_strict_evaluation():
    train_path = REPO_ROOT / "code" / "SSDG" / "train_ssdg.py"
    spec = importlib.util.spec_from_file_location("phase1_external_eval_train_ssdg", train_path)
    assert spec is not None and spec.loader is not None
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    args = SimpleNamespace(
        use_muse_ssdg=False,
        muse_external_final_eval=False,
        phase1_external_final_eval=True,
        checkpoint_selection="final_only",
    )

    assert train_module._external_final_eval_requested(args) is True
    result = train_module._run_final_heldout_evaluation(
        args,
        model=None,
        data_ctx=None,
        device=None,
        checkpoint_path="/srv/run/final_ssdg.pth",
    )

    assert result["status"] == "DELEGATED_TO_EXTERNAL_PHASE1_EVAL"
    assert result["checkpoint"] == str(Path("/srv/run/final_ssdg.pth"))


def test_metric_split_requires_strict_reconstruction_and_writes_four_artifacts(tmp_path):
    launcher = _load_launcher()
    rows = []
    for scenario, sat_correct in (
        ("leo_clear_weak", 8),
        ("leo_low_elev_weak", 7),
        ("leo_rain_weak", 6),
    ):
        rows.append(
            {
                "name": "source_v_select_rx1",
                "rx_idx": 1,
                "rx_label": "1-19",
                "days_label": [1, 2, 3],
                "scenario": scenario,
                "clean_acc": 90.0,
                "clean_correct": 9,
                "clean_total": 10,
                "sat_acc": float(sat_correct * 10),
                "sat_correct": sat_correct,
                "sat_total": 10,
            }
        )
    payload = {
        "checkpoint": str(tmp_path / "final_ssdg.pth"),
        "checkpoint_epoch": 200,
        "eval_on": "source_v_select",
        "group_loader": "source_v_select",
        "reconstruction_audit": {
            "strict_requested": True,
            "checkpoint_load_strict": True,
            "fallback_used": False,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "shape_mismatches": 0,
        },
        "rows": rows,
    }
    (tmp_path / "metrics_joint.json").write_text(json.dumps(payload), encoding="utf-8")

    launcher._split_metrics(tmp_path)

    expected = {
        "clean": 90.0,
        "leo_clear_weak": 80.0,
        "leo_low_elev_weak": 70.0,
        "leo_rain_weak": 60.0,
    }
    for scenario, accuracy in expected.items():
        metric_path = tmp_path / f"metrics_{scenario}.json"
        log_path = tmp_path / f"eval_{scenario}.log"
        assert metric_path.is_file() and log_path.is_file()
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        assert metric["checkpoint_epoch"] == 200
        assert metric["aggregate"]["tx_acc"] == accuracy


def test_main_refuses_to_overwrite_an_existing_run_root(tmp_path):
    launcher = _load_launcher()
    root = tmp_path / "root"
    runs_root = root / "runs" / "formal"
    runs_root.mkdir(parents=True)

    status = launcher.main(
        [
            "--root", str(root),
            "--code-root", str(REPO_ROOT),
            "--python", sys.executable,
            "--run-id", launcher.RUN_ID_DEFAULT,
            "--runs-root", str(runs_root),
            "--log-root", str(root / "logs" / "formal"),
        ]
    )

    assert status == 3


def test_formal_main_rejects_a_partial_seed_matrix_or_non_e200(tmp_path):
    launcher = _load_launcher()
    root = tmp_path / "root"

    partial_status = launcher.main(
        [
            "--root", str(root),
            "--code-root", str(REPO_ROOT),
            "--python", sys.executable,
            "--run-id", launcher.RUN_ID_DEFAULT,
            "--seeds", "713101",
            "--epochs", "200",
            "--dry-run",
        ]
    )
    short_status = launcher.main(
        [
            "--root", str(root),
            "--code-root", str(REPO_ROOT),
            "--python", sys.executable,
            "--run-id", launcher.RUN_ID_DEFAULT,
            "--epochs", "1",
            "--dry-run",
        ]
    )

    assert partial_status == 2
    assert short_status == 2


def test_explicit_smoke_mode_allows_one_seed_e1_with_a_distinct_identity(tmp_path, capsys):
    launcher = _load_launcher()
    smoke_run_id = "phase1_adv3b03_src5_day123_smoke_e1_20260830_r1"

    status = launcher.main(
        [
            "--root", str(tmp_path / "root"),
            "--code-root", str(REPO_ROOT),
            "--python", sys.executable,
            "--run-id", smoke_run_id,
            "--seeds", "713101",
            "--epochs", "1",
            "--smoke",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["run_id"] == smoke_run_id
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["candidate_id"].endswith("_SMOKE_E1")
