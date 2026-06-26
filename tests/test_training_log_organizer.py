import json
from pathlib import Path

from tools.training_log_organizer import (
    _route_from_context,
    parse_baseline_metrics_json,
    parse_federated_jsonl,
    parse_stdout_log,
    summarize_groups,
)


def test_parse_stdout_log_classifies_centralized_bex_run(tmp_path: Path):
    log_path = tmp_path / "BEX02_fishr002_mixed_e170_20260518.log"
    log_path.write_text(
        "\n".join(
            [
                "EXP_ID=BEX02_fishr002_mixed_e170",
                "CONFIG=BEX02_fishr002_mixed_e170",
                "CMD=python3 -u train.py --train_mode centralized --wisig_train_ratio 0.1 --model_variant lite_d --lambda_fishr 0.02 --run_name BEX02_fishr002_mixed_e170",
                "[EPOCH-END] E001/170",
                "[EPOCH-END] E170/170",
                "Training finished. best_test_overall_tx_acc=90.11% at epoch 64 -> best_test_overall_model.pth",
                "Training finished. best_unseen_day_unseen_rx_tx_acc=85.97% at epoch 63 -> best_strict_udu_model.pth",
                "[SAT-TEST] scenario=clear_leo selected=test_unseen_day_unseen_rx overall_tx=45.50% strict_udu=44.25% (100/200)",
            ]
        ),
        encoding="utf-8",
    )

    record = parse_stdout_log(log_path, tmp_path)

    assert record["training_route"] == "centralized"
    assert record["model_key"] == "BEX02_fishr002_mixed_e170"
    assert record["train_ratio"] == "0.1"
    assert record["best_overall_acc"] == "90.11"
    assert record["best_strict_udu_acc"] == "85.97"
    assert record["sat_clear_leo_strict_udu"] == "44.25"
    assert record["last_step"] == "170"


def test_parse_stdout_out_files_as_training_logs(tmp_path: Path):
    log_path = tmp_path / "CEN_A86_c74_highsat_fishr_dsq_r010.out"
    log_path.write_text(
        "\n".join(
            [
                "EXP_ID=CEN_A86_c74_highsat_fishr_dsq_r010",
                "CMD=python3 -u train.py --wisig_train_ratio 0.1 --run_name CEN_A86_c74_highsat_fishr_dsq_r010",
                "[EPOCH-END] E200/200",
                "Training finished. best_unseen_day_unseen_rx_tx_acc=84.25% at epoch 188 -> strict.pth",
            ]
        ),
        encoding="utf-8",
    )

    record = parse_stdout_log(log_path, tmp_path)

    assert record["source_kind"] == "stdout_log"
    assert record["training_route"] == "centralized"
    assert record["best_strict_udu_acc"] == "84.25"


def test_parse_federated_jsonl_uses_all_rounds_for_best_and_latest(tmp_path: Path):
    run_dir = tmp_path / "FSDG50_fedprox_receiver_ra_bex02_baseline_sat"
    run_dir.mkdir()
    jsonl_path = run_dir / "logs.jsonl"
    rows = [
        {
            "round": 1,
            "train_mode": "fedprox",
            "fl_local_objective": "receiver_agnostic_bex02",
            "global_test_overall_acc": 70.0,
            "global_strict_udu_acc": 64.0,
            "global_eval_acc": 80.0,
            "fedprox_mu": 0.01,
        },
        {
            "round": 2,
            "train_mode": "fedprox",
            "fl_local_objective": "receiver_agnostic_bex02",
            "global_test_overall_acc": 72.0,
            "global_strict_udu_acc": 68.5,
            "global_eval_acc": 81.0,
            "fedprox_mu": 0.01,
        },
        {
            "round": 3,
            "train_mode": "fedprox",
            "fl_local_objective": "receiver_agnostic_bex02",
            "global_test_overall_acc": 71.0,
            "global_strict_udu_acc": 67.0,
            "global_eval_acc": 81.5,
            "fedprox_mu": 0.01,
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    record = parse_federated_jsonl(jsonl_path, tmp_path)

    assert record["training_route"] == "federated"
    assert record["model_key"] == "FSDG50_fedprox_receiver_ra_bex02_baseline_sat"
    assert record["objective"] == "receiver_agnostic_bex02"
    assert record["best_strict_udu_acc"] == "68.5"
    assert record["best_strict_udu_step"] == "2"
    assert record["latest_strict_udu_acc"] == "67.0"
    assert record["last_step"] == "3"


def test_parse_baseline_metrics_json_classifies_comparison_model(tmp_path: Path):
    run_dir = tmp_path / "cvs_baseline_queue_20260508_111311" / "cvcnn_seed1337"
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best": {
                    "epoch": 187,
                    "test": {"overall": {"tx_acc": 61.25}},
                    "extra_tests": {
                        "sat_channel": {
                            "clear_leo": {"strict_udu": 19.05},
                        }
                    },
                },
                "final": {
                    "epoch": 200,
                    "test": {
                        "named": {
                            "test_unseen_day_unseen_rx": {"tx_acc": 57.5},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    record = parse_baseline_metrics_json(metrics_path, tmp_path)

    assert record["training_route"] == "comparison_baseline"
    assert record["model_key"] == "cvcnn"
    assert record["best_step"] == "187"
    assert record["best_overall_acc"] == "61.25"
    assert record["latest_strict_udu_acc"] == "57.5"
    assert record["sat_clear_leo_strict_udu"] == "19.05"


def test_scheduler_logs_are_not_folded_into_federated_runs():
    path = Path("logs/fed_fewshot_dg/scheduler_CORE_20260521_145702.log")

    assert _route_from_context(path, {}, "fedavg queued command") == "scheduler"


def test_best_base_followup_stdout_remains_centralized(tmp_path: Path):
    log_path = tmp_path / "best_base_followup" / "BBF10_mixed_e180_20260519_055136.log"
    log_path.parent.mkdir()
    log_path.write_text(
        "\n".join(
            [
                "EXP_ID=BBF10_mixed_e180",
                "CMD=python3 -u train.py --wisig_train_ratio 0.2 --run_name BBF10_mixed_e180 --lambda_fishr 0.02",
                "Training finished. best_test_overall_tx_acc=90.38% at epoch 70 -> best.pth",
                "Training finished. best_unseen_day_unseen_rx_tx_acc=86.80% at epoch 70 -> strict.pth",
            ]
        ),
        encoding="utf-8",
    )

    record = parse_stdout_log(log_path, tmp_path)

    assert record["training_route"] == "centralized"
    assert record["family"] == "BEX"


def test_summarize_groups_collapses_duplicate_sources_by_semantic_key():
    records = [
        {
            "training_route": "centralized",
            "family": "BEX02",
            "model_key": "BEX02_fishr002_mixed_e170",
            "objective": "",
            "train_ratio": "0.2",
            "client_key": "",
            "best_strict_udu_acc": "85.0",
            "best_overall_acc": "90.0",
            "source_path": "logs/a.log",
        },
        {
            "training_route": "centralized",
            "family": "BEX02",
            "model_key": "BEX02_fishr002_mixed_e170",
            "objective": "",
            "train_ratio": "0.2",
            "client_key": "",
            "best_strict_udu_acc": "86.0",
            "best_overall_acc": "91.0",
            "source_path": "server_log_backups/a.log",
        },
    ]

    grouped = summarize_groups(records)

    assert len(grouped) == 1
    assert grouped[0]["source_count"] == "2"
    assert grouped[0]["best_strict_udu_acc"] == "86.0"
    assert grouped[0]["best_source_path"] == "server_log_backups/a.log"


def test_group_leaderboard_falls_back_to_overall_when_strict_is_missing():
    records = [
        {
            "training_route": "comparison_baseline",
            "family": "cvcnn",
            "model_key": "cvcnn_a",
            "best_strict_udu_acc": "",
            "best_overall_acc": "80.0",
            "source_path": "a",
        },
        {
            "training_route": "comparison_baseline",
            "family": "cvcnn",
            "model_key": "cvcnn_b",
            "best_strict_udu_acc": "",
            "best_overall_acc": "87.0",
            "source_path": "b",
        },
    ]

    grouped = summarize_groups(records)

    assert grouped[0]["model_key"] == "cvcnn_b"
