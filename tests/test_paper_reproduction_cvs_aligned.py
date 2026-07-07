import math
import subprocess
import sys
from pathlib import Path

import torch

from paper_reproduction.cvs_aligned.metrics import compute_cvs_stage2_metrics
from paper_reproduction.cvs_aligned.evaluate import (
    _embed,
    _filter_kwargs_for_callable,
    _prototype_predict,
    _scenario_counts_for_steps,
    _support_head_finetune_predict,
)
from paper_reproduction.cvs_aligned.protocol import validate_stage2_protocol_payload


def test_stage2c_metrics_report_old_new_unknown_and_open_set_curves():
    true = torch.tensor([0, 0, 1, 1, 6, 6, 7, 7, -1, -1, -1, -1])
    pred = torch.tensor([0, -1, 1, 6, 6, 0, 7, -1, -1, 6, 0, -1])
    unknown_scores = torch.tensor([0.1, 0.8, 0.2, 0.4, 0.1, 0.5, 0.2, 0.9, 0.95, 0.7, 0.6, 0.99])

    metrics = compute_cvs_stage2_metrics(
        true_labels=true,
        predicted_labels=pred,
        unknown_scores=unknown_scores,
        old_labels={0, 1},
        new_labels={6, 7},
    )

    assert metrics["old_acc"] == 0.5
    assert metrics["target_old_full_acc"] == 0.5
    assert metrics["target_old_accepted_acc"] == 2 / 3
    assert metrics["target_old_coverage"] == 0.75
    assert metrics["seen_new_acc"] == 0.5
    assert metrics["unknown_FAR"] == 0.5
    assert metrics["unknown_to_seen_new_rate"] == 0.25
    assert metrics["unknown_to_old_rate"] == 0.25
    assert metrics["old_to_seen_new_rate"] == 0.25
    assert metrics["seen_new_to_old_rate"] == 0.25
    assert metrics["H_old_new"] == 0.5
    assert 0.0 <= metrics["AUROC"] <= 1.0
    assert 0.0 <= metrics["FPR95"] <= 1.0


def test_stage2b_does_not_emit_seen_new_identity_when_no_new_labels():
    true = torch.tensor([0, 0, 1, 1, -1, -1])
    pred = torch.tensor([0, -1, 1, 0, -1, 1])
    unknown_scores = torch.tensor([0.1, 0.8, 0.2, 0.4, 0.95, 0.3])

    metrics = compute_cvs_stage2_metrics(
        true_labels=true,
        predicted_labels=pred,
        unknown_scores=unknown_scores,
        old_labels={0, 1},
        new_labels=set(),
    )

    assert metrics["stage2_seen_new_identity_evaluated"] is False
    assert "seen_new_acc" not in metrics
    assert "H_old_new" not in metrics
    assert metrics["unknown_FAR"] == 0.5
    assert not math.isnan(metrics["AUROC"])


def test_stage2_protocol_payload_requires_disjoint_receivers_and_tx_sets():
    payload = {
        "stage": "Stage2-C",
        "source_receiver_labels": ["1-1", "1-19"],
        "target_receiver_labels": ["20-1"],
        "target_old_tx_labels": ["14-10", "14-7"],
        "target_new_tx_labels": ["1-16", "1-18"],
        "target_unknown_tx_labels": ["10-1", "10-10"],
        "k_shot": 5,
        "target_channel_view": "satellite/LEO",
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "threshold_scope": "support_only_no_unknown_query",
    }

    checked = validate_stage2_protocol_payload(payload)

    assert checked["stage2_protocol_valid"] is True
    assert checked["stage2_seen_new_identity_allowed"] is True

    bad = dict(payload)
    bad["target_unknown_tx_labels"] = ["1-18"]
    try:
        validate_stage2_protocol_payload(bad)
    except ValueError as exc:
        assert "Y_new and Y_unknown" in str(exc)
    else:
        raise AssertionError("overlapping Y_new/Y_unknown should be rejected")


def test_stage2b_old_only_allows_no_unknown_when_rejection_disabled():
    payload = {
        "stage": "Stage2-B",
        "source_receiver_labels": ["1-1", "1-19"],
        "target_receiver_labels": ["20-1", "3-19"],
        "target_old_tx_labels": ["14-10", "14-7"],
        "target_new_tx_labels": [],
        "target_unknown_tx_labels": [],
        "unknown_rejection_enabled": False,
        "k_shot": 5,
        "target_channel_view": "satellite/LEO",
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "threshold_scope": "support_only_no_unknown_query",
    }

    checked = validate_stage2_protocol_payload(payload)

    assert checked["stage"] == "Stage2-B"
    assert checked["stage2_seen_new_identity_allowed"] is False
    assert checked["unknown_rejection_enabled"] is False


def test_satellite_train_scenario_counts_are_round_robin_and_auditable():
    scenarios = ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]

    assert _scenario_counts_for_steps(scenarios, 200) == {
        "leo_clear_weak": 67,
        "leo_low_elev_weak": 67,
        "leo_rain_weak": 66,
    }
    assert _scenario_counts_for_steps(scenarios, 600) == {
        "leo_clear_weak": 200,
        "leo_low_elev_weak": 200,
        "leo_rain_weak": 200,
    }


def test_stage2_protocol_allows_clean_control_line_without_deployment_primary_claim():
    payload = {
        "stage": "Stage2-C",
        "source_receiver_labels": ["1-1", "1-19"],
        "target_receiver_labels": ["20-1"],
        "target_old_tx_labels": ["14-10", "14-7"],
        "target_new_tx_labels": ["1-16"],
        "target_unknown_tx_labels": ["10-1"],
        "k_shot": 5,
        "target_channel_view": "clean",
        "target_channel_scenarios": ["clean"],
        "threshold_scope": "support_only_no_unknown_query",
    }

    checked = validate_stage2_protocol_payload(payload)

    assert checked["stage2_protocol_valid"] is True
    assert checked["is_clean_control"] is True
    assert checked["is_deployment_primary"] is False


def test_cvs_aligned_cli_dry_run_validates_formal_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        """{
  "baseline": "protonet_cda",
  "stage": "Stage2-C",
  "manysig_pkl": "REMOTE/ManySig.pkl",
  "manytx_pkl": "REMOTE/ManyTx.pkl",
  "source_receiver_labels": ["1-1", "1-19", "14-7", "18-2", "19-2", "2-1", "2-19"],
  "target_receiver_labels": ["20-1"],
  "target_old_tx_labels": ["14-10", "14-7"],
  "target_new_tx_labels": ["1-16", "1-18"],
  "target_unknown_tx_labels": ["10-1", "10-10"],
  "k_shot": 5,
  "query_per_tx": 4,
  "target_channel_view": "satellite/LEO",
  "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
  "threshold_scope": "support_only_no_unknown_query"
}""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_reproduction.cvs_aligned.evaluate",
            "--config",
            str(config),
            "--dry-run",
            "--formal",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"cvs_extension": true' in result.stdout
    assert '"stage": "Stage2-C"' in result.stdout
    assert "unknown_FAR" in result.stdout
    assert "H_old_new" in result.stdout


def test_riei_drift_stage2c_leo_configs_validate_formally():
    root = Path(__file__).resolve().parents[1]
    configs = [
        "paper_reproduction/configs/riei_fd_cvs_stage2c_k5_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/riei_fd_cvs_stage2c_k10_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/drift_cvs_stage2c_k5_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/drift_cvs_stage2c_k10_leo_protonet_cda_n607.json",
    ]

    for config in configs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "paper_reproduction.cvs_aligned.evaluate",
                "--config",
                config,
                "--dry-run",
                "--formal",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert '"cvs_extension": true' in result.stdout
        assert '"stage": "Stage2-C"' in result.stdout
        assert "unknown_FAR" in result.stdout


def test_finaltest_r010_stage2b_old_only_configs_validate_formally():
    root = Path(__file__).resolve().parents[1]
    configs = [
        "paper_reproduction/configs/riei_fd_finaltest_r010_cvs_stage2b_k5_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/riei_fd_finaltest_r010_cvs_stage2b_k10_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/drift_finaltest_r010_cvs_stage2b_k5_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/drift_finaltest_r010_cvs_stage2b_k10_leo_protonet_cda_n607.json",
    ]

    for config in configs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "paper_reproduction.cvs_aligned.evaluate",
                "--config",
                config,
                "--dry-run",
                "--formal",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert '"stage": "Stage2-B"' in result.stdout
        assert '"unknown_rejection_enabled": false' in result.stdout
        assert "seen_new_acc" not in result.stdout
        assert "unknown_FAR" not in result.stdout


def test_current_sat_supervised_stage2b_old_only_configs_validate_formally():
    root = Path(__file__).resolve().parents[1]
    configs = [
        "paper_reproduction/configs/riei_fd_current_sat_supervised_r010_cvs_stage2b_k5_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/riei_fd_current_sat_supervised_r010_cvs_stage2b_k10_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/drift_current_sat_supervised_r010_cvs_stage2b_k5_leo_protonet_cda_n607.json",
        "paper_reproduction/configs/drift_current_sat_supervised_r010_cvs_stage2b_k10_leo_protonet_cda_n607.json",
    ]

    for config in configs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "paper_reproduction.cvs_aligned.evaluate",
                "--config",
                config,
                "--dry-run",
                "--formal",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert '"stage": "Stage2-B"' in result.stdout
        assert '"unknown_rejection_enabled": false' in result.stdout
        assert "seen_new_acc" not in result.stdout
        assert "unknown_FAR" not in result.stdout


def test_riei_and_drift_embedding_routes_use_identity_features():
    class RieiDummy(torch.nn.Module):
        def forward(self, x):
            return {"z_e": torch.ones(x.shape[0], 3), "z_r": torch.zeros(x.shape[0], 2)}

    class DriftDummy(torch.nn.Module):
        def forward(self, x, grl_lambda=1.0):
            del grl_lambda
            return {"z_tx": torch.full((x.shape[0], 4), 2.0), "z_rx": torch.zeros(x.shape[0], 2)}

    x = torch.zeros(2, 2, 256)

    riei_z = _embed(RieiDummy(), "riei_fd", x, torch.device("cpu"))
    drift_z = _embed(DriftDummy(), "drift", x, torch.device("cpu"))

    assert riei_z.shape == torch.Size([2, 3])
    assert torch.allclose(riei_z, torch.ones(2, 3))
    assert drift_z.shape == torch.Size([2, 4])
    assert torch.allclose(drift_z, torch.full((2, 4), 2.0))


def test_filter_kwargs_for_callable_drops_remote_incompatible_dataset_args():
    def old_dataset_ctor(*, out_len, equalized):
        return out_len, equalized

    filtered = _filter_kwargs_for_callable(
        old_dataset_ctor,
        {"out_len": 256, "equalized": 1, "sample_strategy": "front"},
    )

    assert filtered == {"out_len": 256, "equalized": 1}


def test_prototype_predict_supports_euclidean_scores_and_rejection():
    support_z = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    support_y = torch.tensor([0, 0, 1, 1])
    query_z = torch.tensor([[0.05, 0.0], [3.05, 3.0], [10.0, 10.0]])

    pred, scores, info = _prototype_predict(support_z, support_y, query_z, margin=0.1, metric="euclidean")

    assert pred.tolist() == [0, 1, -1]
    assert scores[2] > scores[0]
    assert info["gate_method"] == "prototype_euclidean_support_quantile"
    assert info["unknown_score_kind"] == "min_euclidean_distance"


def test_prototype_predict_can_disable_rejection_for_old_only_domain_adaptation():
    support_z = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    support_y = torch.tensor([0, 0, 1, 1])
    query_z = torch.tensor([[0.05, 0.0], [3.05, 3.0], [10.0, 10.0]])

    pred, scores, info = _prototype_predict(
        support_z,
        support_y,
        query_z,
        margin=0.1,
        metric="euclidean",
        rejection_enabled=False,
    )

    assert pred.tolist() == [0, 1, 1]
    assert scores[2] > scores[0]
    assert info["gate_method"] == "prototype_euclidean_no_rejection"
    assert info["unknown_rejection_enabled"] is False


def test_prototype_predict_normalized_euclidean_removes_feature_norm_bias():
    support_z = torch.tensor([[100.0, 0.0], [120.0, 0.0], [0.0, 1.0], [0.0, 1.2]])
    support_y = torch.tensor([0, 0, 1, 1])
    query_z = torch.tensor([[0.0, 12.0], [9.0, 0.1]])

    raw_pred, _, raw_info = _prototype_predict(
        support_z,
        support_y,
        query_z,
        margin=0.0,
        metric="euclidean",
        rejection_enabled=False,
    )
    norm_pred, norm_scores, norm_info = _prototype_predict(
        support_z,
        support_y,
        query_z,
        margin=0.0,
        metric="normalized_euclidean",
        rejection_enabled=False,
    )

    assert raw_pred.tolist() == [1, 1]
    assert raw_info["gate_method"] == "prototype_euclidean_no_rejection"
    assert norm_pred.tolist() == [1, 0]
    assert norm_scores.shape == torch.Size([2])
    assert norm_info["gate_method"] == "prototype_normalized_euclidean_no_rejection"
    assert norm_info["unknown_score_kind"] == "min_normalized_euclidean_distance"


def test_support_head_finetune_uses_support_labels_only_for_head_adaptation():
    support_z = torch.tensor([[2.0, 0.0], [2.2, 0.1], [0.0, 2.0], [0.1, 2.2]])
    support_y = torch.tensor([0, 0, 1, 1])
    query_z = torch.tensor([[2.1, 0.0], [0.0, 2.1], [-2.0, -2.0]])

    pred, scores, info = _support_head_finetune_predict(
        support_z,
        support_y,
        query_z,
        margin=0.0,
        steps=80,
        lr=0.05,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )

    assert pred[0].item() == 0
    assert pred[1].item() == 1
    assert scores.shape == torch.Size([3])
    assert info["gate_method"] == "support_head_confidence_quantile"
    assert info["support_finetune_steps"] == 80
