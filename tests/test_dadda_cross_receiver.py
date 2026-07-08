from __future__ import annotations

import subprocess
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _synthetic_manysig_compact() -> dict:
    rx_labels = ["1-1", "8-8", "19-2", "20-1", "2-1", "7-14", "2-19", "1-19", "14-7", "7-7", "rx10", "rx11"]
    data = []
    for tx_i in range(6):
        tx_rows = []
        for rx_i in range(12):
            rx_rows = []
            for day_i in range(4):
                eq_rows = []
                for _eq_i in range(2):
                    samples = np.zeros((2, 320, 2), dtype=np.float32)
                    samples[:, :, 0] = float(tx_i + 1)
                    samples[:, :, 1] = float(rx_i + day_i)
                    samples[:, 0, 0] = 50.0 + tx_i
                    eq_rows.append(samples)
                rx_rows.append(eq_rows)
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": [f"tx{i}" for i in range(6)],
        "rx_list": rx_labels,
        "capture_date_list": ["d0", "d1", "d2", "d3"],
        "equalized_list": [0, 1],
    }


def test_dadda_model_outputs_paper_named_modules():
    from paper_reproduction.dadda_cross_receiver.model import DADDANet

    model = DADDANet(
        num_classes=6,
        feature_dim=16,
        multiscale_dim=16,
        base_channels=4,
        classifier_hidden1=16,
        classifier_hidden2=8,
    )
    outputs = model(torch.randn(3, 2, 256))

    assert outputs["global_features"].shape == (3, 16)
    assert outputs["local_features"].shape == (3, 16)
    assert outputs["logits"].shape == (3, 6)
    assert model.classify(torch.randn(2, 256, 2)).shape == (2, 6)


def test_dadda_dynamic_objective_combines_ce_mmd_lmmd():
    from paper_reproduction.dadda_cross_receiver.losses import (
        dadda_objective,
        dynamic_adaptive_factor,
        lmmd_loss,
        mmd_loss,
    )

    source_global = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], requires_grad=True)
    target_global = source_global.detach().clone().requires_grad_(True)
    shifted = target_global + 2.0

    assert mmd_loss(source_global, target_global) < mmd_loss(source_global, shifted)

    source_outputs = {
        "global_features": source_global,
        "local_features": torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], requires_grad=True),
        "logits": torch.tensor([[4.0, 0.1], [0.2, 3.0], [2.5, 0.4]], requires_grad=True),
    }
    target_outputs = {
        "global_features": shifted,
        "local_features": torch.tensor([[0.2, 1.1], [1.2, 0.1], [2.3, 1.0]], requires_grad=True),
        "logits": torch.tensor([[3.0, 0.2], [0.5, 2.8], [2.2, 0.3]], requires_grad=True),
    }
    labels = torch.tensor([0, 1, 0])

    local = lmmd_loss(source_outputs["local_features"], target_outputs["local_features"], labels, target_outputs["logits"])
    alpha = dynamic_adaptive_factor(torch.tensor(2.0), torch.tensor(6.0))
    assert torch.isclose(alpha, torch.tensor(0.25), atol=1e-6)
    assert torch.isfinite(local)

    terms = dadda_objective(source_outputs, target_outputs, labels, tradeoff_lambda=0.7)
    expected = F.cross_entropy(source_outputs["logits"], labels) + 0.7 * terms["dynamic_joint"]
    assert expected.requires_grad
    assert torch.allclose(terms["loss"], expected, atol=1e-6)
    assert set(terms) == {"loss", "cross_entropy", "mmd", "lmmd", "lmmd_sum", "alpha", "dynamic_joint"}
    expected_alpha = terms["mmd"] / (terms["mmd"] + terms["lmmd_sum"] + 1e-8)
    assert torch.allclose(terms["alpha"], expected_alpha, atol=1e-6)
    assert 0.0 <= float(terms["alpha"]) <= 1.0


def test_dadda_schedules_match_paper_formula():
    from paper_reproduction.dadda_cross_receiver.train import lambda_schedule, learning_rate_schedule

    assert lambda_schedule(0.0) == 0.0
    assert math.isclose(lambda_schedule(0.5), 2.0 / (1.0 + math.exp(-5.0)) - 1.0, rel_tol=1e-6)
    assert learning_rate_schedule(0.0, base_lr=0.0001) == 0.0001
    assert math.isclose(
        learning_rate_schedule(0.5, base_lr=0.0001),
        0.0001 / ((1.0 + 5.0) ** 0.75),
        rel_tol=1e-9,
    )


def test_lmmd_skips_classes_missing_from_source_or_target_batch():
    from paper_reproduction.dadda_cross_receiver.losses import lmmd_loss

    source_features = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    target_features = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    source_labels = torch.tensor([0, 0])
    target_probs = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    loss = lmmd_loss(source_features, target_features, source_labels, target_probs, num_classes=2)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_dadda_dry_run_declares_closed_set_uda_not_cvs():
    from paper_reproduction.dadda_cross_receiver.train import build_dry_run_payload

    payload = build_dry_run_payload(
        {
            "paper_scope": "paper_faithful",
            "cvs_extension": False,
            "dataset": "WiSig ManySig",
            "total_receivers": 12,
            "capture_days": 4,
            "tx_count": 6,
            "source_target_tasks": ["1-1->8-8"],
            "target_labels_scope": "evaluation_only",
        }
    )

    assert payload["method_id"] == "dadda_cross_receiver"
    assert payload["paper"].startswith("Cross-Receiver Radio Frequency Fingerprint Identification")
    assert payload["cvs_extension"] is False
    assert payload["target_labels_scope"] == "evaluation_only"
    assert payload["paper_task_plan"][0]["compare_method_ids"][-1] == "dadda"
    assert "not target-new enrollment" in payload["claim_blocks"]


def test_dadda_manysig_task_builder_keeps_target_labels_evaluation_only():
    from paper_reproduction.dadda_cross_receiver.data import build_manysig_task_datasets

    built = build_manysig_task_datasets(_synthetic_manysig_compact(), task="1-1->8-8", max_samples_per_combo=1)
    source_x, source_y, _source_d, source_meta = built["source"][0]
    target_x, target_y, _target_d, target_meta = built["target"][0]

    assert source_x.shape == (2, 256)
    assert target_x.shape == (2, 256)
    assert int(source_y) in range(6)
    assert int(target_y) in range(6)
    assert source_meta["rx"] == "1-1"
    assert target_meta["rx"] == "8-8"
    assert built["meta"]["target_label_role"] == "hidden_for_UDA_training_available_for_final_accuracy_only"
    assert built["meta"]["source_tx_ids"] == [0, 1, 2, 3, 4, 5]
    assert built["meta"]["target_tx_ids"] == [0, 1, 2, 3, 4, 5]


def test_dadda_manysig_task_builder_resolves_non_index_receiver_labels():
    from paper_reproduction.dadda_cross_receiver.data import build_manysig_task_datasets

    built = build_manysig_task_datasets(_synthetic_manysig_compact(), task="20-1->2-1", max_samples_per_combo=1)

    assert built["meta"]["source_receiver_labels"] == ["20-1"]
    assert built["meta"]["target_receiver_labels"] == ["2-1"]


def test_dadda_manysig_task_builder_requires_all_six_tx_classes():
    from paper_reproduction.dadda_cross_receiver.data import build_manysig_task_datasets

    compact = _synthetic_manysig_compact()
    for day_i in range(4):
        compact["data"][5][0][day_i][1] = np.zeros((0, 320, 2), dtype=np.float32)

    try:
        build_manysig_task_datasets(compact, task="1-1->8-8", max_samples_per_combo=1)
    except ValueError as exc:
        assert "all six TX classes" in str(exc)
    else:
        raise AssertionError("missing source TX class should fail the closed-set DADDA protocol")


def test_dadda_smoke_runner_trains_source_only_and_dadda_rows(tmp_path):
    from paper_reproduction.dadda_cross_receiver.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["1-1->8-8"],
        methods=["source_only", "proposed"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        seed=5,
        device="cpu",
        model_config={
            "feature_dim": 8,
            "multiscale_dim": 8,
            "base_channels": 2,
            "classifier_hidden1": 8,
            "classifier_hidden2": 4,
        },
    )

    assert result["method_id"] == "dadda_cross_receiver"
    assert result["result_claim_status"] == "smoke_or_formal_metrics_depend_on_dataset"
    assert [row["method"] for row in result["rows"]] == ["source_only", "proposed"]
    assert all(row["target_labels_scope"] == "evaluation_only" for row in result["rows"])
    assert all(0.0 <= row["target_accuracy"] <= 1.0 for row in result["rows"])
    assert "alpha_mean" in result["rows"][1]["history"][0]
    assert (tmp_path / "1-1_to_8-8_source_only.pt").exists()
    assert (tmp_path / "1-1_to_8-8_proposed.pt").exists()


def test_dadda_training_loop_stops_at_shorter_source_target_stream():
    from paper_reproduction.dadda_cross_receiver.model import DADDANet
    from paper_reproduction.dadda_cross_receiver.train import run_dadda_training_loop

    model = DADDANet(
        num_classes=6,
        feature_dim=8,
        multiscale_dim=8,
        base_channels=2,
        classifier_hidden1=8,
        classifier_hidden2=4,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    source_batches = [
        (torch.randn(2, 2, 256), torch.tensor([0, 1])),
        (torch.randn(2, 2, 256), torch.tensor([2, 3])),
        (torch.randn(2, 2, 256), torch.tensor([4, 5])),
    ]
    target_batches = [(torch.randn(2, 2, 256), torch.tensor([0, 1]))]

    result = run_dadda_training_loop(
        model,
        source_batches,
        target_batches,
        optimizer=optimizer,
        epochs=1,
        device="cpu",
    )

    assert result["batches"] == 1
    assert result["history"][0]["batches"] == 1


def test_dadda_cli_requires_formal_for_real_table2(tmp_path):
    output_path = tmp_path / "should_not_exist.json"
    command = [
        sys.executable,
        "-m",
        "paper_reproduction.dadda_cross_receiver.train",
        "--config",
        "paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json",
        "--run-table2",
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)

    assert completed.returncode != 0
    assert "--formal is required" in completed.stderr
    assert not output_path.exists()
