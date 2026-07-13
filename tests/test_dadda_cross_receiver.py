from __future__ import annotations

import subprocess
import sys
import math
from types import SimpleNamespace
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
    from paper_reproduction.DADDA.model import DADDANet

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


def test_dadda_default_model_locks_paper_widths_and_multiscale_branches():
    from paper_reproduction.DADDA.model import DADDANet

    model = DADDANet(num_classes=6)
    outputs = model(torch.randn(2, 2, 256))

    assert outputs["global_features"].shape == (2, 128)
    assert outputs["local_features"].shape == (2, 128)
    assert model.classifier.net[0].out_features == 512
    assert model.classifier.net[2].out_features == 128
    assert model.multiscale_extractor.branch1[0].kernel_size == (1,)
    assert model.multiscale_extractor.branch2[2].kernel_size == (3,)
    assert model.multiscale_extractor.branch3[2].kernel_size == (5,)
    assert model.multiscale_extractor.branch4[0].__class__.__name__ == "AvgPool1d"


def test_dadda_conv2d_paper_model_uses_fig4_kernel_shapes():
    from paper_reproduction.DADDA.model import DADDANet

    model = DADDANet(
        num_classes=6,
        feature_dim=16,
        multiscale_dim=16,
        base_channels=4,
        classifier_hidden1=16,
        classifier_hidden2=8,
        model_variant="conv2d_paper",
    )
    outputs = model(torch.randn(3, 2, 256))

    assert outputs["global_features"].shape == (3, 16)
    assert outputs["local_features"].shape == (3, 16)
    assert outputs["logits"].shape == (3, 6)
    assert model.multiscale_extractor.branch1[0].kernel_size == (2, 1)
    assert model.multiscale_extractor.branch2[2].kernel_size == (1, 3)
    assert model.multiscale_extractor.branch3[2].kernel_size == (1, 5)
    assert model.multiscale_extractor.branch4[0].__class__.__name__ == "AvgPool2d"


def test_dadda_dynamic_objective_combines_ce_mmd_lmmd():
    from paper_reproduction.DADDA.losses import (
        dadda_objective,
        dynamic_adaptive_factor,
        lmmd_loss,
        mmd_loss,
    )

    source_global = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], requires_grad=True)
    target_global = source_global.detach().clone().requires_grad_(True)
    shifted = target_global + 2.0

    assert mmd_loss(source_global, target_global) < mmd_loss(source_global, shifted)

    source_local = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], requires_grad=True)
    target_local = torch.tensor([[0.2, 1.1], [1.2, 0.1], [2.3, 1.0]], requires_grad=True)
    source_outputs = {
        "global_features": source_global,
        "local_features": source_local,
        "logits": torch.tensor([[4.0, 0.1], [0.2, 3.0], [2.5, 0.4]], requires_grad=True),
    }
    target_outputs = {
        "global_features": shifted,
        "local_features": target_local,
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
    explicit_dynamic = (1.0 - terms["alpha"]) * terms["mmd"] + terms["alpha"] * terms["lmmd_sum"]
    assert torch.allclose(terms["dynamic_joint"], explicit_dynamic, atol=1e-6)
    assert 0.0 <= float(terms["alpha"]) <= 1.0
    terms["loss"].backward()
    assert float(source_global.grad.abs().sum()) > 0.0
    assert float(target_global.grad.abs().sum()) > 0.0
    assert float(source_local.grad.abs().sum()) > 0.0
    assert float(target_local.grad.abs().sum()) > 0.0


def test_dadda_mmd_uses_one_shared_batch_bandwidth():
    from paper_reproduction.DADDA.losses import estimate_rbf_bandwidth, mmd_loss, rbf_kernel

    source = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, 0.0], [2.5, 2.0], [4.0, 1.0]], dtype=torch.float32)
    bandwidth = estimate_rbf_bandwidth(source, target)
    expected = (
        rbf_kernel(source, source, bandwidth=bandwidth).mean()
        + rbf_kernel(target, target, bandwidth=bandwidth).mean()
        - 2.0 * rbf_kernel(source, target, bandwidth=bandwidth).mean()
    ).clamp_min(0.0)

    assert torch.allclose(mmd_loss(source, target), expected)


def test_dadda_objective_supports_fixed_alpha_ablation():
    from paper_reproduction.DADDA.losses import dadda_objective

    source_outputs = {
        "global_features": torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True),
        "local_features": torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True),
        "logits": torch.tensor([[4.0, 0.1], [0.2, 3.0]], requires_grad=True),
    }
    target_outputs = {
        "global_features": torch.tensor([[0.2, 0.0], [1.2, 1.0]], requires_grad=True),
        "local_features": torch.tensor([[0.2, 1.1], [1.2, 0.1]], requires_grad=True),
        "logits": torch.tensor([[3.0, 0.2], [0.5, 2.8]], requires_grad=True),
    }
    terms = dadda_objective(
        source_outputs,
        target_outputs,
        torch.tensor([0, 1]),
        tradeoff_lambda=1.0,
        alpha_mode="fixed",
        fixed_alpha=0.5,
    )

    assert torch.isclose(terms["alpha"], torch.tensor(0.5), atol=1e-6)
    expected_dynamic = 0.5 * terms["mmd"] + 0.5 * terms["lmmd_sum"]
    assert torch.allclose(terms["dynamic_joint"], expected_dynamic, atol=1e-6)


def test_dadda_objective_can_treat_dynamic_alpha_as_batch_weight():
    from paper_reproduction.DADDA.losses import dadda_objective

    source_outputs = {
        "global_features": torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True),
        "local_features": torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True),
        "logits": torch.tensor([[4.0, 0.1], [0.2, 3.0]], requires_grad=True),
    }
    target_outputs = {
        "global_features": torch.tensor([[0.3, 0.0], [1.4, 1.0]], requires_grad=True),
        "local_features": torch.tensor([[0.2, 1.1], [1.2, 0.1]], requires_grad=True),
        "logits": torch.tensor([[3.0, 0.2], [0.5, 2.8]], requires_grad=True),
    }
    terms = dadda_objective(
        source_outputs,
        target_outputs,
        torch.tensor([0, 1]),
        tradeoff_lambda=1.0,
        detach_dynamic_alpha=True,
    )

    assert 0.0 <= float(terms["alpha"]) <= 1.0
    assert terms["loss"].requires_grad
    terms["loss"].backward()
    assert float(source_outputs["global_features"].grad.abs().sum()) > 0.0
    assert float(target_outputs["local_features"].grad.abs().sum()) > 0.0


def test_dadda_schedules_match_paper_formula():
    from paper_reproduction.DADDA.train import lambda_schedule, learning_rate_schedule

    assert lambda_schedule(0.0) == 0.0
    assert math.isclose(lambda_schedule(0.5), 2.0 / (1.0 + math.exp(-5.0)) - 1.0, rel_tol=1e-6)
    assert learning_rate_schedule(0.0, base_lr=0.0001) == 0.0001
    assert math.isclose(
        learning_rate_schedule(0.5, base_lr=0.0001),
        0.0001 / ((1.0 + 5.0) ** 0.75),
        rel_tol=1e-9,
    )


def test_lmmd_skips_classes_missing_from_source_or_target_batch():
    from paper_reproduction.DADDA.losses import lmmd_loss

    source_features = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    target_features = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    source_labels = torch.tensor([0, 0])
    target_probs = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    loss = lmmd_loss(source_features, target_features, source_labels, target_probs, num_classes=2)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_lmmd_sum_and_mean_reductions_use_classwise_terms():
    from paper_reproduction.DADDA.losses import lmmd_loss

    source_features = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    target_features = torch.tensor([[0.5, 0.0], [1.5, 0.0]])
    source_labels = torch.tensor([0, 1])
    target_probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    mean_loss = lmmd_loss(
        source_features,
        target_features,
        source_labels,
        target_probs,
        num_classes=2,
        target_is_probabilities=True,
    )
    sum_loss = lmmd_loss(
        source_features,
        target_features,
        source_labels,
        target_probs,
        num_classes=2,
        reduction="sum",
        target_is_probabilities=True,
    )

    assert torch.isfinite(mean_loss)
    assert torch.allclose(sum_loss, mean_loss * 2.0, atol=1e-6)


def test_dadda_dry_run_declares_closed_set_uda_not_cvs():
    from paper_reproduction.DADDA.train import build_dry_run_payload

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
    assert "2-D paper-shaped" in payload["algorithm"]
    assert payload["cvs_extension"] is False
    assert payload["target_labels_scope"] == "evaluation_only"
    assert payload["paper_task_plan"][0]["compare_method_ids"][-1] == "dadda"
    assert "not target-new enrollment" in payload["claim_blocks"]
    assert "Table V" in payload["paper_evidence_targets"]
    assert "Table VI" in payload["paper_evidence_targets"]
    assert payload["pending_paper_artifacts"][0]["paper_item"] == "Fig.5"


def test_dadda_paper_config_keeps_energy_normalization_enabled():
    import json

    config_path = Path(__file__).resolve().parents[1] / "paper_reproduction" / "configs" / "dadda_cross_receiver_manysig_paper_faithful.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["normalize"] is True
    assert config["paper_domain_sample_count"] is None
    assert "6 x 4000" in config["paper_sample_interpretation"]
    assert config.get("allow_unnormalized_ablation") is None


def test_dadda_paper_config_rejects_unnormalized_formal_runs():
    from paper_reproduction.DADDA.train import validate_formal_or_smoke_settings, validate_paper_faithful_config

    config = {
        "cvs_extension": False,
        "target_labels_scope": "evaluation_only",
        "epochs": 100,
        "normalize": False,
    }

    try:
        validate_paper_faithful_config(config)
    except ValueError as exc:
        assert "normalization" in str(exc)
    else:
        raise AssertionError("paper-faithful DADDA config must reject normalize=false")

    validate_paper_faithful_config({**config, "allow_unnormalized_ablation": True})

    try:
        validate_formal_or_smoke_settings(
            config={**config, "allow_unnormalized_ablation": False},
            settings={"epochs": 100, "normalize": False},
            smoke=False,
            max_samples_per_combo=None,
            max_batches_per_epoch=None,
        )
    except SystemExit as exc:
        assert "--no-normalize is not paper-faithful" in str(exc)
    else:
        raise AssertionError("formal DADDA run must reject --no-normalize without ablation flag")


def test_dadda_paper_artifact_plan_covers_pending_figures_and_tables():
    from paper_reproduction.DADDA.experiment_plans import build_paper_artifact_plan

    plan = build_paper_artifact_plan()

    assert plan["formal_result"] is False
    assert plan["table2_methods"]["missing"] == ["dann", "dan", "dsan", "wd", "dcoral", "cdan"]
    snr_plan = plan["pending_paper_artifacts"][0]
    assert snr_plan["paper_item"] == "Fig.5"
    assert len(snr_plan["tasks"]) == 2
    assert snr_plan["snr_db"] == [0, 5, 10, 15, 20]
    table3 = plan["pending_paper_artifacts"][1]
    assert table3["paper_item"] == "Table III"
    assert len(table3["variants"]) == 8
    table4 = plan["pending_paper_artifacts"][2]
    assert [item["variant"] for item in table4["variants"]] == ["fixed_0p5", "dynamic"]
    analysis_items = {item["paper_item"] for item in plan["pending_paper_artifacts"][3]["artifacts"]}
    assert {"Fig.6", "Fig.7", "Fig.8", "Table V", "Table VI"} <= analysis_items


def test_dadda_manysig_task_builder_keeps_target_labels_evaluation_only():
    from paper_reproduction.DADDA.data import build_manysig_task_datasets

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


def test_dadda_task_builder_can_match_paper_domain_sample_count_and_preprocessing():
    from paper_reproduction.DADDA.data import build_manysig_task_datasets

    built = build_manysig_task_datasets(
        _synthetic_manysig_compact(),
        task="1-1->8-8",
        paper_domain_sample_count=24,
        normalize=False,
        crop_mode="center",
    )

    assert len(built["source"]) == 24
    assert len(built["target"]) == 24
    assert built["meta"]["source_domain_sample_cap"]["applied"] is True
    assert built["meta"]["source_domain_sample_cap"]["groups"] == 24
    assert built["meta"]["source_domain_sample_cap"]["base_per_group"] == 1
    assert built["meta"]["preprocessing"]["normalize"] is False
    assert built["meta"]["preprocessing"]["crop_mode"] == "center"


def test_dadda_target_train_loader_does_not_expose_target_labels():
    from paper_reproduction.DADDA.data import build_manysig_task_loaders

    loaders = build_manysig_task_loaders(_synthetic_manysig_compact(), task="1-1->8-8", batch_size=4, max_samples_per_combo=1)
    target_batch = next(iter(loaders["target_train"]))
    eval_batch = next(iter(loaders["target_eval"]))

    assert set(target_batch) == {"iq"}
    assert "label" not in target_batch
    assert len(eval_batch) == 4


def test_dadda_manysig_task_builder_resolves_non_index_receiver_labels():
    from paper_reproduction.DADDA.data import build_manysig_task_datasets

    built = build_manysig_task_datasets(_synthetic_manysig_compact(), task="20-1->2-1", max_samples_per_combo=1)

    assert built["meta"]["source_receiver_labels"] == ["20-1"]
    assert built["meta"]["target_receiver_labels"] == ["2-1"]


def test_dadda_manysig_task_builder_requires_all_six_tx_classes():
    from paper_reproduction.DADDA.data import build_manysig_task_datasets

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
    from paper_reproduction.DADDA.train import run_table2_reproduction

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
    assert result["result_claim_status"] == "smoke_only_not_paper_formal"
    assert [row["method"] for row in result["rows"]] == ["source_only", "proposed"]
    assert all(row["target_labels_scope"] == "evaluation_only" for row in result["rows"])
    assert all(0.0 <= row["target_accuracy"] <= 1.0 for row in result["rows"])
    assert result["expected_table2_tasks"] == 12
    assert result["completed_task_count"] == 1
    assert result["missing_task_ids"]
    assert result["paper_scope"] == "paper_faithful_closed_set_single_source_UDA"
    assert result["cvs_extension"] is False
    assert result["not_cvs_stage2"] is True
    assert "alpha_mean" in result["rows"][1]["history"][0]
    assert "checkpoint_sha256" in result["rows"][1]
    assert (tmp_path / "1-1_to_8-8_source_only.pt").exists()
    assert (tmp_path / "1-1_to_8-8_proposed.pt").exists()


def test_dadda_smoke_runner_trains_literal_dadda_method(tmp_path):
    from paper_reproduction.DADDA.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["1-1->8-8"],
        methods=["dadda"],
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

    assert result["rows"][0]["method"] == "dadda"
    assert result["result_claim_status"] == "smoke_only_not_paper_formal"
    assert result["rows"][0]["result_claim_status"] == "smoke_only_not_paper_formal"
    assert (tmp_path / "1-1_to_8-8_dadda.pt").exists()


def test_dadda_runner_forwards_detach_target_probabilities(monkeypatch, tmp_path):
    from paper_reproduction.DADDA import train as train_module

    observed = []

    def fake_loop(*args, **kwargs):
        observed.append(
            (
                kwargs["detach_target_probabilities"],
                kwargs["alpha_mode"],
                kwargs["fixed_alpha"],
                kwargs["detach_dynamic_alpha"],
            )
        )
        return {"history": [{"epoch": 1, "batches": 1}]}

    monkeypatch.setattr(train_module, "run_dadda_training_loop", fake_loop)
    result = train_module.run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["1-1->8-8"],
        methods=["dadda"],
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
        detach_target_probabilities=True,
        alpha_mode="fixed",
        fixed_alpha=0.5,
        detach_dynamic_alpha=True,
    )

    assert observed == [(True, "fixed", 0.5, True)]
    assert result["detach_target_probabilities"] is True
    assert result["alpha_mode"] == "fixed"
    assert result["fixed_alpha"] == 0.5
    assert result["detach_dynamic_alpha"] is True


def test_dadda_table2_resets_seed_for_each_task_method(monkeypatch, tmp_path):
    from paper_reproduction.DADDA import train as train_module

    seed_calls = []

    def fake_set_seed(value):
        seed_calls.append(int(value))

    def fake_loop(*args, **kwargs):
        return {"history": [{"epoch": 1, "batches": 1}]}

    monkeypatch.setattr(train_module, "set_seed", fake_set_seed)
    monkeypatch.setattr(train_module, "run_dadda_training_loop", fake_loop)

    result = train_module.run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["1-1->8-8", "8-8->1-1"],
        methods=["dadda"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        seed=7,
        device="cpu",
        model_config={
            "feature_dim": 8,
            "multiscale_dim": 8,
            "base_channels": 2,
            "classifier_hidden1": 8,
            "classifier_hidden2": 4,
        },
    )

    assert seed_calls == [7, 7, 7]
    assert [row["task_seed"] for row in result["rows"]] == [7, 7]
    assert "independent of lane order" in result["seed_policy"]


def test_dadda_missing_paper_baselines_are_structured_rows(tmp_path):
    from paper_reproduction.DADDA.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["1-1->8-8"],
        methods=["dan", "cdan"],
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

    assert [row["status"] for row in result["rows"]] == ["not_implemented", "not_implemented"]
    assert all(row["paper_table2_required"] is True for row in result["rows"])
    assert all(row["result_claim_status"] == "missing_required_paper_baseline" for row in result["rows"])
    assert all(row["claim_blocks"] for row in result["rows"])


def test_dadda_training_loop_stops_at_shorter_source_target_stream():
    from paper_reproduction.DADDA.model import DADDANet
    from paper_reproduction.DADDA.train import run_dadda_training_loop

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


def test_dadda_table2_settings_default_to_paper_config_and_gate_smoke():
    from paper_reproduction.DADDA.train import resolve_table2_run_settings, validate_formal_or_smoke_settings

    config = {"epochs": 100, "batch_size": 128, "momentum": 0.9, "weight_decay": 0.0005}
    args = SimpleNamespace(epochs=None, batch_size=None, learning_rate=None, momentum=None, weight_decay=None)
    settings = resolve_table2_run_settings(config, args)

    assert settings["epochs"] == 100
    assert settings["batch_size"] == 128
    assert settings["learning_rate"] == 0.0001
    assert settings["momentum"] == 0.9
    assert settings["weight_decay"] == 0.0005
    assert settings["detach_target_probabilities"] is False
    assert settings["alpha_mode"] == "dynamic"
    assert settings["fixed_alpha"] == 0.5
    assert settings["detach_dynamic_alpha"] is False
    validate_formal_or_smoke_settings(
        config=config,
        settings=settings,
        smoke=False,
        max_samples_per_combo=None,
        max_batches_per_epoch=None,
    )

    limited = dict(settings, epochs=1)
    try:
        validate_formal_or_smoke_settings(
            config=config,
            settings=limited,
            smoke=False,
            max_samples_per_combo=1,
            max_batches_per_epoch=None,
        )
    except SystemExit as exc:
        assert "--smoke is required" in str(exc)
    else:
        raise AssertionError("non-paper formal settings must require --smoke")

    validate_formal_or_smoke_settings(
        config=config,
        settings=limited,
        smoke=True,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
    )


def test_dadda_table2_settings_allow_cli_to_disable_config_detach():
    from paper_reproduction.DADDA.train import resolve_table2_run_settings

    config = {"detach_target_probabilities": True}
    args = SimpleNamespace(
        epochs=None,
        batch_size=None,
        learning_rate=None,
        momentum=None,
        weight_decay=None,
        paper_domain_sample_count=None,
        normalize=None,
        crop_mode=None,
        detach_target_probabilities=False,
        alpha_mode=None,
        fixed_alpha=None,
    )
    settings = resolve_table2_run_settings(config, args)

    assert settings["detach_target_probabilities"] is False


def test_dadda_cli_requires_formal_for_real_table2(tmp_path):
    output_path = tmp_path / "should_not_exist.json"
    command = [
        sys.executable,
        "-m",
        "paper_reproduction.DADDA.train",
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
