from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch


def test_receiver_agnostic_model_matches_paper_shapes():
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet

    model = ReceiverAgnosticUDANet(num_tx=6, feature_dim=128, classifier_hidden_dim=128)
    batch = torch.randn(4, 2, 256)

    outputs = model(batch, grl_lambda=0.5, return_activations=True)

    assert outputs["features"].shape == (4, 128)
    assert outputs["tx_logits"].shape == (4, 6)
    assert outputs["domain_logits"].shape == (4, 2)
    assert len(outputs["activations"]) == 4


def test_receiver_agnostic_model_rejects_non_256_iq_inputs():
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet

    model = ReceiverAgnosticUDANet(num_tx=6)

    with pytest.raises(ValueError, match="256"):
        model(torch.randn(2, 2, 320))


def test_grl_reverses_feature_gradient_sign():
    from baselines.common.grl import gradient_reverse

    x = torch.tensor([2.0], requires_grad=True)
    y = gradient_reverse(x, 0.5) * 4.0

    y.backward()

    assert torch.allclose(x.grad, torch.tensor([-2.0]))


def test_dann_loss_combines_source_tx_and_two_class_domain_ce():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import dann_loss

    source_outputs = {
        "tx_logits": torch.randn(4, 6, requires_grad=True),
        "domain_logits": torch.randn(4, 2, requires_grad=True),
    }
    target_outputs = {
        "tx_logits": torch.randn(5, 6, requires_grad=True),
        "domain_logits": torch.randn(5, 2, requires_grad=True),
    }
    source_labels = torch.tensor([0, 1, 2, 3])

    losses = dann_loss(source_outputs, target_outputs, source_labels)

    assert set(losses) == {"loss", "loss_tx", "loss_domain"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_lmmd_loss_uses_target_probabilities_without_target_labels():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import lmmd_loss

    source_features = torch.randn(5, 4, requires_grad=True)
    target_features = torch.randn(6, 4, requires_grad=True)
    source_labels = torch.tensor([0, 1, 2, 0, 1])
    target_probs = torch.softmax(torch.randn(6, 3), dim=1)

    loss = lmmd_loss(source_features, target_features, source_labels, target_probs, num_classes=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert source_features.grad is not None
    assert target_features.grad is not None


def test_lmmd_loss_rejects_non_probability_target_inputs():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import lmmd_loss

    with pytest.raises(ValueError, match="sum to 1"):
        lmmd_loss(
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.tensor([0, 1, 2]),
            torch.ones(3, 3),
            num_classes=3,
        )


def test_stage2_lmmd_objective_matches_eq16_components():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import stage2_lmmd_objective
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet

    model = ReceiverAgnosticUDANet(num_tx=3)
    source_outputs = model(torch.randn(4, 2, 256), return_activations=True)
    target_outputs = model(torch.randn(5, 2, 256), return_activations=True)
    source_labels = torch.tensor([0, 1, 2, 0])

    losses = stage2_lmmd_objective(source_outputs, target_outputs, source_labels, num_classes=3, lmmd_lambda=0.25)

    assert set(losses) == {"loss", "loss_tx", "loss_lmmd"}
    assert torch.allclose(losses["loss"], losses["loss_tx"] + 0.25 * losses["loss_lmmd"])
    assert torch.isfinite(losses["loss"])


def test_stage2_lmmd_objective_can_align_feature_layer_only():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import stage2_lmmd_objective
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet

    model = ReceiverAgnosticUDANet(num_tx=3)
    source_outputs = model(torch.randn(4, 2, 256))
    target_outputs = model(torch.randn(5, 2, 256))
    source_labels = torch.tensor([0, 1, 2, 0])

    losses = stage2_lmmd_objective(
        source_outputs,
        target_outputs,
        source_labels,
        num_classes=3,
        lmmd_lambda=0.05,
        lmmd_layers="features",
    )

    assert torch.allclose(losses["loss"], losses["loss_tx"] + 0.05 * losses["loss_lmmd"])
    assert torch.isfinite(losses["loss"])


def test_uncertainty_sampling_orders_hard_target_samples():
    from paper_reproduction.receiver_agnostic_twostage_uda.sampling import rank_uncertain_samples

    logits = torch.tensor(
        [
            [6.0, 0.1, 0.0],
            [1.1, 1.0, 0.9],
            [2.0, 1.9, 0.1],
            [3.0, 0.2, 0.1],
        ]
    )

    assert rank_uncertain_samples(logits, strategy="entropy", k=2).tolist() == [1, 2]
    assert rank_uncertain_samples(logits, strategy="margin", k=2).tolist() == [1, 2]
    assert rank_uncertain_samples(logits, strategy="least_confidence", k=2).tolist() == [1, 2]


def test_random_sampling_is_reproducible_with_seed():
    from paper_reproduction.receiver_agnostic_twostage_uda.sampling import rank_uncertain_samples

    logits = torch.randn(12, 3)

    first = rank_uncertain_samples(logits, strategy="random", k=5, seed=17)
    second = rank_uncertain_samples(logits, strategy="random", k=5, seed=17)

    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 5


def test_paper_faithful_protocol_builds_receiver_ratio_plan():
    from paper_reproduction.receiver_agnostic_twostage_uda.protocol import (
        build_receiver_ratio_plan,
        validate_paper_faithful_config,
    )

    config = {
        "paper_scope": "paper_faithful",
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "source_receiver_counts": [1, 2, 3, 4, 6],
        "tx_count": 6,
        "capture_days": 4,
        "preprocessing": {
            "equalized": 1,
            "normalize": True,
            "normalization": "RMS/power normalization",
            "crop_mode": "left",
            "out_len": 256,
            "cfo_policy": "preserved",
        },
        "target_unlabeled_allowed": True,
        "cvs_extension": False,
    }

    checked = validate_paper_faithful_config(config)
    plan = build_receiver_ratio_plan(checked)

    assert [row["ratio"] for row in plan] == ["1:11", "2:10", "3:9", "4:8", "6:6"]
    assert plan[-1]["table_i_target_receiver_count"] == 6
    assert plan[-1]["table_i_paper_reference_accuracy"] == [0.89, 0.94, 0.87, 0.91, 0.92, 0.92]
    assert plan[-1]["table_i_reference_only"] is True
    assert plan[0]["receiver_split_ids"] is None
    assert plan[0]["seed"] is None
    assert plan[0]["fine_tune_iterations_to_report"][-1] == 100
    assert checked["claim_boundary"] == "paper-faithful closed-set cross-receiver UDA"


def test_paper_faithful_protocol_rejects_cvs_extension_mixing():
    from paper_reproduction.receiver_agnostic_twostage_uda.protocol import validate_paper_faithful_config

    config = {
        "paper_scope": "paper_faithful",
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "source_receiver_counts": [6],
        "tx_count": 6,
        "capture_days": 4,
        "preprocessing": {
            "equalized": 1,
            "normalize": True,
            "normalization": "RMS/power normalization",
            "crop_mode": "left",
            "out_len": 256,
            "cfo_policy": "preserved",
        },
        "target_unlabeled_allowed": True,
        "cvs_extension": True,
    }

    try:
        validate_paper_faithful_config(config)
    except ValueError as exc:
        assert "cvs_extension" in str(exc)
    else:
        raise AssertionError("paper-faithful config must reject cvs_extension mixing")


def _synthetic_manysig_compact() -> dict:
    data = []
    for tx_i in range(6):
        tx_rows = []
        for rx_i in range(12):
            rx_rows = []
            for day_i in range(4):
                eq_rows = []
                for eq_i in range(2):
                    sample = np.zeros((2, 320, 2), dtype=np.float32)
                    sample[:, :, 0] = float(tx_i + 1)
                    sample[:, :, 1] = float(rx_i + day_i + eq_i)
                    sample[:, 0, 0] = 1000.0 + tx_i
                    eq_rows.append(sample)
                rx_rows.append(eq_rows)
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": [f"tx{i}" for i in range(6)],
        "rx_list": [f"rx{i}" for i in range(12)],
        "capture_date_list": [f"day{i}" for i in range(4)],
        "equalized_list": [0, 1],
    }


def test_manysig_receiver_uda_dataset_contract_uses_first256_equalized_rx_domains():
    from paper_reproduction.receiver_agnostic_twostage_uda.data import build_manysig_receiver_uda_datasets

    built = build_manysig_receiver_uda_datasets(
        _synthetic_manysig_compact(),
        source_receivers=["rx0", "rx1"],
        target_receivers=["rx2"],
        max_samples_per_combo=1,
    )
    source_x, source_y, source_d, source_meta = built["source"][0]
    target_x, _, target_d, target_meta = built["target"][0]

    assert source_x.shape == (2, 256)
    assert abs(float(source_x[0, 0])) > 1.0
    assert source_meta["equalized"] == 1
    assert target_meta["equalized"] == 1
    assert set(built["meta"]["source_receiver_ids"]).isdisjoint(built["meta"]["target_receiver_ids"])
    assert built["meta"]["preprocessing"]["crop_mode"] == "left"
    assert built["meta"]["target_label_role"] == "hidden_for_UDA_available_only_for_eval_or_optional_finetune"
    assert int(source_y) in range(6)
    assert int(source_d) in built["meta"]["source_receiver_ids"]
    assert int(target_d) in built["meta"]["target_receiver_ids"]


def test_finetune_budget_and_source_replay_helpers_are_bounded():
    from paper_reproduction.receiver_agnostic_twostage_uda.sampling import (
        balanced_source_replay_indices,
        fine_tune_budget_from_unlabeled,
    )

    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    replay = balanced_source_replay_indices(labels, per_class=2, seed=7)

    assert fine_tune_budget_from_unlabeled(2500) == 50
    assert fine_tune_budget_from_unlabeled(3) == 1
    assert replay.numel() == 5
    assert torch.bincount(labels[replay], minlength=3).tolist() == [2, 2, 1]


def test_single_batch_training_steps_are_reachable_without_target_labels():
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet
    from paper_reproduction.receiver_agnostic_twostage_uda.steps import (
        dann_stage1_train_step,
        fig8_finetune_train_step,
        lmmd_stage2_train_step,
    )

    model = ReceiverAgnosticUDANet(num_tx=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    source = {"iq": torch.randn(4, 2, 256), "label": torch.tensor([0, 1, 2, 0])}
    target = {"iq": torch.randn(5, 2, 256), "label": torch.tensor([2, 2, 1, 0, 1])}
    before = next(model.parameters()).detach().clone()

    dann_metrics = dann_stage1_train_step(model, source, target, optimizer)
    lmmd_metrics = lmmd_stage2_train_step(model, source, target, optimizer, num_classes=3, lmmd_lambda=0.1)
    finetune_metrics = fig8_finetune_train_step(model, source, optimizer)

    assert set(dann_metrics) == {"loss", "loss_tx", "loss_domain"}
    assert set(lmmd_metrics) == {"loss", "loss_tx", "loss_lmmd"}
    assert torch.isfinite(torch.tensor([dann_metrics["loss"], lmmd_metrics["loss"], finetune_metrics["loss"]])).all()
    assert not torch.allclose(before, next(model.parameters()).detach())


def test_fig8_selection_and_batch_composition_keep_roles():
    from paper_reproduction.receiver_agnostic_twostage_uda.steps import (
        compose_fig8_finetune_batch,
        select_fig8_labeled_target_indices,
    )

    logits = torch.randn(2500, 3)
    selected = select_fig8_labeled_target_indices(logits, strategy="random", seed=5)
    source_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    batch = compose_fig8_finetune_batch(
        torch.randn(2500, 2, 256),
        torch.arange(2500) % 3,
        selected["selected"],
        torch.randn(6, 2, 256),
        source_labels,
        source_replay_per_class=1,
        seed=11,
    )

    assert selected["budget"] == 50
    assert selected["result_claim"] is False
    assert batch["iq"].shape[0] == 53
    assert batch["role"].tolist().count(1) == 50
    assert batch["role"].tolist().count(0) == 3


def test_non_dry_run_does_not_write_output(tmp_path):
    output = tmp_path / "should_not_exist.json"
    config = "paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_reproduction.receiver_agnostic_twostage_uda.train",
            "--config",
            config,
            "--output",
            str(output),
        ],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert not output.exists()


def test_formal_training_smoke_writes_paper_scoped_rows(tmp_path):
    pkl_path = tmp_path / "ManySig.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(_synthetic_manysig_compact(), f)
    output_dir = tmp_path / "formal"
    config = "paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_reproduction.receiver_agnostic_twostage_uda.train",
            "--config",
            config,
            "--formal",
            "--manysig-pkl",
            str(pkl_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--limit-ratios",
            "1",
            "--methods",
            "source_only,dann_lmmd",
            "--source-epochs",
            "1",
            "--stage1-epochs",
            "1",
            "--stage2-epochs",
            "1",
            "--max-train-steps",
            "1",
            "--max-samples-per-combo",
            "1",
            "--batch-size",
            "4",
            "--eval-batch-size",
            "8",
            "--num-workers",
            "0",
            "--progress-every",
            "0",
            "--transductive-target-eval",
            "--lmmd-lambda",
            "0.05",
            "--lmmd-layers",
            "features",
        ],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [row["method"] for row in rows] == ["source_only", "dann_lmmd"]
    assert summary["paper_scope"] == "paper-faithful closed-set cross-receiver UDA"
    assert rows[0]["artifact_type"] == "formal_training_result"
    assert rows[0]["source_receiver_count"] == 1
    assert rows[0]["target_receiver_count"] == 11
    assert rows[0]["target_eval_protocol"] == "transductive_all_target_unlabeled_for_UDA_and_eval"
    assert rows[0]["target_adapt_size"] == rows[0]["target_eval_size"]
    assert rows[1]["hyperparameters"]["lmmd_lambda"] == 0.05
    assert rows[1]["hyperparameters"]["lmmd_layers"] == "features"
    assert rows[0]["claim_blocks"] == [
        "not CVS Stage2-C",
        "not satellite/LEO deployment evidence",
        "not open-set or new-class registration",
    ]
