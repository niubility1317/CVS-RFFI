from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import SSDG.train_ssdg as train_ssdg  # noqa: E402
from cvsrffi.phase1_pamr import (  # noqa: E402
    FROZEN_PAMR_LAMBDA,
    PAMRConfig,
    PAMRConfigurationError,
    PAMRRuntimeError,
    add_pamr_to_loss,
    pamr_config_receipt,
    pamr_gradient_status,
    pamr_loss,
    remap_pamr_local_labels_to_head_rows,
    resolve_pamr_local_head_class_binding,
    pamr_shared_encoder_parameters,
    pamr_shared_gradient_relation,
    pamr_unscaled_gradient,
    require_finite_pamr_gradient,
    resolve_pamr_classifier_weight,
    strict_pamr_warm_start,
    update_pamr_gradient_receipt,
    update_pamr_gradient_relation_receipt,
    update_pamr_receipt,
    validate_pamr_args,
    validate_pamr_binding,
    validate_pamr_terminal_receipt,
    write_pamr_failure_receipt,
)


AUDIT_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_pamr_audit6_20260809.sh"
FULL_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_pamr12_20260809.sh"


def _frozen_args(*, enabled: bool = True, audit_only: bool = False, epochs: int | None = None):
    return SimpleNamespace(
        phase1_pamr_frozen_mode=True,
        phase1_pamr_enabled=enabled,
        phase1_pamr_audit_only=audit_only,
        lambda_pamr=FROZEN_PAMR_LAMBDA if enabled else 0.0,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        epochs=(1 if audit_only else 40) if epochs is None else epochs,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        id_feature_key="feat_joint",
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        use_concat_sat_channel_aug=False,
        use_unlabeled=False,
        use_tx_rx_balanced_sampler=False,
        use_aug=False,
        use_mixstyle=False,
        reject_head=False,
        phase1_ccpc_leo_frozen_mode=False,
        phase1_ccpc_leo_enabled=False,
        phase1_ccpc_leo_gradient_audit_only=False,
        lambda_ccpc_leo=0.0,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )


def _paired_margin_tensors():
    # Clean points are raw-cosine correct and have margin 1; LEO point 0 is
    # intentionally flipped so its margin is -1 and hinge is exactly 2.
    clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    leo = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    weight = torch.nn.Parameter(torch.eye(2))
    return clean, leo, labels, weight


def test_pamr_uses_raw_cosine_clean_gate_detached_margin_and_tx_equal_hinge():
    clean, leo, labels, weight = _paired_margin_tensors()
    loss, info = pamr_loss(leo, clean, labels, weight)

    # Class 0 hinge=2 and class 1 hinge=0: equal-TX mean is 1, not a
    # sample-count or CosFace-label-margin artefact.
    assert torch.isclose(loss.detach(), torch.tensor(1.0), atol=1e-6)
    assert info["valid_anchors_by_tx"] == {"0": 1, "1": 1}
    assert info["active_hinges_by_tx"] == {"0": 1, "1": 0}
    assert info["clean_gate_raw_cosine"] is True
    assert info["clean_margin_detached"] is True
    assert info["class_weight_detached"] is True
    assert info["tx_equal_aggregation"] is True


def test_pamr_clean_and_classifier_weight_have_no_gradient_but_leo_does():
    clean, leo, labels, weight = _paired_margin_tensors()
    loss, _ = pamr_loss(leo, clean, labels, weight)
    loss.backward()

    assert clean.grad is None
    assert weight.grad is None
    assert leo.grad is not None
    assert float(leo.grad.abs().sum()) > 0.0


def test_pamr_is_label_permutation_equivariant():
    clean, leo, labels, weight = _paired_margin_tensors()
    direct, direct_info = pamr_loss(leo, clean, labels, weight)
    permutation = torch.tensor([1, 0])
    permuted, permuted_info = pamr_loss(
        leo,
        clean,
        permutation[labels],
        torch.nn.Parameter(weight.detach()[permutation].clone()),
    )
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-7)
    assert direct_info["valid_anchors"] == permuted_info["valid_anchors"]
    assert direct_info["active_hinges"] == permuted_info["active_hinges"]


def test_pamr_has_no_rx_or_domain_inputs_and_rejects_nonfinite_or_binding_drift():
    signature = inspect.signature(pamr_loss)
    assert "rx" not in " ".join(signature.parameters).lower()
    assert "domain" not in " ".join(signature.parameters).lower()
    clean, leo, labels, weight = _paired_margin_tensors()
    bad = leo.detach().clone()
    bad[0, 0] = float("nan")
    with pytest.raises(PAMRRuntimeError, match="non-finite"):
        pamr_loss(bad.requires_grad_(True), clean, labels, weight)
    with pytest.raises(PAMRRuntimeError, match="dimension"):
        pamr_loss(leo, clean, labels, torch.nn.Parameter(torch.ones(2, 3)))


def test_pamr_raw_gradient_and_shared_encoder_relation_are_unscaled_and_finite():
    encoder = torch.nn.Linear(2, 2, bias=False)
    weight = torch.nn.Parameter(torch.eye(2))
    model = SimpleNamespace(
        id_backbone=torch.nn.Module(),
    )
    model.id_backbone.encoder = encoder
    model.id_backbone.cls_head = torch.nn.Module()
    model.id_backbone.cls_head.head = torch.nn.Module()
    model.id_backbone.cls_head.head.weight = weight
    clean_input = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    leo_input = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    clean = encoder(clean_input)
    leo = encoder(leo_input)
    labels = torch.tensor([0, 1])
    loss, _ = pamr_loss(leo, clean, labels, weight)
    raw = pamr_unscaled_gradient(loss, leo, loss_weight=FROZEN_PAMR_LAMBDA)
    status = pamr_gradient_status(raw)
    assert status["finite"] is True
    require_finite_pamr_gradient(status)
    relation = pamr_shared_gradient_relation(
        (leo.square().mean() + clean.square().mean()),
        loss,
        pamr_shared_encoder_parameters(model),
        loss_weight=FROZEN_PAMR_LAMBDA,
    )
    assert relation["shared_parameter_count"] >= 1.0
    assert relation["norm_ratio"] >= 0.0


def test_pamr_finite_zero_is_recorded_but_none_and_nonfinite_fail_closed():
    status = pamr_gradient_status(torch.zeros(4, 2))
    assert status == {"finite": True, "nonzero": False, "zero": True, "nonfinite": False}
    require_finite_pamr_gradient(status)
    with pytest.raises(PAMRRuntimeError, match="no LEO feature gradient"):
        pamr_gradient_status(None)
    bad_status = pamr_gradient_status(torch.tensor([float("nan")]))
    with pytest.raises(PAMRRuntimeError, match="non-finite"):
        require_finite_pamr_gradient(bad_status)


def test_pamr_failure_receipt_is_atomic_minimal_and_best_effort_does_not_mask(tmp_path, monkeypatch, capsys):
    receipt = pamr_config_receipt(PAMRConfig(True, True, FROZEN_PAMR_LAMBDA, audit_only=True))
    original = PAMRRuntimeError("P1-PAMR LEO feature gradient is non-finite")
    written = write_pamr_failure_receipt(
        tmp_path,
        candidate_id="F1G",
        run_id="pamr-audit",
        receipt=receipt,
        error=original,
        failure_stage="pre_scaled_backward_first_effective_pamr_gradient_audit",
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["schema"] == "cvs.phase1.pamr_failure_receipt.v1"
    assert payload["status"] == "FAIL_CLOSED"
    assert payload["error_fingerprint"] == "PAMR_LEO_GRADIENT_NONFINITE"
    assert "gradient is non-finite" not in written.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".pamr_failure_receipt.*.tmp"))

    def writer_failure(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(train_ssdg, "write_pamr_failure_receipt", writer_failure)
    caught = None
    try:
        try:
            raise original
        except PAMRRuntimeError as error:
            train_ssdg._persist_pamr_failure_receipt(
                out_dir=tmp_path,
                args=SimpleNamespace(candidate_id="F1G", run_id="pamr-audit"),
                pamr_receipt=receipt,
                error=error,
                failure_stage="test",
            )
            raise
    except PAMRRuntimeError as error:
        caught = error
    assert caught is original
    assert "[P1-PAMR-FAILURE-RECEIPT] persistence_failed writer_exception_type=OSError" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("message", "fingerprint"),
    (
        ("P1-PAMR has no LEO feature gradient", "PAMR_LEO_GRADIENT_MISSING"),
        ("P1-PAMR LEO feature gradient is non-finite", "PAMR_LEO_GRADIENT_NONFINITE"),
        ("P1-PAMR shared encoder gradient is non-finite", "PAMR_SHARED_GRADIENT_FAILURE"),
    ),
)
def test_pamr_failure_receipt_classifies_none_nonfinite_and_shared_gradient(tmp_path, message, fingerprint):
    receipt = pamr_config_receipt(PAMRConfig(True, True, FROZEN_PAMR_LAMBDA, audit_only=True))
    written = write_pamr_failure_receipt(
        tmp_path,
        candidate_id="F1G",
        run_id="pamr-audit",
        receipt=receipt,
        error=PAMRRuntimeError(message),
        failure_stage="test",
    )
    assert json.loads(written.read_text(encoding="utf-8"))["error_fingerprint"] == fingerprint


def test_train_limits_raw_gradient_relation_to_first_effective_audit_batch_only():
    source = inspect.getsource(train_ssdg.train)
    start = source.index("pamr_audit_has_effective_batch")
    end = source.index("if loss_is_finite:", start)
    audit_block = source[start:end]
    assert "and pamr_audit_only" in audit_block
    assert 'not bool(pamr_receipt.get("pamr_gradient_audit_completed", False))' in audit_block
    assert "update_pamr_receipt(pamr_receipt, pamr_batch_info)" in source
    assert "update_pamr_gradient_receipt(" in audit_block
    assert "pamr_shared_gradient_relation(" in audit_block
    assert 'pamr_receipt["pamr_gradient_audit_completed"] = True' in audit_block


def test_terminal_receipt_separates_audit_gradient_health_from_formal_coverage():
    receipt = pamr_config_receipt(PAMRConfig(True, True, FROZEN_PAMR_LAMBDA, audit_only=True))
    receipt["expected_tx_class_ids"] = [0, 1]
    receipt["class_count"] = 2
    batch = {
        "rows": 2,
        "classes": 2,
        "valid_anchors": 2,
        "active_hinges": 2,
        "valid_anchors_by_tx": {"0": 1, "1": 1},
        "active_hinges_by_tx": {"0": 1, "1": 1},
    }
    zero_only = update_pamr_receipt(
        receipt, batch, leo_grad_nonzero=False, leo_grad_zero=True, leo_grad_nonfinite=False
    )
    zero_only = update_pamr_gradient_relation_receipt(
        zero_only,
        {"shared_parameter_count": 1.0, "cosine": 0.2, "norm_ratio": 0.4},
    )
    with pytest.raises(PAMRRuntimeError, match="at least one nonzero"):
        validate_pamr_terminal_receipt(zero_only)
    passed = update_pamr_receipt(
        zero_only, batch, leo_grad_nonzero=True, leo_grad_zero=False, leo_grad_nonfinite=False
    )
    passed["pamr_gradient_audit_completed"] = True
    terminal = validate_pamr_terminal_receipt(passed)
    assert terminal["pamr_terminal_gradient_contract_passed"] is True
    assert terminal["pamr_terminal_gradient_contract"] == (
        "AUDIT_RAW_NONZERO_GRADIENT_AND_PER_TX_ANCHOR_HINGE_COVERAGE"
    )

    missing_hinge = pamr_config_receipt(PAMRConfig(True, True, FROZEN_PAMR_LAMBDA, audit_only=True))
    missing_hinge["expected_tx_class_ids"] = [0, 1]
    missing_hinge = update_pamr_receipt(
        missing_hinge,
        {**batch, "active_hinges": 1, "active_hinges_by_tx": {"0": 1, "1": 0}},
        leo_grad_nonzero=True,
        leo_grad_zero=False,
        leo_grad_nonfinite=False,
    )
    missing_hinge = update_pamr_gradient_relation_receipt(
        missing_hinge,
        {"shared_parameter_count": 1.0, "cosine": 0.1, "norm_ratio": 0.2},
    )
    with pytest.raises(PAMRRuntimeError, match="zero active hinge"):
        validate_pamr_terminal_receipt(missing_hinge)

    formal = pamr_config_receipt(PAMRConfig(True, True, FROZEN_PAMR_LAMBDA, audit_only=False))
    formal["expected_tx_class_ids"] = [0, 1]
    formal = update_pamr_receipt(formal, batch)
    assert formal["pamr_grad_nonzero_batches"] == 0
    assert formal["pamr_shared_gradient_relation_batches"] == 0
    formal_terminal = validate_pamr_terminal_receipt(formal)
    assert formal_terminal["pamr_terminal_gradient_contract"] == "FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE"
    assert formal_terminal["pamr_terminal_gradient_contract_passed"] is True


def test_control_path_is_identity_and_frozen_cli_rejects_drift_and_teacher_stacking():
    base = torch.tensor(1.25, requires_grad=True)
    assert add_pamr_to_loss(base, None, PAMRConfig(True, False, 0.0)) is base
    active = _frozen_args()
    config = validate_pamr_args(active)
    assert config.enabled is True and config.loss_weight == FROZEN_PAMR_LAMBDA
    assert pamr_config_receipt(config)["uses_external_ema_teacher"] is False

    wrong_lambda = _frozen_args()
    wrong_lambda.lambda_pamr = 0.01
    with pytest.raises(PAMRConfigurationError, match="lambda_pamr"):
        validate_pamr_args(wrong_lambda)
    for name, value in (
        ("lambda_teacher_clean_kl", 0.01),
        ("lambda_teacher_sat_kl", 0.01),
        ("lambda_teacher_zid_mse", 0.01),
        ("teacher_ckpt", "teacher.pth"),
        ("phase1_ccpc_leo_enabled", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(PAMRConfigurationError, match=name):
            validate_pamr_args(bad)


def test_audit_is_one_epoch_g_only_and_skips_heldout_performance_path():
    audit = _frozen_args(audit_only=True)
    config = validate_pamr_args(audit)
    assert config.audit_only is True
    receipt = pamr_config_receipt(config)
    assert receipt["technical_only"] is True
    assert receipt["technical_only_claim"] == "NO_PERFORMANCE_RESULT"
    not_g = _frozen_args(enabled=False, audit_only=True)
    with pytest.raises(PAMRConfigurationError, match="requires the frozen enabled G arm"):
        validate_pamr_args(not_g)
    wrong_length = _frozen_args(audit_only=True, epochs=40)
    with pytest.raises(PAMRConfigurationError, match="--epochs 1"):
        validate_pamr_args(wrong_length)

    frozen_eval = train_ssdg._resolve_frozen_phase1_evaluation(
        SimpleNamespace(), object(), {}, object(), "unused.pth", technical_only=True, selection_source="technical"
    )
    assert frozen_eval["status"] == "SKIPPED_TECHNICAL_AUDIT"
    skipped = train_ssdg._pamr_technical_audit_skip_receipt("source_val_clean")
    assert skipped == {
        "status": "SKIPPED_TECHNICAL_AUDIT",
        "selection_source": "TECHNICAL_ONLY",
        "claim": "NO_PERFORMANCE_RESULT",
        "scope": "source_val_clean",
    }
    source = inspect.getsource(train_ssdg.train)
    assert "if pamr_audit_only:\n            val_stats = {" in source
    assert "source_val_heavy_eval_ran = (not pamr_audit_only)" in source
    assert "final_source_val_tail = _pamr_technical_audit_skip_receipt" in source
    assert "final_zid_leakage_probe = _pamr_technical_audit_skip_receipt" in source
    assert "technical_only=pamr_audit_only" not in source
    assert '_pamr_technical_audit_skip_receipt("frozen_phase1_heldout")' in source
    assert '"phase1_pamr_terminal_receipt.json"' in source


def test_model_binding_requires_feat_joint_head_dimension_and_class_order():
    weight = torch.nn.Parameter(torch.eye(2))
    model = SimpleNamespace(id_backbone=torch.nn.Module())
    model.id_backbone.cls_head = torch.nn.Module()
    model.id_backbone.cls_head.head = torch.nn.Module()
    model.id_backbone.cls_head.head.weight = weight
    clean = torch.randn(2, 2, requires_grad=True)
    leo = torch.randn(2, 2, requires_grad=True)
    out_clean = {"z_id_key": "feat_joint", "z_id": clean, "tx_logits": torch.randn(2, 2)}
    out_leo = {"z_id_key": "feat_joint", "z_id": leo, "tx_logits": torch.randn(2, 2)}
    assert resolve_pamr_classifier_weight(model) is weight
    assert validate_pamr_binding(model=model, out_clean=out_clean, out_leo=out_leo, tx_labels=torch.tensor([0, 1])) is weight
    out_leo["z_id_key"] = "feat_cls"
    with pytest.raises(PAMRRuntimeError, match="feat_joint"):
        validate_pamr_binding(model=model, out_clean=out_clean, out_leo=out_leo, tx_labels=torch.tensor([0, 1]))


def test_pamr_global_six_to_local_four_binding_is_explicit_and_live_head_strict():
    global_tx_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_tx_order = ["20-15", "20-19", "6-15", "8-20"]
    filtered, partition = train_ssdg._phase1_tx_partition_view(
        {"tx_list": global_tx_order, "data": list(range(6))},
        train_spec=",".join(local_tx_order),
        known_validation_spec="14-7",
        proxy_unknown_spec="14-10",
    )
    assert filtered["tx_list"] == local_tx_order
    assert partition["dataset_tx_order"] == global_tx_order
    assert partition["training_view_contiguous_reindex"] == {
        "0": "20-15", "1": "20-19", "2": "6-15", "3": "8-20"
    }
    binding = resolve_pamr_local_head_class_binding(
        local_class_order=local_tx_order,
        source_train_tx=partition["source_known_train_tx"],
        checkpoint_train_tx=local_tx_order,
        dataset_class_order=partition["dataset_tx_order"],
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["dataset_class_count"] == 6
    assert binding["local_data_class_count"] == 4
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert binding["local_to_head_class_ids"] == [0, 1, 2, 3]
    assert len(binding["class_order_binding_sha256"]) == 64
    assert torch.equal(
        remap_pamr_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0, 2]),
    )
    assert '"num_classes": int(infer_nc)' in inspect.getsource(train_ssdg._build_ssdg_wisig_data)


def test_pamr_rejects_global_six_head_or_checkpoint_tx_order_drift_for_local_four_data():
    global_tx_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_tx_order = ["20-15", "20-19", "6-15", "8-20"]
    with pytest.raises(PAMRConfigurationError, match="live classifier head row count"):
        resolve_pamr_local_head_class_binding(
            local_class_order=local_tx_order,
            source_train_tx=local_tx_order,
            checkpoint_train_tx=local_tx_order,
            dataset_class_order=global_tx_order,
            local_data_class_count=4,
            checkpoint_head_class_count=6,
            live_head_class_count=6,
        )
    with pytest.raises(PAMRConfigurationError, match="checkpoint train TX class order"):
        resolve_pamr_local_head_class_binding(
            local_class_order=local_tx_order,
            source_train_tx=local_tx_order,
            checkpoint_train_tx=["20-19", "20-15", "6-15", "8-20"],
            dataset_class_order=global_tx_order,
            local_data_class_count=4,
            checkpoint_head_class_count=4,
            live_head_class_count=4,
        )


def test_strict_warm_start_is_weights_only_with_exact_keys():
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    receipt = strict_pamr_warm_start(
        target,
        source.state_dict(),
        baseline_path="geosat_c_final.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="final",
    )
    assert receipt["strict_model_keys"] is True
    assert receipt["optimizer_state_restored"] is False
    assert receipt["rng_state_restored"] is False
    with pytest.raises(PAMRConfigurationError, match="strict baseline"):
        strict_pamr_warm_start(
            torch.nn.Linear(3, 2),
            {"weight": source.weight.detach().clone()},
            baseline_path="geosat_c_final.pth",
            baseline_sha256="b" * 64,
            checkpoint_epoch=40,
            checkpoint_role="final",
        )


def test_launchers_have_frozen_matrix_and_dry_run_closure():
    audit_text = AUDIT_LAUNCHER.read_text(encoding="utf-8")
    assert re.findall(r"^launch_fold (\d) (\d)$", audit_text, flags=re.MULTILINE) == [
        (str(index), str(index - 1)) for index in range(1, 7)
    ]
    assert "--epochs 1" in audit_text and "--phase1_pamr_audit_only true" in audit_text
    full_text = FULL_LAUNCHER.read_text(encoding="utf-8")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", full_text, flags=re.MULTILINE)
    assert calls == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    assert "GEOSAT_CKPT_ROOT:-${PROJECT_ROOT}/runs/phase1_loto_clsgeo12_20260808_v1" in full_text
    assert "lambda_pamr 0.05" in full_text and "lambda_pamr 0)" in full_text
    assert "lambda_teacher_clean_kl 0" in full_text and "teacher_ckpt \"\"" in full_text

    for launcher, expected in ((AUDIT_LAUNCHER, 6), (FULL_LAUNCHER, 12)):
        completed = subprocess.run(
            ["bash", f"scripts/{launcher.name}", "--dry-run"],
            cwd=str(CODE_ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        assert completed.stdout.count("[DRY-RUN]") == expected
