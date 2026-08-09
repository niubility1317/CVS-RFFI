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
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import SSDG.train_ssdg as train_ssdg  # noqa: E402
from cvsrffi.phase1_cb_sfce import (  # noqa: E402
    FROZEN_CB_SFCE_GAMMA,
    FROZEN_CB_SFCE_LAMBDA,
    FROZEN_CB_SFCE_SCENARIOS,
    CBSFCEConfig,
    CBSFCEConfigurationError,
    CBSFCERuntimeError,
    add_cb_sfce_to_loss,
    cb_sfce_config_receipt,
    cb_sfce_loss,
    cb_sfce_shared_encoder_and_head_parameters,
    cb_sfce_shared_gradient_relation,
    remap_cb_sfce_local_labels_to_head_rows,
    resolve_cb_sfce_classifier_weight,
    resolve_cb_sfce_local_head_class_binding,
    strict_cb_sfce_warm_start,
    update_cb_sfce_gradient_relation_receipt,
    update_cb_sfce_receipt,
    validate_cb_sfce_args,
    validate_cb_sfce_logit_binding,
    validate_cb_sfce_terminal_receipt,
    write_cb_sfce_failure_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_cb_sfce12_20260809.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40):
    return SimpleNamespace(
        phase1_cb_sfce_frozen_mode=True,
        phase1_cb_sfce_enabled=enabled,
        lambda_cb_sfce=FROZEN_CB_SFCE_LAMBDA if enabled else 0.0,
        cb_sfce_gamma=FROZEN_CB_SFCE_GAMMA,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        epochs=epochs,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        sat_cons_start_epoch=1,
        sat_view_prob=1.0,
        sat_train_scenarios=",".join(FROZEN_CB_SFCE_SCENARIOS),
        sat_view_schedule="",
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
        phase1_pamr_frozen_mode=False,
        phase1_pamr_enabled=False,
        phase1_pamr_audit_only=False,
        lambda_pamr=0.0,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )


def _logits_and_labels():
    logits = torch.tensor(
        [[2.0, 0.0], [1.5, 0.0], [0.0, 2.0]], requires_grad=True
    )
    labels = torch.tensor([0, 0, 1])
    return logits, labels


def _binding_model(class_count: int = 2):
    model = SimpleNamespace(id_backbone=torch.nn.Module())
    model.id_backbone.encoder = torch.nn.Linear(2, 2, bias=False)
    model.id_backbone.cls_head = torch.nn.Module()
    model.id_backbone.cls_head.head = torch.nn.Module()
    model.id_backbone.cls_head.head.weight = torch.nn.Parameter(torch.eye(class_count, 2))
    model.id_backbone.cls_head.imp_merge = torch.nn.Linear(2, 2)
    model.id_backbone.cls_head.dac_head = torch.nn.Linear(2, 1)
    model.id_backbone.cls_head.pa_head = torch.nn.Linear(2, 1)
    return model


def test_cb_sfce_is_present_tx_equal_focal_ce_with_no_clean_or_domain_input():
    logits, labels = _logits_and_labels()
    loss, info = cb_sfce_loss(logits, labels)
    per_row = (1.0 - logits.detach().softmax(dim=1).gather(1, labels[:, None]).squeeze(1)) * F.cross_entropy(
        logits.detach(), labels, reduction="none"
    )
    expected = 0.5 * (per_row[:2].mean() + per_row[2:].mean())
    assert torch.allclose(loss.detach(), expected, atol=1e-7)
    assert info["per_tx_rows"] == {"0": 2, "1": 1}
    assert info["present_tx_equal_aggregation"] is True
    assert info["single_satellite_logits_only"] is True
    signature = inspect.signature(cb_sfce_loss)
    assert "clean" not in " ".join(signature.parameters).lower()
    assert "rx" not in " ".join(signature.parameters).lower()
    assert "domain" not in " ".join(signature.parameters).lower()


def test_cb_sfce_has_live_satellite_gradient_and_is_label_permutation_equivariant():
    logits, labels = _logits_and_labels()
    direct, _ = cb_sfce_loss(logits, labels)
    direct.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0

    perm = torch.tensor([1, 0])
    permuted_logits = logits.detach()[:, perm].clone().requires_grad_(True)
    permuted, info = cb_sfce_loss(permuted_logits, perm[labels])
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-7)
    assert info["per_tx_rows"] == {"0": 1, "1": 2}


def test_cb_sfce_rejects_nonfinite_labels_outside_rows_and_gamma_drift():
    logits, labels = _logits_and_labels()
    bad = logits.detach().clone()
    bad[0, 0] = float("nan")
    with pytest.raises(CBSFCERuntimeError, match="non-finite"):
        cb_sfce_loss(bad.requires_grad_(True), labels)
    with pytest.raises(CBSFCERuntimeError, match="outside"):
        cb_sfce_loss(logits, torch.tensor([0, 0, 2]))
    with pytest.raises(CBSFCEConfigurationError, match="gamma"):
        cb_sfce_loss(logits, labels, gamma=2.0)


def test_local_four_binding_is_strict_and_identity_mapped_from_global_six():
    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_cb_sfce_local_head_class_binding(
        local_class_order=local_order,
        source_train_tx=local_order,
        checkpoint_train_tx=local_order,
        dataset_class_order=global_order,
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert binding["local_to_head_class_ids"] == [0, 1, 2, 3]
    assert torch.equal(
        remap_cb_sfce_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0, 2]),
    )
    with pytest.raises(CBSFCEConfigurationError, match="class counts"):
        resolve_cb_sfce_local_head_class_binding(
            local_class_order=local_order,
            source_train_tx=local_order,
            checkpoint_train_tx=local_order,
            dataset_class_order=global_order,
            local_data_class_count=4,
            checkpoint_head_class_count=6,
            live_head_class_count=6,
        )
    with pytest.raises(CBSFCEConfigurationError, match="checkpoint TX"):
        resolve_cb_sfce_local_head_class_binding(
            local_class_order=local_order,
            source_train_tx=local_order,
            checkpoint_train_tx=["20-19", "20-15", "6-15", "8-20"],
            dataset_class_order=global_order,
            local_data_class_count=4,
            checkpoint_head_class_count=4,
            live_head_class_count=4,
        )
    with pytest.raises(CBSFCEConfigurationError, match="exactly four"):
        resolve_cb_sfce_local_head_class_binding(
            local_class_order=local_order[:3],
            source_train_tx=local_order[:3],
            checkpoint_train_tx=local_order[:3],
            dataset_class_order=global_order,
            local_data_class_count=3,
            checkpoint_head_class_count=3,
            live_head_class_count=3,
        )


def test_binding_and_first_batch_raw_gradient_relation_cover_logits_encoder_and_head():
    model = _binding_model(4)
    encoder = model.id_backbone.encoder
    x_clean = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]])
    x_sat = torch.tensor([[0.5, 1.0], [1.0, 0.5], [0.7, 0.4], [0.4, 0.7]])
    labels = torch.tensor([0, 1, 2, 3])
    clean_logits = encoder(x_clean) @ model.id_backbone.cls_head.head.weight.t()
    sat_logits = encoder(x_sat) @ model.id_backbone.cls_head.head.weight.t()
    assert resolve_cb_sfce_classifier_weight(model) is model.id_backbone.cls_head.head.weight
    validate_cb_sfce_logit_binding(
        model=model,
        tx_logits=sat_logits,
        tx_labels=labels,
        expected_class_ids=[0, 1, 2, 3],
    )
    focal, _ = cb_sfce_loss(sat_logits, labels)
    relation = cb_sfce_shared_gradient_relation(
        F.cross_entropy(clean_logits, labels),
        focal,
        cb_sfce_shared_encoder_and_head_parameters(model),
        loss_weight=FROZEN_CB_SFCE_LAMBDA,
    )
    assert relation["raw_unscaled"] is True
    assert relation["diagnostic_only"] is True
    assert relation["shared_encoder"]["parameter_count"] == 1.0
    assert relation["classifier_head"]["parameter_count"] == 1.0
    assert relation["shared_encoder"]["norm_ratio"] >= 0.0
    with pytest.raises(CBSFCERuntimeError, match="gradient is missing"):
        cb_sfce_shared_gradient_relation(
            F.cross_entropy(clean_logits, labels),
            torch.tensor(1.0, requires_grad=True),
            cb_sfce_shared_encoder_and_head_parameters(model),
            loss_weight=FROZEN_CB_SFCE_LAMBDA,
        )
    with pytest.raises(CBSFCERuntimeError, match="gradient is non-finite"):
        cb_sfce_shared_gradient_relation(
            F.cross_entropy(clean_logits, labels),
            (model.id_backbone.encoder.weight * float("nan")).sum(),
            cb_sfce_shared_encoder_and_head_parameters(model),
            loss_weight=FROZEN_CB_SFCE_LAMBDA,
        )
    with pytest.raises(CBSFCERuntimeError, match="gradient norm is zero"):
        cb_sfce_shared_gradient_relation(
            (clean_logits * 0.0).sum(),
            focal,
            cb_sfce_shared_encoder_and_head_parameters(model),
            loss_weight=FROZEN_CB_SFCE_LAMBDA,
        )
    with pytest.raises(CBSFCERuntimeError, match="gradient norm is zero"):
        cb_sfce_shared_gradient_relation(
            F.cross_entropy(clean_logits, labels),
            (sat_logits * 0.0).sum(),
            cb_sfce_shared_encoder_and_head_parameters(model),
            loss_weight=FROZEN_CB_SFCE_LAMBDA,
        )
    with pytest.raises(CBSFCERuntimeError, match="live head"):
        validate_cb_sfce_logit_binding(
            model=model,
            tx_logits=torch.randn(4, 5, requires_grad=True),
            tx_labels=labels,
            expected_class_ids=[0, 1, 2, 3],
        )


def test_lite_d_source_only_forward_backward_smoke_has_no_query_input():
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(19)
    model = build_dual_model(
        num_classes=4,
        num_domains=1,
        dataset="wisig",
        input_len=128,
        model_variant="lite_d",
    ).train()
    x_clean = torch.randn(4, 2, 128)
    x_satellite = torch.randn(4, 2, 128)
    labels = torch.tensor([0, 1, 2, 3])
    out_clean = model(x_clean, y_tx=labels, domain_labels=None, return_aux=True)
    out_sat = model(x_satellite, y_tx=labels, domain_labels=None, return_aux=True)
    validate_cb_sfce_logit_binding(
        model=model,
        tx_logits=out_sat["tx_logits"],
        tx_labels=labels,
        expected_class_ids=[0, 1, 2, 3],
    )
    focal, _ = cb_sfce_loss(out_sat["tx_logits"], labels)
    base = F.cross_entropy(out_clean["tx_logits"], labels) + 0.10 * F.kl_div(
        F.log_softmax(out_sat["tx_logits"], dim=1),
        out_clean["tx_logits"].detach().softmax(dim=1),
        reduction="batchmean",
    )
    relation = cb_sfce_shared_gradient_relation(
        base,
        focal,
        cb_sfce_shared_encoder_and_head_parameters(model),
        loss_weight=FROZEN_CB_SFCE_LAMBDA,
    )
    (base + FROZEN_CB_SFCE_LAMBDA * focal).backward()
    assert relation["shared_encoder"]["parameter_count"] > 0.0
    assert relation["classifier_head"]["parameter_count"] == 1.0
    assert any(
        parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0.0
        for parameter in model.id_backbone.parameters()
    )


def test_terminal_receipt_requires_all_local_four_by_three_cells_and_audit():
    receipt = cb_sfce_config_receipt(CBSFCEConfig(True, True, FROZEN_CB_SFCE_LAMBDA, 1.0))
    receipt["expected_tx_class_ids"] = [0, 1, 2, 3]
    for scenario in FROZEN_CB_SFCE_SCENARIOS:
        logits = torch.eye(4, requires_grad=True) * 2.0
        labels = torch.tensor([0, 1, 2, 3])
        _, info = cb_sfce_loss(logits, labels)
        receipt = update_cb_sfce_receipt(receipt, info, scenario=scenario)
    with pytest.raises(CBSFCERuntimeError, match="first-batch"):
        validate_cb_sfce_terminal_receipt(receipt)
    relation = {
        "shared_encoder": {"parameter_count": 1.0, "base_norm": 1.0, "cb_sfce_norm": 0.2, "cosine": -0.1, "norm_ratio": 0.2},
        "classifier_head": {"parameter_count": 1.0, "base_norm": 1.0, "cb_sfce_norm": 0.3, "cosine": 0.1, "norm_ratio": 0.3},
        "raw_unscaled": True,
        "diagnostic_only": True,
    }
    none_cos = {
        **relation,
        "shared_encoder": {**relation["shared_encoder"], "cosine": None},
    }
    with pytest.raises(CBSFCERuntimeError, match="cosine is missing"):
        update_cb_sfce_gradient_relation_receipt(
            cb_sfce_config_receipt(CBSFCEConfig(True, True, FROZEN_CB_SFCE_LAMBDA, 1.0)),
            none_cos,
        )
    non_raw = {**relation, "raw_unscaled": False}
    with pytest.raises(CBSFCERuntimeError, match="raw_unscaled"):
        update_cb_sfce_gradient_relation_receipt(
            cb_sfce_config_receipt(CBSFCEConfig(True, True, FROZEN_CB_SFCE_LAMBDA, 1.0)),
            non_raw,
        )
    receipt = update_cb_sfce_gradient_relation_receipt(receipt, relation)
    terminal = validate_cb_sfce_terminal_receipt(receipt)
    assert terminal["cb_sfce_terminal_contract_passed"] is True
    assert len(terminal["cb_sfce_cells"]) == 12

    missing = cb_sfce_config_receipt(CBSFCEConfig(True, True, FROZEN_CB_SFCE_LAMBDA, 1.0))
    missing["expected_tx_class_ids"] = [0, 1, 2, 3]
    missing = update_cb_sfce_gradient_relation_receipt(missing, relation)
    with pytest.raises(CBSFCERuntimeError, match="coverage"):
        validate_cb_sfce_terminal_receipt(missing)


def test_control_is_identity_and_frozen_config_rejects_drift_and_stacking():
    base = torch.tensor(1.25, requires_grad=True)
    assert add_cb_sfce_to_loss(base, None, CBSFCEConfig(True, False, 0.0, 1.0)) is base
    config = validate_cb_sfce_args(_frozen_args())
    assert config.enabled is True and config.loss_weight == FROZEN_CB_SFCE_LAMBDA
    control = validate_cb_sfce_args(_frozen_args(enabled=False))
    assert control.enabled is False and control.loss_weight == 0.0
    receipt = cb_sfce_config_receipt(config)
    assert receipt["uses_clean_logits"] is False
    assert receipt["uses_teacher"] is False
    assert receipt["uses_explicit_z_alignment"] is False
    assert validate_cb_sfce_terminal_receipt(
        cb_sfce_config_receipt(CBSFCEConfig(True, False, 0.0, 1.0))
    )["cb_sfce_terminal_contract"] == "CONTROL_ARM_NOT_APPLICABLE"

    for name, value in (
        ("lambda_cb_sfce", 0.05),
        ("cb_sfce_gamma", 2.0),
        ("sat_train_scenarios", "leo_clear_weak,leo_rain_weak"),
        ("sat_view_schedule", "1:leo_clear_weak"),
        ("lambda_teacher_clean_kl", 0.01),
        ("teacher_ckpt", "teacher.pth"),
        ("phase1_pamr_enabled", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(CBSFCEConfigurationError, match=name.split("_")[0] if name == "sat_train_scenarios" else name):
            validate_cb_sfce_args(bad)


def test_strict_warm_start_and_best_effort_failure_receipt_do_not_mask(tmp_path, monkeypatch, capsys):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    warm = strict_cb_sfce_warm_start(
        target,
        source.state_dict(),
        baseline_path="geosat_c_final.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="final",
    )
    assert warm["strict_model_keys"] is True
    assert warm["optimizer_state_restored"] is False
    with pytest.raises(CBSFCEConfigurationError, match="strict baseline"):
        strict_cb_sfce_warm_start(
            target,
            {"weight": source.weight.detach().clone()},
            baseline_path="geosat_c_final.pth",
            baseline_sha256="a" * 64,
            checkpoint_epoch=40,
            checkpoint_role="final",
        )

    receipt = cb_sfce_config_receipt(CBSFCEConfig(True, True, 0.10, 1.0))
    original = CBSFCERuntimeError("P1-CB-SFCE shared_encoder gradient is non-finite")
    written = write_cb_sfce_failure_receipt(
        tmp_path,
        candidate_id="F1G",
        run_id="cb-sfce12",
        receipt=receipt,
        error=original,
        failure_stage="test",
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["error_fingerprint"] == "CB_SFCE_NONFINITE"
    assert "gradient is non-finite" not in written.read_text(encoding="utf-8")

    def writer_failure(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(train_ssdg, "write_cb_sfce_failure_receipt", writer_failure)
    caught = None
    try:
        try:
            raise original
        except CBSFCERuntimeError as error:
            train_ssdg._persist_cb_sfce_failure_receipt(
                out_dir=tmp_path,
                args=SimpleNamespace(candidate_id="F1G", run_id="cb-sfce12"),
                cb_sfce_receipt=receipt,
                error=error,
                failure_stage="test",
            )
            raise
    except CBSFCERuntimeError as error:
        caught = error
    assert caught is original
    assert "[P1-CB-SFCE-FAILURE-RECEIPT] persistence_failed writer_exception_type=OSError" in capsys.readouterr().out


def test_train_integration_has_one_first_batch_relation_and_global_round_robin_only():
    source = inspect.getsource(train_ssdg.train)
    assert "cb_sfce_satellite_step" in source
    assert "sat_train_scenarios = list(FROZEN_CB_SFCE_SCENARIOS)" in source
    start = source.index('if bool(getattr(cb_sfce_config, "enabled", False)):')
    end = source.index("pamr_audit_has_effective_batch", start)
    block = source[start:end]
    assert "update_cb_sfce_receipt" in block
    assert "not bool(cb_sfce_receipt.get(\"cb_sfce_gradient_relation_completed\", False))" in block
    assert "cb_sfce_shared_gradient_relation(" in block
    assert "out_l[\"tx_logits\"]" not in source[source.index("loss_cb_sfce_l, cb_sfce_batch_info"):source.index("if bool(getattr(ccpc_config", source.index("loss_cb_sfce_l, cb_sfce_batch_info"))]


def test_launcher_has_frozen_matrix_default_and_dry_run_closure():
    text = LAUNCHER.read_text(encoding="utf-8")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", text, flags=re.MULTILINE)
    assert calls == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    assert "phase1_cb_sfce12_20260809_v1" in text
    assert "GEOSAT_CKPT_ROOT:-${PROJECT_ROOT}/runs/phase1_loto_clsgeo12_20260808_v1" in text
    assert "--lambda_cb_sfce 0.10" in text and "--lambda_cb_sfce 0" in text
    assert "--lambda_sat_cons 0.10" in text
    assert "postfreeze" not in text.lower()
    completed = subprocess.run(
        ["bash", f"scripts/{LAUNCHER.name}", "--dry-run"],
        cwd=str(CODE_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.count("[DRY-RUN]") == 12
