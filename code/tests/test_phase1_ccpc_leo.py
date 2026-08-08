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
from SSDG.train_ssdg import _persist_ccpc_failure_receipt, build_arg_parser  # noqa: E402
from cvsrffi.phase1_ccpc_leo import (  # noqa: E402
    CCPCLEOConfig,
    CCPCLEOConfigurationError,
    CCPCLEORuntimeError,
    FROZEN_CCPC_LAMBDA,
    FROZEN_CCPC_TEMPERATURE,
    _contrastive_loss_from_positive_mask,
    add_ccpc_to_loss,
    ccpc_config_receipt,
    ccpc_leo_gradient_status,
    ccpc_leo_loss,
    ccpc_leo_unscaled_gradient,
    require_finite_ccpc_leo_gradient,
    strict_ccpc_warm_start,
    update_ccpc_receipt,
    update_ccpc_optimizer_receipt,
    validate_ccpc_terminal_receipt,
    validate_ccpc_leo_args,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_ccpc_leo12_20260809.sh"
AUDIT_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_ccpc_leo_gradient_audit6_20260809.sh"


def _frozen_args(*, enabled: bool, gradient_audit_only: bool = False, epochs: int = 40) -> object:
    argv = [
        "--output_dir",
        "unused",
        "--baseline_ckpt",
        "geosat_c_final.pth",
        "--from_scratch",
        "false",
        "--freeze_backbone",
        "false",
        "--epochs",
        str(epochs),
        "--label_epochs",
        str(epochs),
        "--pseudo_epochs",
        "0",
        "--phase1_ccpc_leo_frozen_mode",
        "true",
        "--phase1_ccpc_leo_enabled",
        str(enabled).lower(),
        "--phase1_ccpc_leo_gradient_audit_only",
        str(gradient_audit_only).lower(),
        "--lambda_ccpc_leo",
        "0.02" if enabled else "0",
        "--ccpc_leo_temperature",
        "0.12",
        "--lambda_sat_cls",
        "0",
        "--lambda_sat_cons",
        "0.10",
        "--lambda_domain",
        "0",
        "--lambda_adv",
        "0",
        "--lambda_orth",
        "0",
        "--lambda_cons",
        "0",
        "--lambda_group_ce",
        "0",
        "--lambda_fishr",
        "0",
        "--lambda_u",
        "0",
        "--lambda_ent",
        "0",
        "--lambda_u_domain",
        "0",
        "--lambda_u_adv",
        "0",
        "--lambda_u_sat_cons",
        "0",
        "--lambda_u_direct_metric_accept",
        "0",
        "--lambda_u_quarantine_accept",
        "0",
        "--use_unlabeled",
        "false",
        "--use_aug",
        "false",
        "--use_mixstyle",
        "false",
        "--use_tx_rx_balanced_sampler",
        "false",
        "--use_phase2_ground_prototypes",
        "false",
        "--use_feature_masks",
        "false",
        "--use_txrx_geometry_losses",
        "false",
        "--use_proto_memory",
        "false",
        "--reject_head",
        "false",
        "--checkpoint_selection",
        "final_only",
        "--phase1_source_val_selection_only",
        "true",
        "--phase1_source_train_tx_ids",
        "20-15,20-19,6-15,8-20",
        "--phase1_source_known_validation_tx_ids",
        "14-7",
        "--phase1_source_proxy_unknown_tx_ids",
        "14-10",
    ]
    return build_arg_parser().parse_args(argv)


def _paired_features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean = torch.tensor(
        [[1.0, 0.0, 0.1], [0.9, 0.1, 0.0], [-1.0, 0.0, -0.1], [-0.9, -0.1, 0.0]],
        requires_grad=True,
    )
    leo = torch.tensor(
        [[0.8, 0.2, 0.1], [0.7, 0.3, 0.0], [-0.8, -0.2, -0.1], [-0.7, -0.3, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([3, 3, 8, 8])
    return clean, leo, labels


def test_ccpc_uses_same_tx_positives_and_all_clean_rows_as_denominator():
    clean, leo, labels = _paired_features()
    loss, info = ccpc_leo_loss(leo, clean, labels)

    assert torch.isfinite(loss)
    assert info["rows"] == 4
    assert info["classes"] == 2
    assert info["positive_pairs"] == 8
    assert info["positive_anchors"] == 4
    assert info["clean_detached"] is True
    assert info["temperature"] == FROZEN_CCPC_TEMPERATURE


def test_ccpc_detaches_clean_but_backpropagates_nonzero_gradient_to_leo():
    clean, leo, labels = _paired_features()
    loss, _ = ccpc_leo_loss(leo, clean, labels)
    loss.backward()

    assert clean.grad is None
    assert leo.grad is not None
    assert float(leo.grad.abs().sum()) > 0.0


def test_ccpc_accepts_a_finite_zero_gradient_at_a_legal_stationary_point():
    clean = torch.ones((4, 8), requires_grad=True)
    leo = torch.ones((4, 8), requires_grad=True)
    labels = torch.tensor([3, 3, 8, 8])
    loss, info = ccpc_leo_loss(leo, clean, labels)
    loss.backward()

    status = ccpc_leo_gradient_status(leo.grad)
    assert status == {"finite": True, "nonzero": False, "zero": True, "nonfinite": False}
    require_finite_ccpc_leo_gradient(status)
    receipt = update_ccpc_receipt(
        ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12)),
        info,
        leo_grad_nonzero=status["nonzero"],
        leo_grad_zero=status["zero"],
        leo_grad_nonfinite=status["nonfinite"],
    )
    assert receipt["ccpc_grad_zero_batches"] == 1
    assert receipt["ccpc_grad_nonzero_batches"] == 0


def test_unscaled_ccpc_gradient_audit_ignores_simulated_scaled_intermediate_overflow():
    clean, leo, labels = _paired_features()
    loss, _ = ccpc_leo_loss(leo, clean, labels)
    raw_gradient = ccpc_leo_unscaled_gradient(
        loss,
        leo,
        loss_weight=FROZEN_CCPC_LAMBDA,
    )
    raw_status = ccpc_leo_gradient_status(raw_gradient)
    assert raw_status["finite"] is True
    assert raw_status["nonzero"] is True
    require_finite_ccpc_leo_gradient(raw_status)

    # A retained intermediate after GradScaler.scale(loss).backward() can
    # overflow even when this unscaled autograd.grad result is finite.
    max_abs = float(raw_gradient.detach().abs().max().item())
    factor = torch.tensor(
        (float(torch.finfo(torch.float32).max) / max_abs) * 2.0,
        dtype=torch.float64,
    )
    scaled_emulation = (raw_gradient.detach().to(torch.float64) * factor).to(torch.float32)
    assert ccpc_leo_gradient_status(scaled_emulation)["nonfinite"] is True


def test_unscaled_ccpc_gradient_audit_none_and_nonfinite_remain_fail_closed():
    with pytest.raises(CCPCLEORuntimeError, match="no LEO feature gradient"):
        ccpc_leo_gradient_status(None)
    raw_nonfinite = ccpc_leo_gradient_status(torch.tensor([float("nan")]))
    assert raw_nonfinite["nonfinite"] is True
    with pytest.raises(CCPCLEORuntimeError, match="gradient is non-finite"):
        require_finite_ccpc_leo_gradient(raw_nonfinite)


def test_ccpc_gradient_none_and_nonfinite_remain_fail_closed_and_receipted():
    with pytest.raises(CCPCLEORuntimeError, match="no LEO feature gradient"):
        ccpc_leo_gradient_status(None)

    status = ccpc_leo_gradient_status(torch.tensor([float("nan")]))
    assert status == {"finite": False, "nonzero": False, "zero": False, "nonfinite": True}
    with pytest.raises(CCPCLEORuntimeError, match="gradient is non-finite"):
        require_finite_ccpc_leo_gradient(status)
    receipt = update_ccpc_receipt(
        ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12)),
        {"rows": 4, "classes": 2, "positive_pairs": 8, "clean_detached": True},
        leo_grad_nonzero=status["nonzero"],
        leo_grad_zero=status["zero"],
        leo_grad_nonfinite=status["nonfinite"],
    )
    assert receipt["ccpc_grad_nonfinite_batches"] == 1
    with pytest.raises(CCPCLEORuntimeError, match="rejects non-finite"):
        validate_ccpc_terminal_receipt(receipt)


def test_train_side_ccpc_gradient_failures_atomically_persist_data_free_receipts(tmp_path):
    args = SimpleNamespace(candidate_id="F1G", run_id="phase1_ccpc_leo12_v3")
    receipt = ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12))
    nonfinite = ccpc_leo_gradient_status(torch.tensor([float("nan")]))
    receipt = update_ccpc_receipt(
        receipt,
        {"rows": 4, "classes": 2, "positive_pairs": 8, "clean_detached": True},
        leo_grad_nonzero=nonfinite["nonzero"],
        leo_grad_zero=nonfinite["zero"],
        leo_grad_nonfinite=nonfinite["nonfinite"],
    )
    failure_path = _persist_ccpc_failure_receipt(
        out_dir=tmp_path,
        args=args,
        ccpc_receipt=receipt,
        error=CCPCLEORuntimeError(
            "CCPC-LEO fail-closed: paired LEO feature gradient is non-finite"
        ),
    )
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure_path == tmp_path / "ccpc_failure_receipt.json"
    assert payload["schema"] == "cvs.phase1.ccpc_leo_failure_receipt.v1"
    assert payload["candidate_id"] == "F1G"
    assert payload["run_id"] == "phase1_ccpc_leo12_v3"
    assert payload["error_fingerprint"] == "CCPC_LEO_GRADIENT_NONFINITE"
    assert payload["ccpc_receipt"]["ccpc_grad_nonfinite_batches"] == 1
    assert "raw" not in json.dumps(payload, ensure_ascii=False).lower()
    assert not list(tmp_path.glob(".ccpc_failure_receipt.*.tmp"))

    missing_path = _persist_ccpc_failure_receipt(
        out_dir=tmp_path,
        args=args,
        ccpc_receipt=ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12)),
        error=CCPCLEORuntimeError(
            "CCPC-LEO fail-closed: no LEO feature gradient reached the paired loss"
        ),
    )
    missing_payload = json.loads(missing_path.read_text(encoding="utf-8"))
    assert missing_payload["error_fingerprint"] == "CCPC_LEO_GRADIENT_MISSING"


def test_ccpc_failure_receipt_writer_error_never_masks_original_gradient_failure(
    tmp_path, monkeypatch, capsys
):
    original = CCPCLEORuntimeError(
        "CCPC-LEO fail-closed: paired LEO feature gradient is non-finite"
    )

    def _writer_failure(*_args, **_kwargs):
        raise OSError("simulated receipt writer failure")

    monkeypatch.setattr(train_ssdg, "write_ccpc_failure_receipt", _writer_failure)
    with pytest.raises(CCPCLEORuntimeError) as caught:
        try:
            raise original
        except CCPCLEORuntimeError as error:
            persisted = _persist_ccpc_failure_receipt(
                out_dir=tmp_path,
                args=SimpleNamespace(candidate_id="F1G", run_id="phase1_ccpc_leo12_v3"),
                ccpc_receipt=ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12)),
                error=error,
            )
            assert persisted is None
            raise

    marker = capsys.readouterr().out
    assert caught.value is original
    assert str(caught.value) == str(original)
    assert "[CCPC-LEO-FAILURE-RECEIPT] persistence_failed writer_exception_type=OSError" in marker
    assert str(original) not in marker
    assert str(tmp_path) not in marker
    assert not (tmp_path / "ccpc_failure_receipt.json").exists()


def test_ccpc_terminal_receipt_requires_nonzero_and_no_nonfinite_batches():
    base = ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12))
    zero_only = update_ccpc_receipt(
        base,
        {"rows": 4, "classes": 2, "positive_pairs": 8, "clean_detached": True},
        leo_grad_nonzero=False,
        leo_grad_zero=True,
        leo_grad_nonfinite=False,
    )
    with pytest.raises(CCPCLEORuntimeError, match="at least one nonzero"):
        validate_ccpc_terminal_receipt(zero_only)

    passed = update_ccpc_receipt(
        zero_only,
        {"rows": 4, "classes": 2, "positive_pairs": 8, "clean_detached": True},
        leo_grad_nonzero=True,
        leo_grad_zero=False,
        leo_grad_nonfinite=False,
    )
    passed = update_ccpc_optimizer_receipt(
        passed,
        parameter_grad_finite=True,
        optimizer_step_applied=True,
    )
    passed = update_ccpc_optimizer_receipt(
        passed,
        parameter_grad_finite=True,
        optimizer_step_applied=True,
    )
    terminal = validate_ccpc_terminal_receipt(passed)
    assert terminal["ccpc_terminal_gradient_contract_passed"] is True
    assert terminal["ccpc_terminal_gradient_contract"] == (
        "NONZERO_OBSERVED_NO_NONFINITE_PARAM_FINITE_AND_STEP_OBSERVED"
    )
    assert terminal["ccpc_param_grad_finite_batches"] == 2
    assert terminal["ccpc_optimizer_step_applied_batches"] == 2


def test_ccpc_terminal_receipt_requires_a_finite_parameter_batch_and_step_but_allows_scaler_skip():
    base = ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12))
    single = update_ccpc_receipt(
        base,
        {"rows": 4, "classes": 2, "positive_pairs": 8, "clean_detached": True},
        leo_grad_nonzero=True,
        leo_grad_zero=False,
        leo_grad_nonfinite=False,
    )
    no_finite_parameter = update_ccpc_optimizer_receipt(
        single,
        parameter_grad_finite=False,
        optimizer_step_applied=False,
    )
    with pytest.raises(CCPCLEORuntimeError, match="finite parameter-gradient"):
        validate_ccpc_terminal_receipt(no_finite_parameter)

    no_step = update_ccpc_optimizer_receipt(
        single,
        parameter_grad_finite=True,
        optimizer_step_applied=False,
    )
    with pytest.raises(CCPCLEORuntimeError, match="at least one optimizer step"):
        validate_ccpc_terminal_receipt(no_step)

    with_legal_scaler_skip = update_ccpc_receipt(
        single,
        {"rows": 4, "classes": 2, "positive_pairs": 8, "clean_detached": True},
        leo_grad_nonzero=False,
        leo_grad_zero=True,
        leo_grad_nonfinite=False,
    )
    with_legal_scaler_skip = update_ccpc_optimizer_receipt(
        with_legal_scaler_skip,
        parameter_grad_finite=True,
        optimizer_step_applied=True,
    )
    with_legal_scaler_skip = update_ccpc_optimizer_receipt(
        with_legal_scaler_skip,
        parameter_grad_finite=False,
        optimizer_step_applied=False,
    )
    terminal = validate_ccpc_terminal_receipt(with_legal_scaler_skip)
    assert terminal["ccpc_param_grad_nonfinite_batches"] == 1
    assert terminal["ccpc_optimizer_step_not_applied_batches"] == 1
    assert terminal["ccpc_terminal_gradient_contract_passed"] is True


def test_ccpc_optimizer_receipt_records_parameter_gradient_and_step_outcomes():
    receipt = ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12))
    receipt = update_ccpc_optimizer_receipt(
        receipt,
        parameter_grad_finite=True,
        optimizer_step_applied=True,
    )
    receipt = update_ccpc_optimizer_receipt(
        receipt,
        parameter_grad_finite=False,
        optimizer_step_applied=False,
    )
    assert receipt["ccpc_param_grad_finite_batches"] == 1
    assert receipt["ccpc_param_grad_nonfinite_batches"] == 1
    assert receipt["ccpc_optimizer_step_applied_batches"] == 1
    assert receipt["ccpc_optimizer_step_not_applied_batches"] == 1
    with pytest.raises(CCPCLEORuntimeError, match="cannot record an optimizer step"):
        update_ccpc_optimizer_receipt(
            receipt,
            parameter_grad_finite=False,
            optimizer_step_applied=True,
        )


def test_ccpc_is_label_permutation_equivariant():
    clean, leo, labels = _paired_features()
    direct, direct_info = ccpc_leo_loss(leo, clean, labels)
    permuted, permuted_info = ccpc_leo_loss(leo, clean, torch.tensor([41, 41, 12, 12]))

    assert torch.equal(direct.detach(), permuted.detach())
    assert direct_info["positive_pairs"] == permuted_info["positive_pairs"]


def test_ccpc_does_not_accept_rx_or_domain_metadata():
    signature = inspect.signature(ccpc_leo_loss)
    assert "rx" not in " ".join(signature.parameters).lower()
    assert "domain" not in " ".join(signature.parameters).lower()
    receipt = ccpc_config_receipt(CCPCLEOConfig(True, True, 0.02, 0.12))
    assert receipt["uses_rx_labels"] is False
    assert receipt["uses_domain_labels"] is False
    assert receipt["uses_grl"] is False


def test_ccpc_fails_closed_for_single_class_empty_positive_or_nonfinite_inputs():
    clean, leo, _ = _paired_features()
    with pytest.raises(CCPCLEORuntimeError, match="at least two TX"):
        ccpc_leo_loss(leo, clean, torch.tensor([1, 1, 1, 1]))
    with pytest.raises(CCPCLEORuntimeError, match="at least one same-TX"):
        _contrastive_loss_from_positive_mask(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool))
    bad = leo.detach().clone()
    bad[0, 0] = float("nan")
    with pytest.raises(CCPCLEORuntimeError, match="non-finite"):
        ccpc_leo_loss(bad.requires_grad_(True), clean, torch.tensor([3, 3, 8, 8]))


def test_control_path_returns_the_identical_base_tensor_without_a_ccpc_zero_addition():
    base = torch.tensor(1.25, requires_grad=True)
    control = CCPCLEOConfig(frozen_mode=True, enabled=False, loss_weight=0.0, temperature=0.12)
    output = add_ccpc_to_loss(base, None, control)

    assert output is base
    assert torch.equal(output, base)


def test_cli_enforces_frozen_lambda_temperature_and_forbidden_stacking():
    active = _frozen_args(enabled=True)
    config = validate_ccpc_leo_args(active)
    assert config.enabled is True
    assert config.loss_weight == FROZEN_CCPC_LAMBDA
    assert config.temperature == FROZEN_CCPC_TEMPERATURE

    bad_weight = _frozen_args(enabled=True)
    bad_weight.lambda_ccpc_leo = 0.01
    with pytest.raises(CCPCLEOConfigurationError, match="lambda_ccpc_leo"):
        validate_ccpc_leo_args(bad_weight)

    bad_proxy = _frozen_args(enabled=True)
    bad_proxy.lambda_proxy_unknown = 0.01
    with pytest.raises(CCPCLEOConfigurationError, match="lambda_proxy_unknown"):
        validate_ccpc_leo_args(bad_proxy)

    bad_unlabeled = _frozen_args(enabled=True)
    bad_unlabeled.use_unlabeled = True
    with pytest.raises(CCPCLEOConfigurationError, match="forbids unlabeled"):
        validate_ccpc_leo_args(bad_unlabeled)


def test_cli_forbids_all_teacher_routes_and_teacher_checkpoint():
    for name, value in (
        ("lambda_teacher_clean_kl", 0.01),
        ("lambda_teacher_sat_kl", 0.01),
        ("lambda_teacher_zid_mse", 0.01),
        ("teacher_ckpt", "teacher.pth"),
    ):
        bad = _frozen_args(enabled=True)
        setattr(bad, name, value)
        with pytest.raises(CCPCLEOConfigurationError, match=name):
            validate_ccpc_leo_args(bad)


def test_gradient_audit_mode_is_g_only_and_exactly_15_epochs():
    audit = _frozen_args(enabled=True, gradient_audit_only=True, epochs=15)
    config = validate_ccpc_leo_args(audit)
    assert config.enabled is True
    assert config.gradient_audit_only is True
    receipt = ccpc_config_receipt(config)
    assert receipt["technical_only"] is True
    assert receipt["performance_result_available"] is False
    assert receipt["technical_only_claim"] == "NO_PERFORMANCE_RESULT"

    not_g = _frozen_args(enabled=False, gradient_audit_only=True, epochs=15)
    with pytest.raises(CCPCLEOConfigurationError, match="requires the frozen enabled G arm"):
        validate_ccpc_leo_args(not_g)

    wrong_length = _frozen_args(enabled=True, gradient_audit_only=True, epochs=40)
    with pytest.raises(CCPCLEOConfigurationError, match="--epochs 15"):
        validate_ccpc_leo_args(wrong_length)

    ordinary = _frozen_args(enabled=True, gradient_audit_only=False, epochs=15)
    with pytest.raises(CCPCLEOConfigurationError, match="--epochs 40"):
        validate_ccpc_leo_args(ordinary)


def test_trainer_uses_unscaled_ccpc_gradient_before_gradscaler_backward():
    source = inspect.getsource(train_ssdg.train)
    raw_audit = source.index("ccpc_leo_unscaled_gradient(")
    scaled_backward = source.index("scaler.scale(loss).backward()")
    assert raw_audit < scaled_backward
    assert "ccpc_leo_feature.retain_grad()" not in source
    assert source.index("scaler.unscale_(optimizer)") > scaled_backward
    assert "update_ccpc_optimizer_receipt(" in source


def test_gradient_audit_skips_heldout_evaluator_and_uses_fixed_receipt(monkeypatch):
    def _heldout_must_not_run(*_args, **_kwargs):
        raise AssertionError("heldout evaluator must be unreachable in technical audit mode")

    monkeypatch.setattr(
        train_ssdg,
        "_evaluate_frozen_phase1_checkpoint",
        _heldout_must_not_run,
    )
    frozen_eval = train_ssdg._resolve_frozen_phase1_evaluation(
        SimpleNamespace(),
        object(),
        {},
        object(),
        "unused_checkpoint.pth",
        technical_only=True,
        selection_source="training_final_only",
    )
    assert frozen_eval == {
        "status": "SKIPPED_TECHNICAL_AUDIT",
        "selection_source": "TECHNICAL_ONLY",
        "claim": "NO_PERFORMANCE_RESULT",
    }


def test_gradient_audit_terminal_manifest_path_is_nonpromotable_and_has_no_heldout_call():
    status = train_ssdg._resolve_phase1_terminal_status(
        tail_stopped=False,
        export_failed=False,
        final_blocked=True,
        selected_checkpoint_exists=True,
        heldout_eval_status="FAILED",
        p0_mechanisms_ready=False,
        p1_mechanisms_ready=False,
        endpoint_export_ready=False,
        technical_only=True,
    )
    assert status == "TECHNICAL_AUDIT_COMPLETE"

    source = inspect.getsource(train_ssdg.train)
    assert "_evaluate_frozen_phase1_checkpoint(" not in source
    assert "_resolve_frozen_phase1_evaluation(" in source
    assert "technical_only=ccpc_gradient_audit_only" in source
    assert re.search(
        r'"promotion_ready":\s*\(\s*False\s*if ccpc_gradient_audit_only',
        source,
    )
    assert re.search(
        r'"claim":\s*\(\s*"NO_PERFORMANCE_RESULT"\s*if ccpc_gradient_audit_only',
        source,
    )
    assert '"technical_only": bool(ccpc_gradient_audit_only)' in source
    ccpc_terminal_receipt_source = source[
        source.index('"phase1_ccpc_leo_terminal_receipt.json"') : source.index(
            "completion_receipt ="
        )
    ]
    assert '"technical_only": bool(ccpc_gradient_audit_only)' in ccpc_terminal_receipt_source
    assert re.search(
        r'"promotion_ready":\s*\(\s*False\s*if ccpc_gradient_audit_only',
        ccpc_terminal_receipt_source,
    )
    assert '"NO_PERFORMANCE_RESULT"' in ccpc_terminal_receipt_source


def test_strict_warm_start_proves_exact_keys_and_fresh_optimizer_rng_receipt():
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    receipt = strict_ccpc_warm_start(
        target,
        source.state_dict(),
        baseline_path="geosat_c_final.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="final",
    )
    assert receipt["strict_model_keys"] is True
    assert receipt["missing_model_keys"] == []
    assert receipt["unexpected_model_keys"] == []
    assert receipt["baseline_path"] == "geosat_c_final.pth"
    assert receipt["baseline_sha256"] == "a" * 64
    assert receipt["checkpoint_epoch"] == 40
    assert receipt["checkpoint_role"] == "final"
    assert receipt["optimizer_state_restored"] is False
    assert receipt["rng_state_restored"] is False
    assert receipt["warm_start_mode"] == "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP"
    assert torch.equal(target.weight, source.weight)

    missing = dict(source.state_dict())
    missing.pop("bias")
    with pytest.raises(CCPCLEOConfigurationError, match="strict baseline model-key mismatch"):
        strict_ccpc_warm_start(
            torch.nn.Linear(3, 2),
            missing,
            baseline_path="geosat_c_final.pth",
            baseline_sha256="b" * 64,
            checkpoint_epoch=40,
            checkpoint_role="final",
        )
    unexpected = dict(source.state_dict())
    unexpected["unexpected"] = torch.ones(1)
    with pytest.raises(CCPCLEOConfigurationError, match="strict baseline model-key mismatch"):
        strict_ccpc_warm_start(
            torch.nn.Linear(3, 2),
            unexpected,
            baseline_path="geosat_c_final.pth",
            baseline_sha256="c" * 64,
            checkpoint_epoch=40,
            checkpoint_role="final",
        )


def test_launcher_has_the_frozen_12_row_matrix_and_dry_run_closure():
    text = LAUNCHER.read_text(encoding="utf-8")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", text, flags=re.MULTILINE)
    assert calls == [
        ("1", "C", "0"),
        ("5", "G", "0"),
        ("1", "G", "1"),
        ("5", "C", "1"),
        ("2", "C", "2"),
        ("6", "G", "2"),
        ("2", "G", "3"),
        ("6", "C", "3"),
        ("3", "C", "4"),
        ("3", "G", "5"),
        ("4", "C", "6"),
        ("4", "G", "7"),
    ]
    for required in (
        "phase1_loto_clsgeo12_20260808_v1",
        "--epochs 40",
        "--checkpoint_selection final_only",
        "--phase1_ccpc_leo_frozen_mode true",
        "--lambda_sat_cons 0.10",
        "--lambda_domain 0",
        "--lambda_proxy_unknown 0",
        "--teacher_ckpt \"\"",
        "--lambda_teacher_clean_kl 0",
        "--lambda_teacher_sat_kl 0",
        "--lambda_teacher_zid_mse 0",
        "--reject_head false",
    ):
        assert required in text
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_ccpc_leo12_20260809.sh", "--dry-run"],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    dry_lines = [line for line in completed.stdout.splitlines() if line.startswith("[DRY-RUN]")]
    assert len(dry_lines) == 12
    assert all("--epochs 40" in line and "--checkpoint_selection final_only" in line for line in dry_lines)


def test_gradient_audit_launcher_has_six_g_only_rows_and_dry_run_closure():
    text = AUDIT_LAUNCHER.read_text(encoding="utf-8")
    calls = re.findall(r"^launch_fold (\d) (\d)$", text, flags=re.MULTILINE)
    assert calls == [(str(index), str(index - 1)) for index in range(1, 7)]
    for required in (
        "phase1_loto_clsgeo12_20260808_v1",
        "--epochs 15",
        "--label_epochs 15",
        "--phase1_ccpc_leo_enabled true",
        "--phase1_ccpc_leo_gradient_audit_only true",
        "--lambda_ccpc_leo 0.02",
        "--ccpc_leo_temperature 0.12",
        "--checkpoint_selection final_only",
        "--amp true",
        "--teacher_ckpt \"\"",
    ):
        assert required in text
    assert "launch_arm" not in text
    assert "eval_phase1_ccpc_leo_pair.py" not in text
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_ccpc_leo_gradient_audit6_20260809.sh", "--dry-run"],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    dry_lines = [line for line in completed.stdout.splitlines() if line.startswith("[DRY-RUN]")]
    assert len(dry_lines) == 6
    assert all(
        "--epochs 15" in line
        and "--phase1_ccpc_leo_enabled true" in line
        and "--phase1_ccpc_leo_gradient_audit_only true" in line
        and "--checkpoint_selection final_only" in line
        for line in dry_lines
    )
