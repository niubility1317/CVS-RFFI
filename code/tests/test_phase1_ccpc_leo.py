from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import build_arg_parser  # noqa: E402
from cvsrffi.phase1_ccpc_leo import (  # noqa: E402
    CCPCLEOConfig,
    CCPCLEOConfigurationError,
    CCPCLEORuntimeError,
    FROZEN_CCPC_LAMBDA,
    FROZEN_CCPC_TEMPERATURE,
    _contrastive_loss_from_positive_mask,
    add_ccpc_to_loss,
    ccpc_config_receipt,
    ccpc_leo_loss,
    strict_ccpc_warm_start,
    validate_ccpc_leo_args,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_ccpc_leo12_20260809.sh"


def _frozen_args(*, enabled: bool) -> object:
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
        "40",
        "--label_epochs",
        "40",
        "--pseudo_epochs",
        "0",
        "--phase1_ccpc_leo_frozen_mode",
        "true",
        "--phase1_ccpc_leo_enabled",
        str(enabled).lower(),
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
