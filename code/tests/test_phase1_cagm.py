from __future__ import annotations

import inspect
import json
import math
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
from cvsrffi.phase1_cagm import (  # noqa: E402
    FROZEN_CAGM_CLASS_IDS,
    FROZEN_CAGM_LAMBDA,
    FROZEN_CAGM_SCENARIOS,
    CAGMConfig,
    CAGMConfigurationError,
    CAGMRuntimeError,
    add_cagm_to_loss,
    bind_cagm_optimizer_initial_state,
    bind_cagm_source_data_order,
    cagm_aux_gradient_audit,
    cagm_config_receipt,
    cagm_loss,
    cagm_shared_encoder_and_head_parameters,
    remap_cagm_local_labels_to_head_rows,
    resolve_cagm_classifier_weight,
    resolve_cagm_local_head_class_binding,
    strict_cagm_warm_start,
    update_cagm_common_batch_sequence_receipt,
    update_cagm_gradient_audit_receipt,
    update_cagm_receipt,
    validate_cagm_args,
    validate_cagm_binding,
    validate_cagm_terminal_receipt,
    write_cagm_failure_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_cagm12_20260810.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        phase1_cagm_frozen_mode=True,
        phase1_cagm_enabled=enabled,
        lambda_cagm=FROZEN_CAGM_LAMBDA if enabled else 0.0,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        id_feature_key="feat_joint",
        epochs=epochs,
        label_epochs=epochs,
        pseudo_epochs=0,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        sat_cons_start_epoch=1,
        sat_view_prob=1.0,
        sat_train_scenarios=",".join(FROZEN_CAGM_SCENARIOS),
        sat_view_schedule="",
        use_concat_sat_channel_aug=False,
        use_unlabeled=False,
        use_tx_rx_balanced_sampler=False,
        use_aug=False,
        use_mixstyle=False,
        reject_head=False,
        phase1_ccpc_leo_frozen_mode=False,
        phase1_ccpc_leo_enabled=False,
        lambda_ccpc_leo=0.0,
        phase1_pamr_frozen_mode=False,
        phase1_pamr_enabled=False,
        lambda_pamr=0.0,
        phase1_cb_sfce_frozen_mode=False,
        phase1_cb_sfce_enabled=False,
        lambda_cb_sfce=0.0,
        phase1_gd_proto_nll_frozen_mode=False,
        phase1_gd_proto_nll_enabled=False,
        lambda_gd_proto_nll=0.0,
        phase1_icmt_frozen_mode=False,
        phase1_icmt_enabled=False,
        lambda_icmt=0.0,
        phase1_cp_sfce_frozen_mode=False,
        phase1_cp_sfce_enabled=False,
        lambda_cp_sfce=0.0,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )


def _binding_model() -> torch.nn.Module:
    class _BindingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_backbone = torch.nn.Module()
            self.id_backbone.encoder = torch.nn.Linear(4, 4, bias=False)
            self.id_backbone.cls_head = torch.nn.Module()
            self.id_backbone.cls_head.joint_proj = torch.nn.Linear(4, 4, bias=False)
            self.id_backbone.cls_head.head = torch.nn.Linear(4, 4, bias=False)
            self.id_backbone.cls_head.imp_merge = torch.nn.Linear(4, 4)
            self.id_backbone.cls_head.dac_head = torch.nn.Linear(4, 1)
            self.id_backbone.cls_head.pa_head = torch.nn.Linear(4, 1)

        def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
            z_id = self.id_backbone.cls_head.joint_proj(self.id_backbone.encoder(x))
            return {
                "z_id": z_id,
                "z_id_key": "feat_joint",
                "tx_logits": self.id_backbone.cls_head.head(z_id),
            }

    return _BindingModel()


def _geometry_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Two rows per local class.  The first two classes have clean within-class
    # angular radius; all values are hand-checkable in the test below.
    clean = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    leo = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return clean, leo, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = cagm_config_receipt(CAGMConfig(True, enabled, 0.02 if enabled else 0.0))
    receipt.update(
        {
            "baseline_sha256": "a" * 64,
            "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": "b" * 64,
            "source_labeled_indices_sha256": "c" * 64,
            "source_split_manifest_sha256": "d" * 64,
            "optimizer_initial_state_sha256": "e" * 64,
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "expected_tx_class_ids": [0, 1, 2, 3],
            "common_batch_sequence_sha256": "f" * 64,
            "common_batch_sequence_batches": 3,
            "common_batch_sequence_rows": 24,
            "common_scenario_batches": {scenario: 1 for scenario in FROZEN_CAGM_SCENARIOS},
        }
    )
    return receipt


def _audit() -> dict[str, object]:
    return {
        "shared_encoder": {"parameter_count": 2.0, "norm": 0.2},
        "classifier_head": {
            "parameter_count": 1.0,
            "none_parameters": 1.0,
            "zero_parameters": 0.0,
            "nonzero_parameters": 0.0,
            "none_or_zero_expected": True,
        },
        "raw_unscaled": True,
        "diagnostic_only": True,
    }


def test_formula_is_manual_fixed_ten_term_and_clean_is_detached() -> None:
    clean, leo, labels = _geometry_inputs()
    clean = clean.requires_grad_(True)
    leo = leo.requires_grad_(True)
    loss, info = cagm_loss(clean, leo, labels)
    root_half = 1.0 / math.sqrt(2.0)
    # r_clean=(1-1/sqrt(2),1-1/sqrt(2),0,0), r_leo=(0,0,0,0).
    # The only nonzero Gram deltas are 0-2=-1/sqrt(2),
    # 0-3=1-1/sqrt(2), plus the two radius deltas.
    expected = (
        2.0 * (1.0 - root_half) ** 2
        + root_half**2
        + (1.0 - root_half) ** 2
    ) / 10.0
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    assert info["loss_divisor"] == 10
    assert info["clean_statistics_detached"] is True
    assert info["radius_delta"] == pytest.approx(
        {"tx0": -(1.0 - root_half), "tx1": -(1.0 - root_half), "tx2": 0.0, "tx3": 0.0},
        abs=1e-6,
    )
    assert len(info["gram_delta"]) == 6
    loss.backward()
    assert clean.grad is None
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) > 0


def test_label_permutation_common_rotation_invariance_and_nonisometry_response() -> None:
    clean, leo, labels = _geometry_inputs()
    direct, _ = cagm_loss(clean.requires_grad_(True), leo.requires_grad_(True), labels)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _ = cagm_loss(
        clean.clone().requires_grad_(True),
        leo.clone().requires_grad_(True),
        permutation[labels],
    )
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-6)

    torch.manual_seed(17)
    orthogonal, _ = torch.linalg.qr(torch.randn(4, 4))
    rotated, _ = cagm_loss(
        (clean @ orthogonal).requires_grad_(True),
        (leo @ orthogonal).requires_grad_(True),
        labels,
    )
    assert torch.allclose(direct.detach(), rotated.detach(), atol=1e-6)

    nonisometry = torch.tensor(
        [
            [1.0, 0.35, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.25],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    distorted, _ = cagm_loss(
        (clean @ nonisometry).requires_grad_(True),
        (leo @ nonisometry).requires_grad_(True),
        labels,
    )
    assert not torch.allclose(direct.detach(), distorted.detach(), atol=1e-5)


def test_joint_zero_mask_only_excludes_aux_and_fail_closed_cases() -> None:
    clean, leo, labels = _geometry_inputs()
    clean = torch.cat([clean[:2], clean[:2], clean[2:4], clean[2:4], clean[4:6], clean[4:6], clean[6:], clean[6:]], dim=0)
    leo = torch.cat([leo[:2], leo[:2], leo[2:4], leo[2:4], leo[4:6], leo[4:6], leo[6:], leo[6:]], dim=0)
    labels = torch.arange(4).repeat_interleave(4)
    clean[0] = 0.0
    leo[4] = 0.0
    clean[8] = 0.0
    leo[8] = 0.0
    _, info = cagm_loss(clean.requires_grad_(True), leo.requires_grad_(True), labels)
    assert info["total_rows"] == 16
    assert info["valid_rows"] == 13
    assert info["clean_zero_rows"] == 2
    assert info["leo_zero_rows"] == 2
    assert info["union_zero_rows"] == 3
    assert info["both_zero_rows"] == 1
    assert info["per_tx_valid_rows"] == {"0": 3, "1": 3, "2": 3, "3": 4}

    nonfinite = clean.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(CAGMRuntimeError, match="non-finite"):
        cagm_loss(nonfinite.requires_grad_(True), leo.requires_grad_(True), labels)

    centroid_zero, centroid_zero_leo, centroid_zero_labels = _geometry_inputs()
    centroid_zero[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    centroid_zero[1] = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    with pytest.raises(CAGMRuntimeError, match="centroid norm is zero"):
        cagm_loss(
            centroid_zero.requires_grad_(True),
            centroid_zero_leo.requires_grad_(True),
            centroid_zero_labels,
        )

    missing = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3])
    with pytest.raises(CAGMRuntimeError, match="n_c>=2"):
        cagm_loss(clean[:8].requires_grad_(True), leo[:8].requires_grad_(True), missing)


def test_local_binding_config_and_control_identity_are_strict() -> None:
    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_cagm_local_head_class_binding(
        local_class_order=local_order,
        source_train_tx=local_order,
        checkpoint_train_tx=local_order,
        dataset_class_order=global_order,
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert binding["local_to_head_class_ids"] == list(FROZEN_CAGM_CLASS_IDS)
    assert torch.equal(
        remap_cagm_local_labels_to_head_rows(
            torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]
        ),
        torch.tensor([3, 0, 2]),
    )
    with pytest.raises(CAGMConfigurationError, match="class counts"):
        resolve_cagm_local_head_class_binding(
            local_class_order=local_order,
            source_train_tx=local_order,
            checkpoint_train_tx=local_order,
            dataset_class_order=global_order,
            local_data_class_count=4,
            checkpoint_head_class_count=6,
            live_head_class_count=6,
        )
    config = validate_cagm_args(_frozen_args())
    assert config == CAGMConfig(True, True, 0.02)
    assert cagm_config_receipt(config)["loss_rule"].startswith("DETACHED_CLEAN")
    control = validate_cagm_args(_frozen_args(enabled=False))
    base = torch.tensor(1.5, requires_grad=True)
    assert add_cagm_to_loss(base, None, control) is base
    for name, value in (
        ("lambda_cagm", 0.10),
        ("label_epochs", 39),
        ("sat_train_scenarios", "leo_clear_weak,leo_rain_weak"),
        ("use_unlabeled", True),
        ("phase1_icmt_enabled", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(CAGMConfigurationError):
            validate_cagm_args(bad)


def test_raw_encoder_vjp_is_nonzero_and_head_aux_gradient_is_none_or_zero() -> None:
    torch.manual_seed(41)
    model = _binding_model().train()
    clean, leo, labels = _geometry_inputs()
    out_clean = model.paired_output(clean)
    out_leo = model.paired_output(leo + 0.07)
    validate_cagm_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        expected_class_ids=FROZEN_CAGM_CLASS_IDS,
    )
    aux, _ = cagm_loss(out_clean["z_id"], out_leo["z_id"], labels)
    groups = cagm_shared_encoder_and_head_parameters(model)
    assert any(
        parameter is model.id_backbone.cls_head.joint_proj.weight
        for parameter in groups["shared_encoder"]
    )
    audit = cagm_aux_gradient_audit(aux, groups)
    assert audit["raw_unscaled"] is True and audit["diagnostic_only"] is True
    assert audit["shared_encoder"]["norm"] > 0.0
    assert audit["classifier_head"]["none_or_zero_expected"] is True
    assert audit["classifier_head"]["nonzero_parameters"] == 0.0
    sealed = update_cagm_gradient_audit_receipt(
        cagm_config_receipt(CAGMConfig(True, True, 0.02)), audit
    )
    assert sealed["cagm_gradient_audit_completed"] is True
    aux.backward()
    assert model.id_backbone.cls_head.head.weight.grad is None


def test_lite_d_no_query_smoke_uses_feat_joint_and_no_head_aux_gradient() -> None:
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(37)
    model = build_dual_model(
        num_classes=4, num_domains=1, dataset="wisig", input_len=128, model_variant="lite_d"
    ).train()
    x_clean = torch.randn(8, 2, 128)
    x_satellite = torch.randn(8, 2, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out_clean = model(x_clean, y_tx=labels, domain_labels=None, return_aux=True)
    out_sat = model(x_satellite, y_tx=labels, domain_labels=None, return_aux=True)
    assert out_clean["z_id_key"] == "feat_joint" and out_sat["z_id_key"] == "feat_joint"
    validate_cagm_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_sat,
        tx_labels=labels,
        expected_class_ids=FROZEN_CAGM_CLASS_IDS,
    )
    aux, _ = cagm_loss(out_clean["z_id"], out_sat["z_id"], labels)
    base = F.cross_entropy(out_clean["tx_logits"], labels) + 0.10 * F.kl_div(
        F.log_softmax(out_sat["tx_logits"], dim=1),
        out_clean["tx_logits"].detach().softmax(dim=1),
        reduction="batchmean",
    )
    audit = cagm_aux_gradient_audit(aux, cagm_shared_encoder_and_head_parameters(model))
    (base + FROZEN_CAGM_LAMBDA * aux).backward()
    assert audit["shared_encoder"]["norm"] > 0.0
    assert audit["classifier_head"]["nonzero_parameters"] == 0.0
    assert resolve_cagm_classifier_weight(model).grad is not None  # base only


def test_receipt_closes_scene_and_ten_term_coverage_and_control_stays_zero() -> None:
    clean, leo, labels = _geometry_inputs()
    receipt = _sealed_receipt(enabled=True)
    for scenario in FROZEN_CAGM_SCENARIOS:
        _, info = cagm_loss(clean.requires_grad_(True), leo.requires_grad_(True), labels)
        receipt = update_cagm_receipt(receipt, info, scenario=scenario)
    with pytest.raises(CAGMRuntimeError, match="VJP audit"):
        validate_cagm_terminal_receipt(receipt)
    receipt = update_cagm_gradient_audit_receipt(receipt, _audit())
    terminal = validate_cagm_terminal_receipt(receipt)
    assert terminal["cagm_terminal_contract_passed"] is True
    assert len(terminal["cagm_scenes"]) == 3
    assert len(terminal["cagm_radius_terms"]) == 4
    assert len(terminal["cagm_gram_terms"]) == 6
    assert terminal["cagm_total_rows"] == terminal["cagm_valid_rows"] == 24

    drifted = dict(terminal)
    drifted["cagm_scenes"] = {**terminal["cagm_scenes"]}
    drifted["cagm_scenes"]["leo_clear_weak"] = {
        **terminal["cagm_scenes"]["leo_clear_weak"],
        "valid_rows": 7,
    }
    with pytest.raises(CAGMRuntimeError, match="zero-mask closure"):
        validate_cagm_terminal_receipt(drifted)

    control = validate_cagm_terminal_receipt(_sealed_receipt(enabled=False))
    assert control["cagm_terminal_contract"] == "CONTROL_ARM_NOT_APPLICABLE_COMMON_SEQUENCE_BOUND"


def test_common_batch_sequence_new_adamw_warm_start_and_failure_receipt(tmp_path, monkeypatch, capsys) -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    metadata = {"base_index": torch.arange(10, 18), "rx_i": torch.arange(8), "day_i": torch.arange(8)}
    control = _sealed_receipt(enabled=False)
    guided = _sealed_receipt(enabled=True)
    control["common_batch_sequence_sha256"] = ""
    guided["common_batch_sequence_sha256"] = ""
    control = update_cagm_common_batch_sequence_receipt(
        control, epoch=1, batch_index=1, scenario="leo_clear_weak", source_tx_labels=labels, metadata=metadata
    )
    guided = update_cagm_common_batch_sequence_receipt(
        guided, epoch=1, batch_index=1, scenario="leo_clear_weak", source_tx_labels=labels, metadata=metadata
    )
    assert control["common_batch_sequence_sha256"] == guided["common_batch_sequence_sha256"]
    with pytest.raises(CAGMRuntimeError, match="scenario sequence drifted"):
        update_cagm_common_batch_sequence_receipt(
            guided, epoch=1, batch_index=2, scenario="leo_clear_weak", source_tx_labels=labels, metadata=metadata
        )
    optimizer = torch.optim.AdamW(_binding_model().parameters(), lr=2e-4, weight_decay=1e-4)
    initialized = bind_cagm_optimizer_initial_state(_sealed_receipt(enabled=True), optimizer)
    assert initialized["optimizer_initial_state_empty"] is True
    assert len(initialized["optimizer_initial_state_sha256"]) == 64
    bound = bind_cagm_source_data_order(
        _sealed_receipt(enabled=True), {"labeled_indices_sha256": "1" * 64, "split_manifest_sha256": "2" * 64}
    )
    assert bound["source_labeled_indices_sha256"] == "1" * 64

    source = _binding_model()
    target = _binding_model()
    warm = strict_cagm_warm_start(
        target, source.state_dict(), baseline_path="base.pth", baseline_sha256="a" * 64,
        checkpoint_epoch=40, checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False
    with pytest.raises(CAGMConfigurationError, match="training_final_only"):
        strict_cagm_warm_start(
            target, source.state_dict(), baseline_path="base.pth", baseline_sha256="a" * 64,
            checkpoint_epoch=40, checkpoint_role="source_validation_selected",
        )
    receipt = cagm_config_receipt(CAGMConfig(True, True, 0.02))
    written = write_cagm_failure_receipt(
        tmp_path, candidate_id="F1G", run_id="cagm12", receipt=receipt,
        error=CAGMRuntimeError("P1-CAGM classifier head must have no auxiliary gradient"), failure_stage="test",
    )
    assert json.loads(written.read_text(encoding="utf-8"))["error_fingerprint"] == "CAGM_AUX_GRADIENT_PATH_FAILURE"

    def writer_failure(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(train_ssdg, "write_cagm_failure_receipt", writer_failure)
    original = CAGMRuntimeError("P1-CAGM z contains non-finite values")
    with pytest.raises(CAGMRuntimeError) as caught:
        try:
            raise original
        except CAGMRuntimeError as error:
            train_ssdg._persist_cagm_failure_receipt(
                out_dir=tmp_path, args=SimpleNamespace(candidate_id="F1G", run_id="cagm12"),
                cagm_receipt=receipt, error=error, failure_stage="test",
            )
            raise
    assert caught.value is original
    assert "writer_exception_type=OSError" in capsys.readouterr().out


def test_train_integration_and_launcher_dry_run_have_only_the_frozen_cagm_route() -> None:
    source = inspect.getsource(train_ssdg.train)
    assert "cagm_loss(" in source and "validate_cagm_terminal_receipt" in source
    assert source.index("cagm_aux_gradient_audit(") < source.index("scaler.scale(loss).backward()")
    block_start = source.index('if bool(getattr(cagm_config, "enabled", False)):')
    block_end = source.index("if add_ccpc_to_loss is not None:", block_start)
    block = source[block_start:block_end]
    assert "data_ctx[\"val_loader\"]" not in block
    assert "proxy" not in block.lower() and "held" not in block.lower()
    assert "out_l[\"z_id\"]" in block and "out_sat[\"z_id\"]" in block
    scene_start = source.index('sat_train_scenarios = list(getattr(args, "sat_train_scenario_list"')
    scene_end = source.index("with torch.no_grad():", scene_start)
    scene_block = source[scene_start:scene_end]
    assert "cagm" not in scene_block.lower()
    assert "(int(epoch) + int(batch_idx) - 2) % max(1, len(sat_train_scenarios))" in scene_block

    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12 and {arm for _, arm, _ in calls} == {"C", "G"}
    assert "phase1_cagm12_20260810_v1" in launcher_text
    assert "--lambda_cagm 0.02" in launcher_text and "--lambda_cagm 0" in launcher_text
    assert "postfreeze" not in launcher_text.lower()
    completed = subprocess.run(
        ["bash", f"scripts/{LAUNCHER.name}", "--dry-run"],
        cwd=str(CODE_ROOT), check=True, text=True, capture_output=True,
    )
    assert completed.stdout.count("[DRY-RUN]") == 12
