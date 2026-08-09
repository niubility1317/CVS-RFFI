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
from cvsrffi.phase1_icmt import (  # noqa: E402
    FROZEN_ICMT_CLASS_IDS,
    FROZEN_ICMT_LAMBDA,
    FROZEN_ICMT_SCENARIOS,
    ICMTConfig,
    ICMTConfigurationError,
    ICMTRuntimeError,
    add_icmt_to_loss,
    bind_icmt_optimizer_initial_state,
    bind_icmt_source_data_order,
    icmt_config_receipt,
    icmt_loss,
    icmt_shared_encoder_and_head_parameters,
    icmt_shared_gradient_relation,
    remap_icmt_local_labels_to_head_rows,
    resolve_icmt_classifier_weight,
    resolve_icmt_local_head_class_binding,
    strict_icmt_warm_start,
    update_icmt_common_batch_sequence_receipt,
    update_icmt_gradient_relation_receipt,
    update_icmt_receipt,
    validate_icmt_args,
    validate_icmt_binding,
    validate_icmt_terminal_receipt,
    write_icmt_failure_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_icmt12_20260810.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        phase1_icmt_frozen_mode=True,
        phase1_icmt_enabled=enabled,
        lambda_icmt=FROZEN_ICMT_LAMBDA if enabled else 0.0,
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
        sat_train_scenarios=",".join(FROZEN_ICMT_SCENARIOS),
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
            self.id_backbone.encoder = torch.nn.Linear(3, 3, bias=False)
            self.id_backbone.cls_head = torch.nn.Module()
            self.id_backbone.cls_head.head = torch.nn.Linear(3, 4, bias=False)
            self.id_backbone.cls_head.imp_merge = torch.nn.Linear(3, 3)
            self.id_backbone.cls_head.dac_head = torch.nn.Linear(3, 1)
            self.id_backbone.cls_head.pa_head = torch.nn.Linear(3, 1)

        def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
            z_id = self.id_backbone.encoder(x)
            return {
                "z_id": z_id,
                "z_id_key": "feat_joint",
                "tx_logits": self.id_backbone.cls_head.head(z_id),
            }

    return _BindingModel()


def _features_and_labels() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [1.2, 0.3, 0.1], [1.0, 0.1, 0.2],
            [0.2, 1.1, 0.3], [0.1, 1.2, 0.2],
            [0.2, 0.2, 1.1], [0.3, 0.1, 1.2],
            [1.0, 0.8, 0.2], [0.8, 1.0, 0.2],
        ],
        dtype=torch.float32,
    )
    return features, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])


def _logits_from_known_margins(margins_by_class: list[tuple[float, float]]) -> torch.Tensor:
    """Make true-vs-logsumexp(other) margins equal to the listed values."""

    log_three = float(torch.log(torch.tensor(3.0)).item())
    rows: list[torch.Tensor] = []
    for class_id, margins in enumerate(margins_by_class):
        for margin in margins:
            row = torch.zeros(4, dtype=torch.float32)
            row[class_id] = float(margin) + log_three
            rows.append(row)
    return torch.stack(rows).requires_grad_(True)


def _relation() -> dict[str, object]:
    return {
        "shared_encoder": {
            "parameter_count": 1.0,
            "base_norm": 1.0,
            "icmt_norm": 0.2,
            "cosine": -0.1,
            "norm_ratio": 0.2,
        },
        "classifier_head": {
            "parameter_count": 1.0,
            "base_norm": 1.0,
            "icmt_norm": 0.3,
            "cosine": 0.1,
            "norm_ratio": 0.3,
        },
        "raw_unscaled": True,
        "diagnostic_only": True,
    }


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = icmt_config_receipt(ICMTConfig(True, enabled, 0.05 if enabled else 0.0))
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
            "common_scenario_batches": {scenario: 1 for scenario in FROZEN_ICMT_SCENARIOS},
        }
    )
    return receipt


def test_formula_is_manual_fixed_one_eighth_all_nc_and_strict_tie_zero() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    clean = _logits_from_known_margins([(0.0, 2.0), (1.0, 5.0), (-1.0, 0.0), (2.0, 3.0)])
    leo = _logits_from_known_margins([(0.0, 4.0), (1.0, 2.0), (-3.0, 1.0), (3.0, 5.0)])
    loss, info = icmt_loss(clean, leo, labels)
    # Per-class means are [1,3,-.5,2.5] and [2,1.5,-1,4].  Each lower
    # row contributes its squared distance divided by n_c=2, hence the
    # clean sum is .5+2+.125+.125=2.75 and the LEO sum is 2+.125+2+.5=4.625.
    assert float(loss.detach()) == pytest.approx((2.75 + 4.625) / 8.0, abs=1e-6)
    assert info["loss_divisor"] == 8
    assert info["all_n_c_mean_denominator"] is True
    assert info["views"]["clean"]["per_tx_active_rows"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert info["views"]["leo"]["per_tx_active_rows"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    loss.backward()
    assert clean.grad is not None and leo.grad is not None

    ties = torch.zeros((8, 4), requires_grad=True)
    tie_loss, tie_info = icmt_loss(ties, ties, labels)
    assert float(tie_loss.detach()) == 0.0
    assert all(
        value == 0
        for view in tie_info["views"].values()
        for value in view["per_tx_active_rows"].values()
    )
    source = inspect.getsource(icmt_loss) + inspect.getsource(train_ssdg.icmt_loss)
    assert "softmax" not in inspect.getsource(icmt_loss).lower()
    assert "temperature" not in source.lower()


def test_loss_is_label_permutation_equivariant_and_fails_closed_for_missing_or_nonfinite_rows() -> None:
    _, labels = _features_and_labels()
    clean = _logits_from_known_margins([(0.0, 2.0), (1.0, 5.0), (-1.0, 0.0), (2.0, 3.0)])
    leo = _logits_from_known_margins([(0.0, 4.0), (1.0, 2.0), (-3.0, 1.0), (3.0, 5.0)])
    direct, _ = icmt_loss(clean, leo, labels)
    perm = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(perm)
    permuted, _ = icmt_loss(clean.detach()[:, perm].clone().requires_grad_(True), leo.detach()[:, perm].clone().requires_grad_(True), inverse[labels])
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-6)

    missing = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3])
    with pytest.raises(ICMTRuntimeError, match="n_c>=2"):
        icmt_loss(clean, leo, missing)
    nonfinite = clean.detach().clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ICMTRuntimeError, match="non-finite"):
        icmt_loss(nonfinite.requires_grad_(True), leo, labels)


def test_local_binding_config_and_control_identity_are_strict() -> None:
    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_icmt_local_head_class_binding(
        local_class_order=local_order,
        source_train_tx=local_order,
        checkpoint_train_tx=local_order,
        dataset_class_order=global_order,
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert binding["local_to_head_class_ids"] == list(FROZEN_ICMT_CLASS_IDS)
    assert torch.equal(
        remap_icmt_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0, 2]),
    )
    with pytest.raises(ICMTConfigurationError, match="class counts"):
        resolve_icmt_local_head_class_binding(
            local_class_order=local_order,
            source_train_tx=local_order,
            checkpoint_train_tx=local_order,
            dataset_class_order=global_order,
            local_data_class_count=4,
            checkpoint_head_class_count=6,
            live_head_class_count=6,
        )
    config = validate_icmt_args(_frozen_args())
    assert config == ICMTConfig(True, True, 0.05)
    assert icmt_config_receipt(config)["loss_rule"].startswith("RAW_PRE_SOFTMAX")
    control = validate_icmt_args(_frozen_args(enabled=False))
    base = torch.tensor(1.5, requires_grad=True)
    assert add_icmt_to_loss(base, None, control) is base
    for name, value in (
        ("lambda_icmt", 0.10),
        ("label_epochs", 39),
        ("sat_train_scenarios", "leo_clear_weak,leo_rain_weak"),
        ("use_unlabeled", True),
        ("phase1_pamr_audit_only", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(ICMTConfigurationError):
            validate_icmt_args(bad)


def test_lite_d_no_query_smoke_raw_logits_zid_and_vjp_audit() -> None:
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
    assert tuple(out_clean["tx_logits"].shape) == (8, 4)
    assert resolve_icmt_classifier_weight(model) is model.id_backbone.cls_head.head.weight
    validate_icmt_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_sat,
        tx_labels=labels,
        expected_class_ids=[0, 1, 2, 3],
    )
    aux, _ = icmt_loss(out_clean["tx_logits"], out_sat["tx_logits"], labels)
    base = F.cross_entropy(out_clean["tx_logits"], labels) + 0.10 * F.kl_div(
        F.log_softmax(out_sat["tx_logits"], dim=1),
        out_clean["tx_logits"].detach().softmax(dim=1),
        reduction="batchmean",
    )
    relation = icmt_shared_gradient_relation(
        base,
        aux,
        icmt_shared_encoder_and_head_parameters(model),
        loss_weight=0.05,
    )
    (base + 0.05 * aux).backward()
    assert relation["raw_unscaled"] is True and relation["diagnostic_only"] is True
    assert relation["shared_encoder"]["base_norm"] > 0.0
    assert relation["shared_encoder"]["icmt_norm"] > 0.0
    assert relation["classifier_head"]["base_norm"] > 0.0
    assert relation["classifier_head"]["icmt_norm"] > 0.0
    assert resolve_icmt_classifier_weight(model).grad is not None


def test_vjp_rejects_head_only_or_detached_encoder_path() -> None:
    torch.manual_seed(41)
    model = _binding_model()
    features, labels = _features_and_labels()
    clean = model.paired_output(features)
    sat = model.paired_output(features + 0.05)
    aux, _ = icmt_loss(clean["tx_logits"], sat["tx_logits"], labels)
    base = F.cross_entropy(clean["tx_logits"], labels)
    relation = icmt_shared_gradient_relation(
        base,
        aux,
        icmt_shared_encoder_and_head_parameters(model),
        loss_weight=0.05,
    )
    receipt = update_icmt_gradient_relation_receipt(
        icmt_config_receipt(ICMTConfig(True, True, 0.05)), relation
    )
    assert receipt["icmt_gradient_relation_completed"] is True

    detached = model.id_backbone.encoder(features).detach().requires_grad_(True)
    detached_logits = model.id_backbone.cls_head.head(detached)
    detached_aux, _ = icmt_loss(detached_logits, detached_logits, labels)
    with pytest.raises(ICMTRuntimeError, match="shared_encoder VJP is None or detached"):
        icmt_shared_gradient_relation(
            base,
            detached_aux,
            icmt_shared_encoder_and_head_parameters(model),
            loss_weight=0.05,
        )
    invalid = _relation()
    invalid["shared_encoder"] = {**invalid["shared_encoder"], "icmt_norm": 0.0}
    with pytest.raises(ICMTRuntimeError, match="head-only, detached, or zero"):
        update_icmt_gradient_relation_receipt(
            icmt_config_receipt(ICMTConfig(True, True, 0.05)), invalid
        )


def test_receipt_closes_all_sixteen_cells_and_control_remains_zero() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    receipt = _sealed_receipt(enabled=True)
    for scenario in FROZEN_ICMT_SCENARIOS:
        clean = _logits_from_known_margins([(0.0, 2.0), (1.0, 5.0), (-1.0, 0.0), (2.0, 3.0)])
        leo = _logits_from_known_margins([(0.0, 4.0), (1.0, 2.0), (-3.0, 1.0), (3.0, 5.0)])
        _, info = icmt_loss(clean, leo, labels)
        receipt = update_icmt_receipt(receipt, info, scenario=scenario)
    with pytest.raises(ICMTRuntimeError, match="VJP receipt"):
        validate_icmt_terminal_receipt(receipt)
    receipt = update_icmt_gradient_relation_receipt(receipt, _relation())
    terminal = validate_icmt_terminal_receipt(receipt)
    assert terminal["icmt_terminal_contract_passed"] is True
    assert len(terminal["icmt_clean_cells"]) == 4
    assert len(terminal["icmt_leo_cells"]) == 12
    assert terminal["icmt_clean_rows"] == terminal["icmt_leo_rows"] == 24
    assert terminal["icmt_clean_cells"]["tx0"]["rows"] == 6
    for scenario in FROZEN_ICMT_SCENARIOS:
        assert terminal["icmt_leo_cells"][f"tx0|{scenario}"]["rows"] == 2

    drifted = dict(terminal)
    drifted["icmt_clean_cells"] = {**terminal["icmt_clean_cells"]}
    drifted["icmt_clean_cells"]["tx0"] = {
        **terminal["icmt_clean_cells"]["tx0"],
        "rows": 7,
        "finite_rows": 7,
    }
    with pytest.raises(ICMTRuntimeError, match="ROWS_NEQ_SUM_THREE_LEO"):
        validate_icmt_terminal_receipt(drifted)

    control = validate_icmt_terminal_receipt(_sealed_receipt(enabled=False))
    assert control["icmt_terminal_contract"] == "CONTROL_ARM_NOT_APPLICABLE_COMMON_SEQUENCE_BOUND"


def test_common_batch_sequence_and_new_adamw_are_identical_for_c_and_g() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    metadata = {"base_index": torch.arange(10, 18), "rx_i": torch.arange(8), "day_i": torch.arange(8)}
    control = _sealed_receipt(enabled=False)
    guided = _sealed_receipt(enabled=True)
    control["common_batch_sequence_sha256"] = ""
    guided["common_batch_sequence_sha256"] = ""
    control = update_icmt_common_batch_sequence_receipt(
        control,
        epoch=1,
        batch_index=1,
        scenario="leo_clear_weak",
        source_tx_labels=labels,
        metadata=metadata,
    )
    guided = update_icmt_common_batch_sequence_receipt(
        guided,
        epoch=1,
        batch_index=1,
        scenario="leo_clear_weak",
        source_tx_labels=labels,
        metadata=metadata,
    )
    assert control["common_batch_sequence_sha256"] == guided["common_batch_sequence_sha256"]
    assert control["common_batch_sequence_rows"] == guided["common_batch_sequence_rows"]
    with pytest.raises(ICMTRuntimeError, match="scenario sequence drifted"):
        update_icmt_common_batch_sequence_receipt(
            guided,
            epoch=1,
            batch_index=2,
            scenario="leo_clear_weak",
            source_tx_labels=labels,
            metadata=metadata,
        )

    optimizer = torch.optim.AdamW(_binding_model().parameters(), lr=2e-4, weight_decay=1e-4)
    initialized = bind_icmt_optimizer_initial_state(_sealed_receipt(enabled=True), optimizer)
    assert initialized["optimizer_initial_state_empty"] is True
    assert len(initialized["optimizer_initial_state_sha256"]) == 64

    bound = bind_icmt_source_data_order(
        _sealed_receipt(enabled=True),
        {"labeled_indices_sha256": "1" * 64, "split_manifest_sha256": "2" * 64},
    )
    assert bound["source_labeled_indices_sha256"] == "1" * 64


def test_warm_start_final_only_failure_receipt_and_train_integration_are_fail_closed(tmp_path, monkeypatch, capsys) -> None:
    source = _binding_model()
    target = _binding_model()
    warm = strict_icmt_warm_start(
        target,
        source.state_dict(),
        baseline_path="base.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False
    with pytest.raises(ICMTConfigurationError, match="training_final_only"):
        strict_icmt_warm_start(
            target,
            source.state_dict(),
            baseline_path="base.pth",
            baseline_sha256="a" * 64,
            checkpoint_epoch=40,
            checkpoint_role="source_validation_selected",
        )
    receipt = icmt_config_receipt(ICMTConfig(True, True, 0.05))
    written = write_icmt_failure_receipt(
        tmp_path,
        candidate_id="F1G",
        run_id="icmt12",
        receipt=receipt,
        error=ICMTRuntimeError("P1-ICMT shared_encoder VJP is None or detached"),
        failure_stage="test",
    )
    assert json.loads(written.read_text(encoding="utf-8"))["error_fingerprint"] == "ICMT_VJP_PATH_FAILURE"

    def writer_failure(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(train_ssdg, "write_icmt_failure_receipt", writer_failure)
    original = ICMTRuntimeError("P1-ICMT raw logits non-finite")
    with pytest.raises(ICMTRuntimeError) as caught:
        try:
            raise original
        except ICMTRuntimeError as error:
            train_ssdg._persist_icmt_failure_receipt(
                out_dir=tmp_path,
                args=SimpleNamespace(candidate_id="F1G", run_id="icmt12"),
                icmt_receipt=receipt,
                error=error,
                failure_stage="test",
            )
            raise
    assert caught.value is original
    assert "writer_exception_type=OSError" in capsys.readouterr().out

    source_text = inspect.getsource(train_ssdg.train)
    assert "icmt_loss(" in source_text
    assert "validate_icmt_terminal_receipt" in source_text
    assert source_text.index("icmt_shared_gradient_relation(") < source_text.index("scaler.scale(loss).backward()")
    block_start = source_text.index('if bool(getattr(icmt_config, "enabled", False)):')
    block_end = source_text.index("if add_ccpc_to_loss is not None:", block_start)
    block = source_text[block_start:block_end]
    assert "data_ctx[\"val_loader\"]" not in block
    assert "proxy" not in block.lower() and "held" not in block.lower()
    assert "out_l[\"tx_logits\"]" in block and "out_sat[\"tx_logits\"]" in block


def test_c_and_g_share_the_global_scene_formula_and_launcher_dry_run_has_twelve_arms() -> None:
    source = inspect.getsource(train_ssdg.train)
    scene_start = source.index('sat_train_scenarios = list(getattr(args, "sat_train_scenario_list"')
    scene_end = source.index("with torch.no_grad():", scene_start)
    scene_block = source[scene_start:scene_end]
    assert "icmt" not in scene_block.lower()
    assert "(int(epoch) + int(batch_idx) - 2) % max(1, len(sat_train_scenarios))" in scene_block
    scenes = tuple(FROZEN_ICMT_SCENARIOS)
    c_sequence = [scenes[(epoch + batch_index - 2) % len(scenes)] for epoch in range(1, 5) for batch_index in range(1, 31)]
    g_sequence = [scenes[(epoch + batch_index - 2) % len(scenes)] for epoch in range(1, 5) for batch_index in range(1, 31)]
    assert c_sequence == g_sequence

    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert calls == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    assert "phase1_icmt12_20260810_v1" in launcher_text
    assert "--lambda_icmt 0.05" in launcher_text and "--lambda_icmt 0" in launcher_text
    assert "postfreeze" not in launcher_text.lower()
    completed = subprocess.run(
        ["bash", f"scripts/{LAUNCHER.name}", "--dry-run"],
        cwd=str(CODE_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.count("[DRY-RUN]") == 12
