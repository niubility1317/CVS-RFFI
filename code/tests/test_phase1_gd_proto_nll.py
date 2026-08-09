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
from cvsrffi.phase1_gd_proto_nll import (  # noqa: E402
    FROZEN_GD_PROTO_NLL_BETA,
    FROZEN_GD_PROTO_NLL_LAMBDA,
    FROZEN_GD_PROTO_NLL_SCENARIOS,
    GDProtoNLLConfig,
    GDProtoNLLConfigurationError,
    GDProtoNLLRuntimeError,
    add_gd_proto_nll_to_loss,
    advance_gd_proto_nll_state,
    fit_gd_proto_nll_geometry,
    gd_proto_nll_config_receipt,
    gd_proto_nll_loss,
    gd_proto_nll_score,
    gd_proto_nll_shared_encoder_and_head_parameters,
    gd_proto_nll_shared_gradient_relation,
    make_gd_proto_nll_state,
    remap_gd_proto_nll_local_labels_to_head_rows,
    resolve_gd_proto_nll_classifier_weight,
    resolve_gd_proto_nll_local_head_class_binding,
    strict_gd_proto_nll_warm_start,
    update_gd_proto_nll_gradient_relation_receipt,
    update_gd_proto_nll_receipt,
    update_gd_proto_nll_state_receipt,
    validate_gd_proto_nll_args,
    validate_gd_proto_nll_feature_binding,
    validate_gd_proto_nll_terminal_receipt,
    write_gd_proto_nll_failure_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_gd_proto_nll12_20260809.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        phase1_gd_proto_nll_frozen_mode=True,
        phase1_gd_proto_nll_enabled=enabled,
        lambda_gd_proto_nll=FROZEN_GD_PROTO_NLL_LAMBDA if enabled else 0.0,
        gd_proto_nll_gamma=1.0,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        id_feature_key="feat_joint",
        epochs=epochs,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        sat_cons_start_epoch=1,
        sat_view_prob=1.0,
        sat_train_scenarios=",".join(FROZEN_GD_PROTO_NLL_SCENARIOS),
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
        requires_grad=True,
    )
    return features, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])


def _nonzero_class_weight() -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
    )


def test_lagged_dro_uses_old_q_fixed_scene_scale_and_detached_ema_update() -> None:
    features, labels = _features_and_labels()
    weight = _nonzero_class_weight()
    state = make_gd_proto_nll_state("cpu")
    state["q"] = torch.tensor([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.05, 0.06, 0.07, 0.14])
    state["q"] = state["q"] / state["q"].sum()
    loss, info = gd_proto_nll_loss(
        features, weight, labels, scenario="leo_low_elev_weak", state=state
    )
    assert info["uses_old_q"] is True
    assert info["fixed_scene_scale"] == 3.0
    assert info["per_tx_rows"] == {"0": 2, "1": 2, "2": 2, "3": 2}
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0.0
    assert weight.grad is not None and float(weight.grad.abs().sum()) > 0.0
    next_state = advance_gd_proto_nll_state(state, info)
    assert torch.isclose(next_state["q"].sum(), torch.tensor(1.0), atol=1e-6)
    assert float(next_state["barl"][1]) == pytest.approx(
        FROZEN_GD_PROTO_NLL_BETA * info["per_tx_loss"]["0"], rel=1e-6
    )
    assert float(next_state["barl"][0]) == 0.0


def test_loss_requires_every_local_four_and_is_label_permutation_equivariant() -> None:
    features, labels = _features_and_labels()
    weight = _nonzero_class_weight()
    state = make_gd_proto_nll_state("cpu")
    direct, _ = gd_proto_nll_loss(features, weight, labels, scenario="leo_clear_weak", state=state)
    perm = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(perm)
    permuted_features = features.detach().clone().requires_grad_(True)
    permuted_weight = torch.nn.Parameter(weight.detach()[perm].clone())
    permuted_state = {
        "q": state["q"].reshape(4, 3)[perm].reshape(-1),
        "barl": state["barl"].reshape(4, 3)[perm].reshape(-1),
    }
    permuted, _ = gd_proto_nll_loss(
        permuted_features,
        permuted_weight,
        inverse[labels],
        scenario="leo_clear_weak",
        state=permuted_state,
    )
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-6)
    with pytest.raises(GDProtoNLLRuntimeError, match="all local4"):
        gd_proto_nll_loss(
            features,
            weight,
            torch.tensor([0, 0, 1, 1, 2, 2, 2, 2]),
            scenario="leo_clear_weak",
            state=state,
        )
    with pytest.raises(GDProtoNLLRuntimeError, match="zero L2 norm"):
        gd_proto_nll_loss(
            torch.zeros_like(features, requires_grad=True),
            weight,
            labels,
            scenario="leo_clear_weak",
            state=state,
        )
    with pytest.raises(GDProtoNLLRuntimeError, match="zero L2 norm"):
        gd_proto_nll_loss(
            features,
            torch.nn.Parameter(torch.zeros_like(weight)),
            labels,
            scenario="leo_clear_weak",
            state=state,
        )
    overflow_feature = torch.full(
        features.shape,
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
        requires_grad=True,
    )
    with pytest.raises(GDProtoNLLRuntimeError, match="non-finite or zero L2 norm"):
        gd_proto_nll_loss(
            overflow_feature,
            weight,
            labels,
            scenario="leo_clear_weak",
            state=state,
        )


def test_local_binding_and_config_are_strict_and_control_is_identity() -> None:
    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_gd_proto_nll_local_head_class_binding(
        local_class_order=local_order,
        source_train_tx=local_order,
        checkpoint_train_tx=local_order,
        dataset_class_order=global_order,
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert torch.equal(
        remap_gd_proto_nll_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0, 2]),
    )
    with pytest.raises(GDProtoNLLConfigurationError, match="class counts"):
        resolve_gd_proto_nll_local_head_class_binding(
            local_class_order=local_order, source_train_tx=local_order,
            checkpoint_train_tx=local_order, dataset_class_order=global_order,
            local_data_class_count=4, checkpoint_head_class_count=6, live_head_class_count=6,
        )
    config = validate_gd_proto_nll_args(_frozen_args())
    assert config == GDProtoNLLConfig(True, True, 0.10, 1.0)
    assert gd_proto_nll_config_receipt(config)["satellite_schedule"] == "BASELINE_EPOCH_BATCH_MODULO_CLEAR_LOW_RAIN"
    control = validate_gd_proto_nll_args(_frozen_args(enabled=False))
    base = torch.tensor(1.5, requires_grad=True)
    assert add_gd_proto_nll_to_loss(base, None, control) is base
    control_receipt = gd_proto_nll_config_receipt(control)
    control_receipt["checkpoint_role"] = "training_final_only"
    assert validate_gd_proto_nll_terminal_receipt(control_receipt)["gd_proto_nll_terminal_contract"] == "CONTROL_ARM_NOT_APPLICABLE"
    role_drift = gd_proto_nll_config_receipt(control)
    role_drift["checkpoint_role"] = "UNSPECIFIED"
    with pytest.raises(GDProtoNLLRuntimeError, match="training_final_only"):
        validate_gd_proto_nll_terminal_receipt(role_drift)
    for name, value in (("lambda_gd_proto_nll", 0.05), ("gd_proto_nll_gamma", 2.0), ("sat_train_scenarios", "leo_clear_weak,leo_rain_weak"), ("use_unlabeled", True), ("phase1_pamr_audit_only", True)):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(GDProtoNLLConfigurationError):
            validate_gd_proto_nll_args(bad)


def test_lite_d_no_query_forward_backward_smoke_and_raw_gradient_audit() -> None:
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(31)
    model = build_dual_model(
        num_classes=4, num_domains=1, dataset="wisig", input_len=128, model_variant="lite_d"
    ).train()
    x_clean = torch.randn(8, 2, 128)
    x_satellite = torch.randn(8, 2, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out_clean = model(x_clean, y_tx=labels, domain_labels=None, return_aux=True)
    out_sat = model(x_satellite, y_tx=labels, domain_labels=None, return_aux=True)
    weight = validate_gd_proto_nll_feature_binding(
        model=model,
        satellite_feature=out_sat["z_id"],
        tx_labels=labels,
        expected_class_ids=[0, 1, 2, 3],
        z_id_key=out_sat["z_id_key"],
    )
    aux, _ = gd_proto_nll_loss(out_sat["z_id"], weight, labels, scenario="leo_clear_weak", state=make_gd_proto_nll_state("cpu"))
    base = F.cross_entropy(out_clean["tx_logits"], labels) + 0.10 * F.kl_div(
        F.log_softmax(out_sat["tx_logits"], dim=1), out_clean["tx_logits"].detach().softmax(dim=1), reduction="batchmean"
    )
    relation = gd_proto_nll_shared_gradient_relation(
        base, aux, gd_proto_nll_shared_encoder_and_head_parameters(model), loss_weight=0.10
    )
    (base + 0.10 * aux).backward()
    assert relation["raw_unscaled"] is True
    assert relation["diagnostic_only"] is True
    assert relation["shared_encoder"]["base_norm"] > 0.0
    assert relation["classifier_head"]["gd_proto_nll_norm"] > 0.0
    assert resolve_gd_proto_nll_classifier_weight(model).grad is not None


def test_terminal_receipt_requires_all_twelve_cells_state_and_first_relation() -> None:
    receipt = gd_proto_nll_config_receipt(GDProtoNLLConfig(True, True, 0.10, 1.0))
    receipt["expected_tx_class_ids"] = [0, 1, 2, 3]
    receipt["checkpoint_role"] = "training_final_only"
    state = make_gd_proto_nll_state("cpu")
    for scenario in FROZEN_GD_PROTO_NLL_SCENARIOS:
        features, labels = _features_and_labels()
        weight = _nonzero_class_weight()
        _, info = gd_proto_nll_loss(features, weight, labels, scenario=scenario, state=state)
        receipt = update_gd_proto_nll_receipt(receipt, info, scenario=scenario)
        state = advance_gd_proto_nll_state(state, info)
        receipt = update_gd_proto_nll_state_receipt(receipt, state)
    with pytest.raises(GDProtoNLLRuntimeError, match="first-batch"):
        validate_gd_proto_nll_terminal_receipt(receipt)
    relation = {
        "shared_encoder": {"parameter_count": 1.0, "base_norm": 1.0, "gd_proto_nll_norm": 0.2, "cosine": -0.1, "norm_ratio": 0.2},
        "classifier_head": {"parameter_count": 1.0, "base_norm": 1.0, "gd_proto_nll_norm": 0.3, "cosine": 0.1, "norm_ratio": 0.3},
        "raw_unscaled": True,
        "diagnostic_only": True,
    }
    receipt = update_gd_proto_nll_gradient_relation_receipt(receipt, relation)
    terminal = validate_gd_proto_nll_terminal_receipt(receipt)
    assert terminal["gd_proto_nll_terminal_contract_passed"] is True
    assert len(terminal["gd_proto_nll_cells"]) == 12
    bad = {**relation, "shared_encoder": {**relation["shared_encoder"], "base_norm": 0.0}}
    with pytest.raises(GDProtoNLLRuntimeError, match="zero norm"):
        update_gd_proto_nll_gradient_relation_receipt(gd_proto_nll_config_receipt(GDProtoNLLConfig(True, True, 0.10, 1.0)), bad)


def test_float64_l_only_gaussian_geometry_uses_ddof_pooling_and_no_threshold() -> None:
    features, labels = _features_and_labels()
    geometry = fit_gd_proto_nll_geometry(features.detach(), labels)
    normalized = F.normalize(features.detach().double(), dim=1)
    expected_raw = torch.stack([
        ((normalized[labels.eq(class_id)] - normalized[labels.eq(class_id)].mean(0)).square()).sum(0) / 1.0
        for class_id in range(4)
    ])
    assert geometry["feature_normalization"] == "float64_l2"
    assert geometry["ddof"] == 1 and geometry["n_by_class"] == [2, 2, 2, 2]
    assert torch.equal(geometry["raw_variances"], expected_raw)
    score = gd_proto_nll_score(features.detach(), geometry)
    assert score.dtype == torch.float64 and tuple(score.shape) == (8,)
    assert torch.isfinite(score).all()
    assert "threshold" not in inspect.signature(gd_proto_nll_score).parameters
    with pytest.raises(GDProtoNLLRuntimeError, match="n_c"):
        fit_gd_proto_nll_geometry(features.detach()[:-1], labels[:-1])


def test_warm_start_failure_receipt_is_best_effort_and_does_not_mask(tmp_path, monkeypatch, capsys) -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    warm = strict_gd_proto_nll_warm_start(
        target, source.state_dict(), baseline_path="base.pth", baseline_sha256="a" * 64,
        checkpoint_epoch=40, checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False
    for role in ("UNSPECIFIED", "source_validation_selected", ""):
        with pytest.raises(GDProtoNLLConfigurationError, match="training_final_only"):
            strict_gd_proto_nll_warm_start(
                target, source.state_dict(), baseline_path="base.pth", baseline_sha256="a" * 64,
                checkpoint_epoch=40, checkpoint_role=role,
            )
    with pytest.raises(GDProtoNLLConfigurationError, match="strict baseline"):
        strict_gd_proto_nll_warm_start(
            target, {"weight": source.weight.detach().clone()}, baseline_path="base.pth",
            baseline_sha256="a" * 64, checkpoint_epoch=40, checkpoint_role="training_final_only",
        )
    receipt = gd_proto_nll_config_receipt(GDProtoNLLConfig(True, True, 0.10, 1.0))
    original = GDProtoNLLRuntimeError("P1-GD-ProtoNLL shared_encoder gradient is non-finite")
    written = write_gd_proto_nll_failure_receipt(
        tmp_path, candidate_id="F1G", run_id="gd12", receipt=receipt, error=original, failure_stage="test"
    )
    assert json.loads(written.read_text(encoding="utf-8"))["error_fingerprint"] == "GD_PROTO_NLL_NONFINITE"

    def writer_failure(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(train_ssdg, "write_gd_proto_nll_failure_receipt", writer_failure)
    with pytest.raises(GDProtoNLLRuntimeError) as caught:
        try:
            raise original
        except GDProtoNLLRuntimeError as error:
            train_ssdg._persist_gd_proto_nll_failure_receipt(
                out_dir=tmp_path, args=SimpleNamespace(candidate_id="F1G", run_id="gd12"),
                gd_proto_nll_receipt=receipt, error=error, failure_stage="test",
            )
            raise
    assert caught.value is original
    assert "writer_exception_type=OSError" in capsys.readouterr().out


def test_train_integration_is_l_only_and_updates_after_scaled_backward() -> None:
    source = inspect.getsource(train_ssdg.train)
    assert "gd_proto_nll_satellite_step" not in source
    assert "gd_proto_nll_loss(" in source
    assert "advance_gd_proto_nll_state(" in source
    assert source.index("scaler.scale(loss).backward()") < source.index("advance_gd_proto_nll_state(")
    block_start = source.index('if bool(getattr(gd_proto_nll_config, "enabled", False)):')
    block_end = source.index("gd_proto_nll_base_loss_l = loss_closed_l", block_start)
    block = source[block_start:block_end]
    assert "data_ctx[\"val_loader\"]" not in block
    assert "proxy" not in block.lower()
    assert "update_gd_proto_nll_receipt" in source
    assert "validate_gd_proto_nll_terminal_receipt" in source


def test_c_and_g_reuse_the_exact_baseline_epoch_batch_scene_sequence() -> None:
    source = inspect.getsource(train_ssdg.train)
    assert "gd_proto_nll_satellite_step" not in source
    scene_start = source.index('sat_train_scenarios = list(getattr(args, "sat_train_scenario_list"')
    scene_end = source.index("with torch.no_grad():", scene_start)
    scene_block = source[scene_start:scene_end]
    assert "gd_proto_nll" not in scene_block
    assert "(int(epoch) + int(batch_idx) - 2) % max(1, len(sat_train_scenarios))" in scene_block
    scenes = tuple(FROZEN_GD_PROTO_NLL_SCENARIOS)
    c_sequence = [scenes[(epoch + batch_idx - 2) % len(scenes)] for epoch in range(1, 5) for batch_idx in range(30)]
    g_sequence = [scenes[(epoch + batch_idx - 2) % len(scenes)] for epoch in range(1, 5) for batch_idx in range(30)]
    assert c_sequence == g_sequence


def test_launcher_has_frozen_twelve_arm_matrix_and_dry_run() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", text, flags=re.MULTILINE)
    assert calls == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    assert "phase1_gd_proto_nll12_20260809_v1" in text
    assert "--lambda_gd_proto_nll 0.10" in text and "--lambda_gd_proto_nll 0" in text
    assert "--gd_proto_nll_gamma 1" in text
    assert "postfreeze" not in text.lower()
    completed = subprocess.run(
        ["bash", f"scripts/{LAUNCHER.name}", "--dry-run"], cwd=str(CODE_ROOT),
        check=True, text=True, capture_output=True,
    )
    assert completed.stdout.count("[DRY-RUN]") == 12
