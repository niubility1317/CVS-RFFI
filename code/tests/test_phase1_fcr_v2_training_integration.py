from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import train  # noqa: E402
from cvsrffi import checkpoint as checkpoint_module  # noqa: E402
from cvsrffi.phase1_fcr_types import FCRV2CapabilityState  # noqa: E402
from cvsrffi.phase1_fcr_v2_schedule import FCRV2Schedule  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


V2_ROWS = ("C1", "C2", "C3", "S0", "S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4", "M5", "M6")


def _args(*, row: str, version: str = "v2") -> argparse.Namespace:
    return argparse.Namespace(
        phase1_method="adv3b02_fcr",
        use_fcr=True,
        fcr_version=version,
        fcr_matrix_row=row,
        fcr_ablation_row="R8",
        epochs=200,
        train_mode="centralized",
        use_concat_sat_channel_aug=False,
        use_meta_ssl_cvs=True,
        ssl_labeled_ratio=0.07,
        ssl_unlabeled_ratio=0.63,
        ssl_val_ratio=0.30,
        lambda_fcr_self=1.0,
        lambda_fcr_swap=1.0,
        lambda_fcr_shared=1.0,
        lambda_fcr_cycle=1.0,
        lambda_fcr_eta=1.0,
        lambda_fcr_factor=1.0,
        lambda_fcr_need=1.0,
        lambda_fcr_phys=1.0,
    )


def _small_model(*, use_fcr: bool = True, fcr_version: str = "v2"):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=use_fcr,
        fcr_version=fcr_version,
    )


@pytest.mark.parametrize("row", V2_ROWS)
def test_v2_all_complete_matrix_rows_resolve(row: str) -> None:
    args = train.resolve_fcr_training_options(_args(row=row))

    assert args.fcr_version == "v2"
    assert args.fcr_matrix_row == row
    assert args.fcr_ablation_row == row
    assert args.fcr_execution_signature.startswith(f"v2:{row}|")
    assert args.use_fcr is (row != "C1")


def test_task7_legacy_row_flag_selects_v2_without_changing_launcher() -> None:
    args = _args(row="")
    args.fcr_version = "v1"
    args.fcr_ablation_row = "S4"

    resolved = train.resolve_fcr_training_options(args)

    assert resolved.fcr_version == "v2"
    assert resolved.fcr_matrix_row == "S4"
    assert resolved.use_fcr is True


@pytest.mark.parametrize("row", ("C2", "S0"))
def test_identity_noop_rows_have_no_auxiliary_losses(row: str) -> None:
    args = train.resolve_fcr_training_options(_args(row=row))
    ready = FCRV2CapabilityState(True, True, True, True, {})

    assert args.fcr_identity_only is True
    assert FCRV2Schedule().state(epoch=120, row=row, capabilities=ready).active_losses == frozenset()
    assert args.fcr_v2_active_losses == ()


def test_v1_row_contract_is_unchanged() -> None:
    args = _args(row="", version="v1")
    args.fcr_ablation_row = "R7"

    resolved = train.resolve_fcr_training_options(args)

    assert resolved.fcr_version == "v1"
    assert resolved.fcr_ablation_row == "R7"
    assert resolved.fcr_physics_ordered_decoder is True
    assert resolved.fcr_three_axis_intervention is False


@pytest.mark.parametrize(
    ("row", "identity_only"),
    (
        ("C2", True),
        ("C3", False),
        ("S0", True),
        ("S1", False),
        ("S2", False),
        ("S3", False),
        ("S4", False),
    ),
)
def test_pre_m1_v2_rows_route_their_real_formal_identity_schema(
    row: str,
    identity_only: bool,
) -> None:
    args = train.resolve_fcr_training_options(_args(row=row))
    model = _small_model().eval()
    with torch.no_grad():
        outputs = model(
            torch.randn(2, 2, 64),
            return_aux=True,
            fcr_identity_only=bool(args.fcr_identity_only),
        )

    assert args.fcr_identity_only is identity_only
    assert outputs["feature_schema"] == "ADV3B02:FCR:z_f_id:unit_l2:160:v2"
    assert outputs["fcr_tx_logits"].shape == (2, 3)
    assert outputs["z_f_id"].shape == (2, 160)
    assert (outputs["fcr_decode"] is None) is identity_only
    legacy_logits = outputs["tx_logits"]

    routed = train.route_formal_identity_outputs(outputs, use_fcr=True)

    assert routed["tx_logits"] is outputs["fcr_tx_logits"]
    assert routed["z_id"] is outputs["z_f_id"]
    assert outputs["tx_logits"] is legacy_logits


def test_formal_identity_route_without_model_still_rejects_unknown_schema() -> None:
    outputs = {
        "tx_logits": torch.zeros(2, 3),
        "fcr_tx_logits": torch.ones(2, 3),
        "z_id": torch.zeros(2, 160),
        "z_f_id": torch.ones(2, 160),
        "feature_schema": "ADV3B02:FCR:unknown",
    }

    with pytest.raises(ValueError, match="incompatible feature schema"):
        train.route_formal_identity_outputs(outputs, use_fcr=True)


def test_forward_identity_only_does_not_run_decoder() -> None:
    torch.manual_seed(6001)
    model = _small_model().eval()
    assert model.fcr is not None
    model.fcr.decoder.forward = Mock(side_effect=AssertionError("decoder called"))

    with torch.no_grad():
        output = model.forward_identity_only(torch.randn(2, 2, 64))

    assert output["tx_logits"].shape == (2, 3)
    assert output["z_id"].shape == (2, 160)
    assert output["fcr_decode"] is None
    model.fcr.decoder.forward.assert_not_called()


def test_identity_projection_accepts_lite_c_checkpoint_geometry_without_decoder() -> None:
    torch.manual_seed(6007)
    model = build_dual_model(
        num_classes=3,
        num_domains=4,
        model_size="S",
        dataset="wisig",
        input_len=128,
        model_variant="lite_c",
        fast_infer_when_no_aux=False,
        use_fcr=True,
        fcr_version="v2",
    ).eval()
    assert model.emb_dim == 192
    assert model.fcr is not None
    model.fcr.decoder.forward = Mock(side_effect=AssertionError("decoder called"))

    with torch.no_grad():
        output = model.forward_identity_only(torch.zeros(2, 2, 128))

    assert output["tx_logits"].shape == (2, 3)
    assert output["z_id"].shape == (2, 160)
    model.fcr.decoder.forward.assert_not_called()


def test_v2_loads_mature_checkpoint_and_copies_identity_head(tmp_path: Path) -> None:
    torch.manual_seed(6002)
    mature = _small_model(use_fcr=False, fcr_version="v1")
    checkpoint = tmp_path / "mature.pth"
    torch.save(
        {
            "model": mature.state_dict(),
            "epoch": 200,
            "candidate_id": "ADV3B02_CORE90_SOFT_E200",
            "args": {"seed": 392005},
        },
        checkpoint,
    )
    candidate = _small_model()

    report = train.load_init_checkpoint_weights(
        candidate,
        str(checkpoint),
        torch.device("cpu"),
        expected_seed=392005,
        expected_epoch=200,
        expected_candidate_id="ADV3B02_CORE90_SOFT_E200",
        require_mature_base_complete=True,
    )

    assert report["expected_seed"] == 392005
    assert report["actual_epoch"] == 200
    assert report["source_only"] is True
    assert candidate.fcr_identity_head_matches_legacy()


def test_v2_final_checkpoint_bundle_is_versioned_and_strictly_reloadable(tmp_path: Path) -> None:
    torch.manual_seed(6008)
    model = _small_model()
    checkpoint = tmp_path / "final.pth"
    args = SimpleNamespace(fcr_version="v2", fcr_matrix_row="M6")

    checkpoint_module.save_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=200,
        args=args,
        split_info={"source_only": True},
        stats={"checkpoint_selection": "final_only"},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = _small_model()
    checkpoint_module.load_fcr_checkpoint_strict(checkpoint, restored)

    assert payload["fcr_bundle"]["bundle_schema"].endswith(".v2")
    assert payload["fcr_bundle"]["feature_schema"].endswith(":v2")
    assert payload["fcr_bundle"]["fcr_version"] == "v2"
    assert payload["stats"]["checkpoint_selection"] == "final_only"


def test_v2_optimizer_groups_are_exhaustive_and_non_overlapping() -> None:
    model = _small_model()

    groups = train.build_fcr_v2_optimizer_param_groups(model, base_lr=1.0e-4, weight_decay=1.0e-3)
    parameters = [parameter for group in groups for parameter in group["params"]]

    assert {group["name"] for group in groups} >= {
        "backbone_early",
        "backbone_late",
        "identity_head",
        "identity_projection",
        "fcr_new",
    }
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_u_s_loss_filter_keeps_only_source_unlabeled_allowlist() -> None:
    reference = torch.tensor(1.0, requires_grad=True)
    components = {
        name: reference * float(index + 1)
        for index, name in enumerate(
            ("identity_ce", "prototype", "tail", "self", "shared_f", "shared_s", "response", "eta", "swap", "cycle", "need", "transplant", "physical", "factor")
        )
    }
    output = train.compute_fcr_v2_routed_losses(
        components,
        row="M6",
        epoch=150,
        role="U_s",
        capabilities=FCRV2CapabilityState(True, True, True, True, {}),
    )

    assert output.active_losses <= frozenset({"self", "shared_f", "shared_s", "response", "eta"})
    assert output.weights["identity_ce"] == 0.0
    assert output.weights["prototype"] == 0.0
    assert output.weights["tail"] == 0.0
    for denied in ("swap", "cycle", "need", "transplant", "physical", "factor"):
        assert output.weights[denied] == 0.0


def _v2_training_pair(model, *, role: str = "L_s"):
    batch = 4
    clean = torch.randn(batch, 2, 64)
    labels = torch.tensor([0, 0, 1, 1]) if role == "L_s" else torch.full((batch,), -1)
    physical_ids = tuple(f"physical:{index}" for index in range(batch))
    metadata = {
        "physical_sample_id": physical_ids,
        "content_record_id": tuple(f"content:{index}" for index in range(batch)),
        "common_preamble_id": ("wisig-common-preamble",) * batch,
        "excitation_bin": torch.zeros(batch, dtype=torch.long),
        "link_condition": ("eq:0",) * batch,
        "crop_offset": torch.tensor([0, 64, 0, 64]),
        "rx_i": torch.zeros(batch, dtype=torch.long),
        "day_i": torch.zeros(batch, dtype=torch.long),
        "eq_i": torch.zeros(batch, dtype=torch.long),
        "sig_i": torch.tensor([0, 1, 0, 1]),
    }
    augmentation = SimpleNamespace(
        x=clean + 0.01 * torch.randn_like(clean),
        scenario="leo_clear_weak",
        physical_sample_id=physical_ids,
        crop_offset=metadata["crop_offset"],
        eta=torch.ones(batch, 9),
        eta_valid_mask=torch.ones(batch, 9, dtype=torch.bool),
        eta_schema_version="fcr-v2/eta-v1",
        eta_fields=(
            "snr_db", "cfo_hz", "residual_cfo_hz", "fD_hz", "pl_db",
            "K_db", "theta_deg", "h_km", "state",
        ),
        eta_units=("dB", "Hz", "Hz", "Hz", "dB", "dB", "degree", "km", "category_index"),
        eta_scales=(20.0, 100000.0, 100000.0, 100000.0, 200.0, 20.0, 90.0, 2000.0, 2.0),
        scenario_by_sample=("leo_clear_weak",) * batch,
    )
    return train.build_fcr_v2_training_pair(
        clean,
        augmentation,
        labels,
        metadata,
        role=role,
        epoch=150,
        seed=392005,
    )


def test_m6_real_v2_objective_uses_metadata_pairs_and_backpropagates() -> None:
    torch.manual_seed(6003)
    model = _small_model().train()
    pair = _v2_training_pair(model)

    output = train.compute_fcr_v2_pair_objective(
        model=model,
        raw_model=model,
        training_pair=pair,
        row="M6",
        epoch=150,
        supervised_components={"identity_ce": torch.tensor(0.5, requires_grad=True)},
        frozen_identity_classifier=train.FrozenADV3B02IdentityClassifier(model),
        collect_gradient_evidence=True,
    )
    output.total.backward()

    assert torch.isfinite(output.total)
    assert output.metrics["active_nuisance_pairs"] > 0
    assert output.metrics["active_fingerprint_pairs"] > 0
    assert output.metrics["factor_fingerprint_axis_loss"] > 0.0
    assert output.metrics["factor_content_axis_loss"] > 0.0
    assert output.metrics["factor_nuisance_axis_loss"] > 0.0
    assert output.components["factor"] > 0.0
    assert output.metrics["gradient_ratios_to_identity_ce"]["factor"] > 0.0
    assert any(parameter.grad is not None for parameter in model.fcr.parameters())


@pytest.mark.parametrize(
    ("row", "component", "raw_loss", "gradient_key"),
    (
        ("M2", "latent_cycle", "cycle", "cycle"),
        ("M4", "transplant", "transplant", "transplant"),
    ),
)
def test_m2_and_m4_use_real_decode_reencode_graphs(
    row: str,
    component: str,
    raw_loss: str,
    gradient_key: str,
) -> None:
    torch.manual_seed(6012)
    model = _small_model().train()
    pair = _v2_training_pair(model)
    classifier = train.FrozenADV3B02IdentityClassifier(model)

    output = train.compute_fcr_v2_pair_objective(
        model=model,
        raw_model=model,
        training_pair=pair,
        row=row,
        epoch=150,
        supervised_components={"identity_ce": torch.tensor(0.5, requires_grad=True)},
        frozen_identity_classifier=classifier,
        collect_gradient_evidence=True,
    )

    assert output.metrics["raw_losses"][raw_loss] > 0.0
    assert output.components[component] > 0.0
    assert output.metrics["gradient_ratios_to_identity_ce"][gradient_key] > 0.0
    assert all(not parameter.requires_grad for parameter in classifier.parameters())


def test_m5_physical_gates_and_response_smoothness_are_on_the_real_graph() -> None:
    torch.manual_seed(6011)
    model = _small_model().train()
    pair = _v2_training_pair(model)

    output = train.compute_fcr_v2_pair_objective(
        model=model,
        raw_model=model,
        training_pair=pair,
        row="M5",
        epoch=150,
        supervised_components={"identity_ce": torch.tensor(0.5, requires_grad=True)},
        frozen_identity_classifier=train.FrozenADV3B02IdentityClassifier(model),
        collect_gradient_evidence=True,
    )

    assert output.metrics["physical_clean_gate_loss"] > 0.0
    assert output.metrics["physical_leo_gate_loss"] > 0.0
    assert output.metrics["response_surface_smoothness"] > 0.0
    assert output.components["phys"] > 0.0
    assert output.metrics["gradient_ratios_to_identity_ce"]["physical"] > 0.0


def test_c2_objective_does_not_call_decoder() -> None:
    torch.manual_seed(6004)
    model = _small_model().train()
    model.fcr_identity_only_training = True
    pair = _v2_training_pair(model)
    model.fcr.decoder.forward = Mock(side_effect=AssertionError("decoder called"))

    output = train.compute_fcr_v2_pair_objective(
        model=model,
        raw_model=model,
        training_pair=pair,
        row="C2",
        epoch=150,
        supervised_components={"identity_ce": torch.tensor(0.5, requires_grad=True)},
    )

    torch.testing.assert_close(output.components["id"], torch.tensor(0.5))
    assert output.components["self"].item() == 0.0
    assert output.components["shared"].item() == 0.0
    model.fcr.decoder.forward.assert_not_called()


def test_defer_target_source_writes_diagnostics_before_return() -> None:
    source = Path(train.__file__).read_text(encoding="utf-8")
    helper = source[source.index("def finalize_fcr_v2_diagnostics_before_return") : source.index("def validate_fcr_pair_for_role")]

    assert 'getattr(args, "final_save_path", "")' in helper
    assert helper.index("write_fcr_v2_diagnostics") < helper.index("[TARGET-EVAL-DEFERRED]")
    assert "query" not in json.dumps(train.fcr_dry_run_payload(train.resolve_fcr_training_options(_args(row="M6"))))
