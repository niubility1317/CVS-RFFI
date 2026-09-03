from __future__ import annotations

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

import post_stage_common  # noqa: E402
import train  # noqa: E402
from SSDG import train_ssdg as ssdg_train  # noqa: E402
from baseline_origin_sat_view import (  # noqa: E402
    BaselineOriginSatViewAugment,
    CRRA_NUISANCE_FIELDS,
    CRRA_NUISANCE_SCALES,
    CRRA_NUISANCE_UNITS,
)
from cvsrffi.phase1_fcr_types import FCRV2FactorOutput  # noqa: E402
from cvsrffi.phase1_fcr_v2_diagnostics import collect_fcr_v2_diagnostics  # noqa: E402
from cvsrffi.phase1_fcr_v2_losses import (  # noqa: E402
    LossMagnitudeEMA,
    asymmetric_clean_teacher_loss,
    complex_physical_gram_loss,
    fingerprint_separation_loss,
    necessity_loss,
    per_class_tail_cvar_loss,
    response_surface_smoothness,
)
from cvsrffi.phase1_fcr_v2_metadata import build_fcr_v2_metadata  # noqa: E402
from cvsrffi.phase1_fcr_v2_schedule import FCRV2Schedule  # noqa: E402
from dataset_wisig import derive_wisig_fcr_metadata  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402
from scripts import predict_phase1_truth_last as truth_last_predictor  # noqa: E402
from scripts import score_phase1_truth_last as truth_last_scorer  # noqa: E402
from scripts.predict_phase1_truth_last import predict_identity_logits  # noqa: E402


def _eta_meta(batch: int) -> dict[str, torch.Tensor | str]:
    return {
        "scenario": "leo_clear_weak",
        **{
            name: torch.full((batch,), float(index + 1))
            for index, name in enumerate(CRRA_NUISANCE_FIELDS)
        },
    }


def _strict_batch_meta(ids: tuple[str, ...]) -> dict[str, object]:
    batch = len(ids)
    return {
        "physical_sample_id": ids,
        "content_record_id": tuple(f"record:{index}" for index in range(batch)),
        "crop_offset": torch.zeros(batch, dtype=torch.long),
        "common_preamble_id": tuple("preamble:verified" for _ in range(batch)),
        "rx_i": torch.zeros(batch, dtype=torch.long),
        "day_i": torch.zeros(batch, dtype=torch.long),
        "link_condition": tuple("eq:1" for _ in range(batch)),
        "excitation_bin": torch.zeros(batch, dtype=torch.long),
    }


def _augment(ids: tuple[str, ...], *, batch_idx: int, order: torch.Tensor | None = None):
    if order is None:
        order = torch.arange(len(ids))
    ordered_ids = tuple(ids[int(index)] for index in order)
    x = torch.arange(len(ids) * 16, dtype=torch.float32).view(len(ids), 2, 8)[order]

    def apply_fn(iq, scenario, args, gen=None, return_meta=False):
        del args, return_meta
        noise = torch.rand(iq.shape, generator=gen, device=iq.device)
        return iq + noise, _eta_meta(int(iq.size(0))) | {"scenario": scenario}

    augment = BaselineOriginSatViewAugment(
        scenarios=("leo_clear_weak", "leo_rain_weak"),
        p=1.0,
        seed=392005,
        apply_fn=apply_fn,
    )
    return augment.transform(
        x,
        args=SimpleNamespace(),
        epoch=37,
        batch_idx=batch_idx,
        batch_meta={
            "physical_sample_id": ordered_ids,
            "crop_offset": torch.zeros(len(ids), dtype=torch.long),
        },
    )


def test_eta_schema_is_named_unit_bound_and_clean_rows_are_unobserved() -> None:
    ids = ("p0", "p1")
    view = _augment(ids, batch_idx=1)
    assert view.eta_fields == CRRA_NUISANCE_FIELDS
    assert view.eta_units == CRRA_NUISANCE_UNITS
    assert view.eta_scales == tuple(CRRA_NUISANCE_SCALES[name] for name in CRRA_NUISANCE_FIELDS)

    batch = _strict_batch_meta(ids)
    batch["tx_id"] = torch.tensor([0, 1])
    metadata = build_fcr_v2_metadata(batch, view)
    assert metadata.eta_fields == CRRA_NUISANCE_FIELDS
    assert not bool(metadata.eta_valid_mask[:2].any())
    assert bool(metadata.eta_valid_mask[2:].all())

    view.eta_units = tuple("wrong" for _ in CRRA_NUISANCE_FIELDS)
    with pytest.raises(ValueError, match="eta_units"):
        build_fcr_v2_metadata(batch, view)


def test_satellite_random_key_is_per_physical_sample_and_order_invariant() -> None:
    ids = ("p0", "p1", "p2", "p3")
    forward = _augment(ids, batch_idx=1)
    permutation = torch.tensor([2, 0, 3, 1])
    reordered = _augment(ids, batch_idx=999_999, order=permutation)
    forward_map = {
        sample_id: (forward.x[index], forward.eta[index], forward.scenario_by_sample[index])
        for index, sample_id in enumerate(forward.physical_sample_id)
    }
    reordered_map = {
        sample_id: (reordered.x[index], reordered.eta[index], reordered.scenario_by_sample[index])
        for index, sample_id in enumerate(reordered.physical_sample_id)
    }
    assert forward_map.keys() == reordered_map.keys()
    for sample_id in forward_map:
        torch.testing.assert_close(forward_map[sample_id][0], reordered_map[sample_id][0])
        torch.testing.assert_close(forward_map[sample_id][1], reordered_map[sample_id][1])
        assert forward_map[sample_id][2] == reordered_map[sample_id][2]


def test_manysig_index_metadata_is_verifiable_and_does_not_fabricate_preamble() -> None:
    derived = derive_wisig_fcr_metadata(
        tx_i=2,
        rx_i=3,
        day_i=4,
        eq_i=1,
        sig_i=19,
        crop_offset=0,
        equalized_value=True,
        physical_sample_id="tx2:rx3:day4:eq1:sig19",
    )
    assert derived["content_record_id"] == "tx2:rx3:day4:eq1:sig19"
    assert derived["link_condition"] == "eq:1"
    assert derived["common_preamble_id"] == ""
    assert derived["excitation_bin"] == -1
    assert derived["fingerprint_pair_capability"] == "MECHANISM_NOT_ACTIVATED:unverifiable_manysig_preamble_and_excitation"


def test_training_pair_rejects_missing_scientific_metadata_instead_of_synthesizing() -> None:
    ids = ("p0", "p1")
    view = _augment(ids, batch_idx=1)
    incomplete = _strict_batch_meta(ids)
    incomplete.pop("common_preamble_id")
    with pytest.raises(ValueError, match="common_preamble_id"):
        train.build_fcr_v2_training_pair(
            torch.zeros(2, 2, 8), view, torch.tensor([0, 1]), incomplete,
            role="L_s", epoch=37, seed=392005,
        )


def test_shared_teacher_gradient_tail_and_persistent_ema_are_real() -> None:
    clean = torch.tensor([[1.0, 0.0]], requires_grad=True)
    leo = torch.tensor([[0.0, 1.0]], requires_grad=True)
    shared = asymmetric_clean_teacher_loss(clean, leo)
    shared.backward()
    assert clean.grad is None or torch.count_nonzero(clean.grad) == 0
    assert leo.grad is not None and torch.count_nonzero(leo.grad) > 0

    per_sample = torch.tensor([0.1, 0.2, 4.0, 5.0], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    tail = per_class_tail_cvar_loss(per_sample, labels, tail_fraction=0.5)
    assert tail.item() == pytest.approx(5.0)

    ema = LossMagnitudeEMA(decay=0.5)
    first = ema.normalize("self", torch.tensor(2.0))
    second = ema.normalize("self", torch.tensor(6.0))
    assert ema.updates["self"] == 2
    assert first.item() != pytest.approx(second.item())


def test_m_rows_inherit_true_swap_and_keep_advanced_losses_during_refinement() -> None:
    ready = SimpleNamespace(
        eta_ready=True, decoder_ready=True, swap_ready=True, fingerprint_ready=True,
        reason_for=lambda _name: None,
    )
    for row in ("M1", "M2", "M3", "M4", "M5", "M6"):
        state = FCRV2Schedule().state(epoch=120, row=row, capabilities=ready)
        assert "swap" in state.active_losses
        final = FCRV2Schedule().state(epoch=180, row=row, capabilities=ready)
        assert "swap" in final.active_losses
        assert 0.10 <= final.scales["swap"] <= 0.25


def test_corrected_necessity_hinge_rewards_larger_drop_gap() -> None:
    full = torch.tensor(2.0, requires_grad=True)
    drop = torch.tensor(2.02, requires_grad=True)
    loss = necessity_loss(full_error=full, drop_error=drop, relative_margin=0.05)
    assert loss.item() > 0.0
    loss.backward()
    assert drop.grad is not None and drop.grad.item() < 0.0

    satisfied = necessity_loss(
        full_error=torch.tensor(2.0), drop_error=torch.tensor(5.0), relative_margin=0.05
    )
    assert satisfied.item() == 0.0


def test_complex_gram_and_response_surface_preserve_phase_structure() -> None:
    phase = torch.linspace(0.0, 1.0, 16)
    first = torch.exp(1j * phase).unsqueeze(0)
    second = torch.exp(1j * (phase.square() + 0.3)).unsqueeze(0)
    assert complex_physical_gram_loss(first, second).item() > 0.0

    smooth = torch.exp(1j * phase).unsqueeze(0)
    perturbed = smooth.clone()
    perturbed[:, 8] *= -1
    assert response_surface_smoothness(perturbed) > response_surface_smoothness(smooth)


def test_different_tx_fingerprint_loss_separates_instead_of_pulling_together() -> None:
    source = torch.tensor([[1.0, 0.0]], requires_grad=True)
    destination = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss = fingerprint_separation_loss(source, destination, cosine_margin=0.2)
    assert loss.item() > 0.0
    loss.backward()
    assert source.grad is not None and destination.grad is not None


def test_v2_checkpoint_rebuild_and_predictor_use_identity_only_path() -> None:
    args = SimpleNamespace(
        num_classes=3, num_domains=2, model_size="S", dataset="wisig", input_len=64,
        sample_rate_hz=25e6, model_variant="lite_d", branch_ablation="none",
        use_fcr=True, fcr_version="v2", fcr_decoder_mode="full_physics",
    )
    model = post_stage_common.build_baseline_model(args, torch.device("cpu")).eval()
    assert model.fcr_version == "v2"
    model.fcr.decoder.forward = Mock(side_effect=AssertionError("predictor invoked decoder"))
    with torch.no_grad():
        logits = predict_identity_logits(model, torch.randn(2, 2, 64))
    assert logits.shape == (2, 3)
    model.fcr.decoder.forward.assert_not_called()


def test_c2_s4_m6_final_checkpoints_close_formal_predictor_and_scorer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_ids = ["opaque-sample-0", "opaque-sample-1"]
    input_package = tmp_path / "predictor_package"
    input_package.mkdir()
    (input_package / "manifest.json").write_text(
        json.dumps({"sample_ids": sample_ids}), encoding="utf-8"
    )

    class TinyOpaqueDataset:
        def __init__(self, package_root: Path):
            manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
            self.sample_ids = list(manifest["sample_ids"])

        def __len__(self) -> int:
            return len(self.sample_ids)

        def __getitem__(self, index: int):
            generator = torch.Generator().manual_seed(9000 + int(index))
            return (
                torch.randn(2, 256, generator=generator),
                -1,
                0,
                {"physical_sample_id": self.sample_ids[index]},
            )

    monkeypatch.setattr(truth_last_predictor, "_OpaqueTargetDataset", TinyOpaqueDataset)
    monkeypatch.setattr(
        truth_last_predictor,
        "apply_sat_channel_for_scenario",
        lambda x, scenario, args, **kwargs: (x, None),
    )
    original_builder = truth_last_predictor.build_exact_ssdg_model_from_checkpoint
    audits: list[tuple[str, dict[str, object], Mock]] = []

    def audited_builder(checkpoint, **kwargs):
        model, audit = original_builder(checkpoint, **kwargs)
        decoder_guard = Mock(side_effect=AssertionError("formal predictor invoked V2 decoder"))
        model.fcr.decoder.forward = decoder_guard
        audits.append((str(checkpoint["args"]["fcr_matrix_row"]), audit, decoder_guard))
        return model, audit

    monkeypatch.setattr(
        truth_last_predictor,
        "build_exact_ssdg_model_from_checkpoint",
        audited_builder,
    )

    for row in ("C2", "S4", "M6"):
        checkpoint_overrides = {
            "num_classes": 6,
            "num_domains": 2,
            "model_size": "S",
            "dataset": "wisig",
            "input_len": 256,
            "sample_rate_hz": 25e6,
            "model_variant": "lite_d",
            "branch_ablation": "none",
            "use_fcr": True,
            "fcr_requested": True,
            "fcr_version": "v2",
            "fcr_matrix_row": row,
            "fcr_ablation_row": row,
            "fcr_decoder_mode": "identity_initialized",
            "fcr_identity_only": row == "C2",
            "sat_seed": 2027,
        }
        parsed = ssdg_train.build_arg_parser().parse_args(
            ["--output_dir", str(tmp_path / f"{row}_parser_output")]
        )
        for key, value in checkpoint_overrides.items():
            setattr(parsed, key, value)
        checkpoint_args = dict(vars(parsed))
        model_args = ssdg_train.merge_checkpoint_args(
            {"args": checkpoint_args}, parsed, input_len=256, num_domains=2
        )
        model_args = ssdg_train._apply_model_cli_args(model_args, parsed)
        reference = ssdg_train.build_baseline_model(
            model_args, torch.device("cpu")
        )
        checkpoint_path = tmp_path / f"{row}_final.pth"
        torch.save(
            {"args": checkpoint_args, "model": reference.state_dict(), "epoch": 200},
            checkpoint_path,
        )
        output_root = tmp_path / f"{row}_output"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict_phase1_truth_last.py",
                "--checkpoint", str(checkpoint_path),
                "--output-root", str(output_root),
                "--input-package", str(input_package),
                "--run-id", "formal-v2-closure",
                "--row-id", row,
                "--batch-size", "2",
                "--num-workers", "0",
                "--device", "cpu",
                "--mode", "predict",
            ],
        )
        truth_last_predictor.main()
        predictions_path = output_root / "predictions.json"
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
        assert payload["record_count"] == len(sample_ids) * 4
        assert {record["scenario"] for record in payload["records"]} == {
            "clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"
        }
        assert {record["row_id"] for record in payload["records"]} == {row}

        truth_path = output_root / "truth_sidecar.json"
        truth_path.write_text(
            json.dumps(
                {
                    "schema": "cvs.phase1.truth_sidecar.v1",
                    "record_count": len(sample_ids),
                    "records": [
                        {"sample_id": sample_id, "label": index}
                        for index, sample_id in enumerate(sample_ids)
                    ],
                }
            ),
            encoding="utf-8",
        )
        score_path = output_root / "score.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "score_phase1_truth_last.py",
                "--predictions", str(predictions_path),
                "--truth", str(truth_path),
                "--output", str(score_path),
            ],
        )
        truth_last_scorer.main()
        score = json.loads(score_path.read_text(encoding="utf-8"))
        assert score["record_count"] == len(sample_ids) * 4

    assert [row for row, _audit, _guard in audits] == ["C2", "S4", "M6"]
    for _row, audit, decoder_guard in audits:
        assert audit["checkpoint_load_strict"] is True
        assert audit["missing_keys"] == 0
        assert audit["unexpected_keys"] == 0
        decoder_guard.assert_not_called()


def test_diagnostics_require_actual_pair_loss_weight_and_gradient_evidence() -> None:
    artifacts = {
        "z_f_id": torch.eye(2),
        "z_tx_state": torch.eye(2),
        "z_s": torch.eye(2),
        "tx_labels": torch.tensor([0, 1]),
        "domain_labels": torch.tensor([0, 1]),
        "probe_train_mask": torch.tensor([True, True]),
        "probe_eval_mask": torch.tensor([True, True]),
    }
    metrics = collect_fcr_v2_diagnostics(
        artifacts,
        resources={
            "configured_lambdas": ("self", "swap", "factor"),
            "pair_counts": {"nuisance": 3, "content": 0, "fingerprint": 0},
            "pair_opportunities": {"nuisance": 4, "content": 4, "fingerprint": 4},
            "effective_weights": {"self": 0.1, "swap": 0.05, "factor": 0.05},
            "nonzero_loss_steps": {"self": 2, "swap": 0, "factor": 0},
            "gradient_ratios_to_identity_ce": {"self": 0.4, "swap": 0.0, "factor": 0.0},
        },
        row_id="M6",
    )
    assert metrics["pair_count"] == 3
    assert metrics["pair_coverage"] == pytest.approx(0.25)
    assert metrics["activation_state"]["configured_lambdas"] == ["self", "swap", "factor"]
    assert metrics["activation_state"]["actual_active_lambdas"] == ["self"]
    assert metrics["activation_state"]["mechanism_status"]["swap"].startswith("MECHANISM_NOT_ACTIVATED")
    assert metrics["activation_state"]["mechanism_status"]["factor"].startswith("MECHANISM_NOT_ACTIVATED")
