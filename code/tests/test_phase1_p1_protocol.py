import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.leakage_probe import frozen_ridge_linear_probe  # noqa: E402
from training_controls import satellite_protocol_manifest  # noqa: E402
from SSDG import train_ssdg  # noqa: E402


def test_satellite_protocol_requires_disjoint_scenario_families():
    manifest = satellite_protocol_manifest(
        ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        ["clear_leo", "low_elev_leo", "rain_leo", "storm_mp", "geo_clear", "mixed_orbit"],
        require_disjoint=True,
    )
    assert manifest["disjoint"] is True
    assert manifest["scenario_overlap"] == []
    assert manifest["family_overlap"] == []
    assert manifest["train_families"] == ["simplified_leo_residual_weak_v1"]
    assert manifest["eval_families"] == ["legacy_satellite_physics_holdout_v1"]
    assert manifest["train_channel_implementations"] == ["leo_residual"]
    assert manifest["eval_channel_implementations"] == ["legacy_full"]
    assert manifest["channel_implementation_overlap"] == []
    assert manifest["config_hash_overlap"] == []
    assert len(manifest["registry_sha256"]) == 64


def test_satellite_protocol_fails_closed_on_in_family_evaluation():
    with pytest.raises(ValueError, match="not held-out"):
        satellite_protocol_manifest(
            ["leo_clear_weak", "leo_low_elev_weak"],
            ["leo_rain_weak"],
            require_disjoint=True,
        )


def test_satellite_protocol_same_implementation_fails_even_when_families_are_distinct(monkeypatch):
    from training_controls import SAT_CHANNEL_PROTOCOL_FAMILIES

    monkeypatch.setitem(SAT_CHANNEL_PROTOCOL_FAMILIES, "leo_rain_weak", "synthetic_distinct_label_only")
    with pytest.raises(ValueError, match="channel_implementation_overlap=leo_residual"):
        satellite_protocol_manifest(
            ["leo_clear_weak"],
            ["leo_rain_weak"],
            require_disjoint=True,
        )


def test_frozen_ridge_probe_reports_nonempty_accuracy_chance_and_excess():
    train_x = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=torch.float32
    )
    train_y = torch.tensor([0, 0, 1, 1])
    eval_x = torch.tensor([[0.95, 0.0], [-0.95, 0.0]], dtype=torch.float32)
    eval_y = torch.tensor([0, 1])

    result = frozen_ridge_linear_probe(train_x, train_y, eval_x, eval_y, ridge=0.01)

    assert result["status"] == "COMPLETE"
    assert result["train_count"] == 4
    assert result["eval_count"] == 2
    assert result["class_count"] == 2
    assert result["accuracy"] == 1.0
    assert result["chance_accuracy"] == 0.5
    assert result["excess_accuracy"] == 0.5
    assert result["balanced_chance_accuracy"] == 0.5
    assert result["balanced_excess_accuracy"] == 0.5


def test_train_ssdg_source_enforces_final_only_selection_and_final_export():
    source = (CODE_ROOT / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")
    assert 'choices=["final_only"]' in source


def test_open_set_gradient_control_covers_source_episode_before_direct_metric_start():
    source = (CODE_ROOT / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")
    assert 'os_control_epoch_ready = bool(getattr(args, "phase1_v2_os_eff_all_phases", True))' in source
    assert "and os_control_epoch_ready" in source
    assert "and open_loss_has_signal" in source
    assert 'parser.add_argument("--os_eff_max_budget"' in source
    assert 'parser.add_argument("--max_grad_norm"' in source
    assert 'selected_checkpoint = final_path' in source
    assert 'default_export_checkpoint = selected_checkpoint' in source
    assert 'save_payload(best_path, payload)' not in source
    assert 'save_payload(latest_path, payload)' not in source
    assert 'save_payload(tail_reference_path, reference_payload)' not in source
    assert 'tail_rejected_E' not in source
    assert '"Phase1 final-only mode forbids tail checkpoint rollback' in source
    assert source.count("save_payload(selected_checkpoint, final_payload)") == 1
    assert '"selection_source": "training_final_only"' in source


def test_source_val_satellite_eval_forces_source_val_loader_instead_of_main_test_alias(monkeypatch):
    captured = {}

    def fake_eval(model, loaders, device, domain_map, scenario_names, args, max_batches):
        captured["loader_names"] = list(loaders)
        captured["eval_sat_on"] = args.eval_sat_on
        return {"clear_leo": {"aggregate": {"tx_total": 8, "tx_acc": 50.0}}}

    monkeypatch.setattr(train_ssdg, "evaluate_sat_scenarios", fake_eval)
    args = SimpleNamespace(
        eval_sat_channel=True,
        eval_sat_scenario_list=["clear_leo"],
        eval_sat_on="main",
        sat_eval_max_batches=-1,
    )
    result = train_ssdg._evaluate_source_val_sat_if_enabled(
        object(),
        {"val_loader": object(), "domain_label_map": {}},
        torch.device("cpu"),
        args,
    )

    assert result["clear_leo"]["aggregate"]["tx_total"] == 8
    assert captured["loader_names"] == ["source_val"]
    assert captured["eval_sat_on"] == "all"


def test_channel_view_labels_exclude_clean_duplicates_when_satellite_transform_is_skipped():
    skipped = train_ssdg._channel_view_labels(8, 4, False, torch.device("cpu"))
    applied = train_ssdg._channel_view_labels(8, 4, True, torch.device("cpu"))

    assert skipped.tolist() == [0] * 8
    assert applied.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_ssdg_dry_run_fails_closed_when_schedule_hides_eval_family_overlap(tmp_path):
    args = train_ssdg.build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path),
            "--dry_run",
            "--sat_train_scenarios",
            "leo_clear_weak",
            "--sat_view_schedule",
            "1@0.5:leo_clear_weak;10@0.8:leo_rain_weak",
            "--eval_sat_channel",
            "true",
            "--eval_sat_scenarios",
            "leo_low_elev_weak",
            "--sat_protocol_disjoint_required",
            "true",
        ]
    )
    with pytest.raises(ValueError, match="not held-out"):
        train_ssdg.train(args)


def test_source_val_heavy_eval_schedule_is_sparse_then_dense_and_final_mandatory():
    args = SimpleNamespace(
        source_val_heavy_eval_start_epoch=10,
        source_val_heavy_eval_interval=10,
        source_val_heavy_eval_final_window=20,
        source_val_heavy_eval_final_interval=2,
    )
    observed = [
        epoch
        for epoch in range(1, 201)
        if train_ssdg._should_run_source_val_heavy_eval(epoch, 200, args)
    ]
    assert observed == list(range(10, 181, 10)) + list(range(182, 201, 2))
    args.source_val_heavy_eval_interval = 999
    args.source_val_heavy_eval_final_interval = 999
    assert train_ssdg._should_run_source_val_heavy_eval(200, 200, args) is True


def test_terminal_status_distinguishes_missing_p1_mechanisms():
    status = train_ssdg._resolve_phase1_terminal_status(
        tail_stopped=False,
        export_failed=False,
        final_blocked=False,
        selected_checkpoint_exists=True,
        heldout_eval_status="COMPLETE",
        p0_mechanisms_ready=True,
        p1_mechanisms_ready=False,
        endpoint_export_ready=True,
    )
    assert status == "NON_PROMOTABLE_P1_DISABLED"
