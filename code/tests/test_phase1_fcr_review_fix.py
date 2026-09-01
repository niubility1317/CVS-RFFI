from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import train  # noqa: E402
from cvsrffi import eval as eval_module  # noqa: E402
from cvsrffi import losses as losses_module  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


FEATURE_SCHEMA = "ADV3B02:FCR:z_f_id:unit_l2:160:v1"
SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_adv3b02_fcr_20260901.sh"


def _formal_options(**overrides):
    values = {
        "phase1_method": "adv3b02_fcr",
        "use_fcr": True,
        "fcr_ablation_row": "R8",
        "epochs": 200,
        "train_mode": "centralized",
        "use_concat_sat_channel_aug": False,
        "use_meta_ssl_cvs": True,
        "ssl_labeled_ratio": 0.07,
        "ssl_unlabeled_ratio": 0.63,
        "ssl_val_ratio": 0.30,
        "lambda_fcr_self": 1.0,
        "lambda_fcr_swap": 1.0,
        "lambda_fcr_shared": 1.0,
        "lambda_fcr_cycle": 1.0,
        "lambda_fcr_eta": 1.0,
        "lambda_fcr_factor": 1.0,
        "lambda_fcr_need": 1.0,
        "lambda_fcr_phys": 1.0,
    }
    values.update(overrides)
    return train.resolve_fcr_training_options(argparse.Namespace(**values))


@pytest.mark.parametrize(
    "deviation",
    [
        {"use_meta_ssl_cvs": False},
        {"ssl_labeled_ratio": 0.0701},
        {"ssl_unlabeled_ratio": 0.6301},
        {"ssl_val_ratio": 0.3001},
    ],
)
def test_p1_1_formal_fcr_meta_ssl_roles_are_frozen_fail_closed(deviation) -> None:
    valid = _formal_options()
    assert valid.use_meta_ssl_cvs is True
    assert (
        valid.ssl_labeled_ratio,
        valid.ssl_unlabeled_ratio,
        valid.ssl_val_ratio,
    ) == (0.07, 0.63, 0.30)
    with pytest.raises(ValueError, match="Meta-SSL|ratio"):
        _formal_options(**deviation)

    ordinary = _formal_options(
        phase1_method="adv3b02",
        use_fcr=False,
        use_meta_ssl_cvs=False,
        ssl_labeled_ratio=0.2,
        ssl_unlabeled_ratio=0.5,
        ssl_val_ratio=0.3,
    )
    assert ordinary.use_fcr is False


def test_p1_2_formal_identity_route_uses_fcr_logits_and_mask_without_legacy_overwrite() -> None:
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
    ).eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 2, 64), return_aux=True)
    assert outputs["fcr_tx_logits"].shape == (2, 3)
    assert outputs["tx_logits"].shape == (2, 3)
    assert outputs["z_f_id"].shape[1] == 160
    assert outputs["feature_schema"] == FEATURE_SCHEMA

    disagree = {
        "tx_logits": torch.tensor([[9.0, 0.0], [9.0, 0.0]], requires_grad=True),
        "fcr_tx_logits": torch.tensor([[0.0, 9.0], [0.0, 9.0]], requires_grad=True),
        "z_id": torch.randn(2, 160),
        "z_f_id": torch.randn(2, 160),
        "feature_schema": FEATURE_SCHEMA,
    }
    route = getattr(train, "route_formal_identity_outputs", None)
    masked_ce = getattr(losses_module, "masked_identity_ce", None)
    assert callable(route) and callable(masked_ce)
    routed = route(disagree, use_fcr=True)
    assert routed["tx_logits"] is disagree["fcr_tx_logits"]
    assert routed["z_id"] is disagree["z_f_id"]
    assert disagree["tx_logits"][0, 0] == 9.0
    labels = torch.tensor([1, -1])
    mask = torch.tensor([True, False])
    loss = masked_ce(routed["tx_logits"], labels, mask, torch.nn.CrossEntropyLoss())
    assert float(loss.detach()) < 1e-3
    zero = masked_ce(routed["tx_logits"], labels, torch.zeros_like(mask), torch.nn.CrossEntropyLoss())
    assert float(zero.detach()) == 0.0

    ordinary = route(disagree, use_fcr=False)
    assert ordinary["tx_logits"] is disagree["tx_logits"]
    assert ordinary["z_id"] is disagree["z_id"]

    loader = DataLoader(_PredictionDataset(), batch_size=2, shuffle=False)
    formal_stats = eval_module.evaluate_loader(
        _DisagreeingModel(),
        loader,
        torch.device("cpu"),
        domain_label_map={},
    )
    assert formal_stats["tx_acc"] == 100.0

    ordinary_model = _DisagreeingModel()
    ordinary_model.use_fcr = False
    ordinary_stats = eval_module.evaluate_loader(
        ordinary_model,
        loader,
        torch.device("cpu"),
        domain_label_map={},
    )
    assert ordinary_stats["tx_acc"] == 0.0


def test_p1_3_all_formal_row_paths_are_contained_and_pairwise_disjoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    expected = {
        "--best_save_path": "best_joint.pth",
        "--latest_save_path": "latest.pth",
        "--best_test_save_path": "best_overall.pth",
        "--best_primary_save_path": "best_primary.pth",
        "--best_unseen_day_unseen_rx_save_path": "best_test_model.pth",
        "--best_unseen_day_seen_rx_save_path": "best_unseen_day_seen_rx.pth",
        "--best_seen_day_unseen_rx_save_path": "best_seen_day_unseen_rx.pth",
        "--best_worst_rx_save_path": "best_worst_rx.pth",
        "--ema_save_path": "ema.pth",
        "--swa_save_path": "swa.pth",
        "--swad_save_path": "swad.pth",
        "--log_dir": "logs",
        "--fcr_diagnostics_path": "fcr_diagnostics.json",
        "--fcr_predictions_path": "fcr_predictions.json",
    }
    for flag, suffix in expected.items():
        assert f'{flag} "${{row_root}}/{suffix}"' in text

    rendered = {
        row: {name: f"/immutable/run/{row}/{suffix}" for name, suffix in expected.items()}
        for row in (f"R{index}" for index in range(9))
    }
    all_paths = [path for paths in rendered.values() for path in paths.values()]
    assert len(all_paths) == len(set(all_paths))
    for row, paths in rendered.items():
        assert all(path.startswith(f"/immutable/run/{row}/") for path in paths.values())


class _PredictionDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (
            torch.full((2, 16), float(index)),
            1,
            0,
            {"physical_sample_id": f"sample:{index}"},
        )


class _DisagreeingModel(torch.nn.Module):
    use_fcr = True

    def forward(self, x, **_kwargs):
        batch = int(x.size(0))
        legacy = torch.tensor([[9.0, 0.0]], device=x.device).expand(batch, -1)
        formal = torch.tensor([[0.0, 9.0]], device=x.device).expand(batch, -1)
        return {
            "tx_logits": legacy,
            "fcr_tx_logits": formal,
            "z_f_id": torch.ones(batch, 160, device=x.device),
            "feature_schema": FEATURE_SCHEMA,
            "dom_logits": torch.zeros(batch, 1, device=x.device),
        }


def _valid_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "scenario": scenario,
            "predicted_class": 1,
            "feature_schema": FEATURE_SCHEMA,
            "row_id": "R8",
            "run_id": "formal_R8",
            "logit_route": "fcr_tx_logits",
        }
        for scenario in SCENARIOS
        for sample_id in ("sample:0", "sample:1")
    ]


def test_p1_4_prediction_export_is_four_scenario_complete_truth_blind_and_validated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validator = getattr(eval_module, "validate_fcr_prediction_records", None)
    exporter = getattr(eval_module, "export_fcr_predictions", None)
    assert callable(validator) and callable(exporter)
    valid = _valid_records()
    summary = validator(
        valid,
        expected_samples_per_scenario=2,
        run_id="formal_R8",
        row_id="R8",
    )
    assert summary["record_count"] == 8

    mutations = []
    mutations.append(valid[:-2])
    duplicate = copy.deepcopy(valid)
    duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
    mutations.append(duplicate)
    wrong_row = copy.deepcopy(valid)
    wrong_row[0]["row_id"] = "R7"
    mutations.append(wrong_row)
    wrong_schema = copy.deepcopy(valid)
    wrong_schema[0]["feature_schema"] = "legacy"
    mutations.append(wrong_schema)
    wrong_route = copy.deepcopy(valid)
    wrong_route[0]["logit_route"] = "tx_logits"
    mutations.append(wrong_route)
    for records in mutations:
        with pytest.raises(ValueError):
            validator(
                records,
                expected_samples_per_scenario=2,
                run_id="formal_R8",
                row_id="R8",
            )

    def identity_sat(x, scenario, args, **kwargs):
        del scenario, args, kwargs
        return x, None

    monkeypatch.setattr(eval_module, "apply_sat_channel_for_scenario", identity_sat)
    for scenario in SCENARIOS[1:]:
        stats = eval_module.evaluate_loader_sat_channel(
            _DisagreeingModel(),
            DataLoader(_PredictionDataset(), batch_size=2, shuffle=False),
            torch.device("cpu"),
            domain_label_map={},
            scenario=scenario,
            args=SimpleNamespace(sat_seed=2027),
        )
        assert stats["tx_acc"] == 100.0
    destination = tmp_path / "predictions.json"
    manifest = exporter(
        _DisagreeingModel(),
        DataLoader(_PredictionDataset(), batch_size=2, shuffle=False),
        torch.device("cpu"),
        args=SimpleNamespace(sat_seed=2027),
        output_path=destination,
        run_id="formal_R8",
        row_id="R8",
    )
    assert destination.is_file()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert manifest["record_count"] == payload["record_count"] == 8
    assert {record["predicted_class"] for record in payload["records"]} == {1}
    assert all(
        not ({"target", "target_label", "truth", "label", "y"} & set(record))
        for record in payload["records"]
    )

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "ARTIFACTS_COMPLETE" not in launcher
    assert "PREDICTIONS_READY" in launcher
    assert "truth" not in launcher.lower()
    assert "scorer" not in launcher.lower()
