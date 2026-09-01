from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
for path in (str(CODE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_source_ssl_defaults_use_single_v030_and_separate_source_holdout() -> None:
    from baselines.common.cvs_data import add_cvs_data_args

    parser = add_cvs_data_args(argparse.ArgumentParser())
    args = parser.parse_args([])

    assert args.wisig_labeled_ratio == 0.07
    assert args.wisig_unlabeled_ratio == 0.63
    assert args.wisig_source_val_ratio == 0.30
    assert args.wisig_source_holdout_rxs == ""


def test_concat_sat_ce_only_adds_only_weighted_tx_ce() -> None:
    from baselines.common.augmentation import concat_sat_ce_only_loss

    clean_loss = torch.tensor(2.0, requires_grad=True)
    sat_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 1])

    total, sat_ce = concat_sat_ce_only_loss(
        clean_loss,
        sat_logits,
        labels,
        weight=0.68,
    )

    expected_sat_ce = F.cross_entropy(sat_logits, labels)
    assert torch.allclose(sat_ce, expected_sat_ce)
    assert torch.allclose(total, clean_loss + 0.68 * expected_sat_ce)


def test_concat_sat_forward_is_one_joint_batch_and_splits_clean_from_satellite() -> None:
    from baselines.common.augmentation import forward_concat_sat_ce_only

    class SpyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shapes: list[tuple[int, ...]] = []

        def forward(self, iq: torch.Tensor, marker: int = 0) -> dict[str, torch.Tensor]:
            self.seen_shapes.append(tuple(iq.shape))
            values = iq[:, 0, 0].unsqueeze(1).repeat(1, 2) + marker
            return {"tx_logits": values, "features": values + 10}

    model = SpyModel()
    clean = torch.zeros(3, 2, 8)
    satellite = torch.ones(3, 2, 8)
    clean_out, sat_logits = forward_concat_sat_ce_only(
        model,
        clean,
        satellite,
        logits_key="tx_logits",
        model_kwargs={"marker": 2},
    )

    assert model.seen_shapes == [(6, 2, 8)]
    assert torch.equal(clean_out["tx_logits"], torch.full((3, 2), 2.0))
    assert torch.equal(sat_logits, torch.full((3, 2), 3.0))


def test_leo_weak_curriculum_changes_only_at_frozen_epoch_boundaries() -> None:
    from baselines.common.augmentation import resolve_sat_view_stage

    schedule = (
        "1@0.30:leo_clear_weak;"
        "41@0.60:leo_low_elev_weak,leo_rain_weak;"
        "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )
    assert resolve_sat_view_stage(schedule, 1) == (("leo_clear_weak",), 0.30)
    assert resolve_sat_view_stage(schedule, 40) == (("leo_clear_weak",), 0.30)
    assert resolve_sat_view_stage(schedule, 41) == (
        ("leo_low_elev_weak", "leo_rain_weak"),
        0.60,
    )
    assert resolve_sat_view_stage(schedule, 91) == (
        ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"),
        0.80,
    )


def test_fold18_matrix_freezes_new_split_and_target_blind_selection(tmp_path: Path) -> None:
    from cvsrffi.phase1_baseline_fold_matrix import build_rows

    rows = build_rows(
        run_id="phase1_riei_drift_newsplit_fold18_s392002_test",
        project_root=Path("/srv/CV-SincNet"),
        wisig_pkl=Path("/data/ManySig.pkl"),
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        python_bin="python3",
        gpu_ids=(4, 5, 6, 7),
    )

    assert [(row.method, row.fold, row.gpu) for row in rows] == [
        ("RIEI", 1, 4),
        ("RIEI", 8, 5),
        ("DRIFT", 1, 6),
        ("DRIFT", 8, 7),
    ]
    by_fold = {fold: [row for row in rows if row.fold == fold] for fold in (1, 8)}
    assert {tuple(row.train_receivers) for row in by_fold[1]} == {(3, 4, 6, 8)}
    assert {tuple(row.train_receivers) for row in by_fold[8]} == {(1, 3, 4, 6)}
    assert {tuple(row.target_receivers) for row in rows} == {(0, 2, 5, 7, 9, 10, 11)}

    for row in rows:
        command = list(row.command)
        joined = " ".join(command)
        assert "--wisig_labeled_ratio 0.07" in joined
        assert "--wisig_unlabeled_ratio 0.63" in joined
        assert "--wisig_source_val_ratio 0.3" in joined
        assert f"--wisig_source_holdout_rxs {row.fold}" in joined
        assert "--wisig_train_days 0,1,2" in joined
        assert "--wisig_test_days 3" in joined
        assert "--wisig_split_seed 392002" in joined
        assert "--seed 392002" in joined
        assert "--epochs 200" in joined
        assert "--use_concat_sat_channel_aug" in joined
        assert "--concat_sat_ce_only" in joined
        assert "--concat_sat_start_epoch 80" in joined
        assert "--lambda_sat_cls 0.68" in joined
        assert "--lambda_sat_cons 0" in joined
        assert (
            "--sat_view_schedule "
            "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;"
            "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
        ) in joined
        assert "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in joined
        assert "--eval_sat_on test_seen_day_unseen_rx,test_unseen_day_unseen_rx" in joined
        assert "--no_test_on_val_improve" in command
        assert "--test_eval_interval 0" in joined
        assert "--paper_eval_last_n 0" in joined
        assert "--final_test_best_by_val" in command
        assert "--final_test_target_only" in command
        assert "V_select" not in joined


class _OneRowDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "iq": torch.tensor([1.0]),
            "label": torch.tensor(0),
            "receiver": torch.tensor(0),
        }


class _ScalarClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        score = x.reshape(x.size(0), -1)[:, 0] * self.weight
        return {"tx_logits": torch.stack([score, -score], dim=1)}


def test_final_target_evaluation_reloads_v_selected_checkpoint(tmp_path: Path) -> None:
    from baselines.common.cvs_trainer import run_validation_gated_training

    model = _ScalarClassifier()
    loader = DataLoader(_OneRowDataset(), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def train_step(model, batch, device, epoch, step):
        with torch.no_grad():
            model.weight.fill_(1.0 if epoch == 1 else -1.0)
        return {"loss": 0.0}

    run_validation_gated_training(
        model=model,
        train_loader=loader,
        val_loader=loader,
        named_test_loaders={
            "test_unseen_day_seen_rx": loader,
            "test_seen_day_unseen_rx": loader,
            "test_unseen_day_unseen_rx": loader,
        },
        device="cpu",
        epochs=2,
        optimizer=optimizer,
        train_step_fn=train_step,
        forward_eval_fn=lambda model, batch, device: model(batch["iq"].to(device)),
        test_on_val_improve=False,
        reload_best_for_final_test=True,
        output_dir=str(tmp_path),
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert model.weight.item() == 1.0
    assert metrics["final"]["checkpoint_source"] == "best_by_val"
    assert metrics["final"]["checkpoint_epoch"] == 1
    assert metrics["final"]["test_overall"]["tx_acc"] == 100.0
