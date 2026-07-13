from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.balanced_tx_rx_sampler import BalancedTxDomainBatchSampler  # noqa: E402
from cvsrffi.losses import PrototypeMemoryBank  # noqa: E402
from post_stage_cli import add_sat_eval_args  # noqa: E402
from SSDG.train_ssdg import (  # noqa: E402
    _backward_with_open_set_projection,
    _select_unlabeled_geometry_masks,
)


@dataclass(frozen=True)
class _Item:
    tx_i: int
    rx_i: int
    day_i: int


class _GridDataset:
    def __init__(self, tx_count: int = 6, domain_count: int = 6, per_cell: int = 4):
        self.index = [
            _Item(tx_i=tx, rx_i=domain, day_i=0)
            for tx in range(tx_count)
            for domain in range(domain_count)
            for _ in range(per_cell)
        ]

    def __len__(self):
        return len(self.index)


def test_balanced_sampler_produces_complete_tx_domain_grid_and_changes_by_epoch():
    sampler = BalancedTxDomainBatchSampler(
        _GridDataset(),
        tx_per_batch=6,
        domain_per_batch=6,
        samples_per_tx_domain=3,
        seed=17,
        drop_last=True,
    )
    sampler.set_epoch(1)
    first = next(iter(sampler))
    sampler.set_epoch(2)
    second = next(iter(sampler))
    assert len(first) == 108
    assert len(second) == 108
    assert first != second
    stats = sampler.batch_geometry_stats(first)
    assert stats["tx_per_batch"] == 6.0
    assert stats["domain_per_batch"] == 6.0
    assert stats["tx_rx_rectangles"] > 0.0


def test_prototype_domain_alignment_is_differentiable_on_current_features():
    bank = PrototypeMemoryBank(
        2,
        2,
        momentum=0.9,
        margin=0.1,
        domain_align_weight=1.0,
        push_weight=0.0,
        min_count=2,
    )
    init_z = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    )
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    d = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    bank.update(init_z, y, d)
    z = init_z.clone().requires_grad_(True)
    loss, info = bank.loss(z, y, d)
    loss.backward()
    assert info["proto_domain_align"] > 0.0
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.norm().item()) > 0.0


def test_default_satellite_evaluation_protocol_is_leo_weak():
    parser = argparse.ArgumentParser()
    add_sat_eval_args(parser)
    args = parser.parse_args([])
    assert args.eval_sat_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"


def test_unlabeled_geometry_masks_do_not_reapply_pseudo_confidence_to_direct_or_invariance():
    pseudo = torch.tensor([True, False, True, False, False])
    core = torch.tensor([True, True, False, False, True])
    valid = torch.tensor([True, True, True, False, True])

    ce, direct, invariant = _select_unlabeled_geometry_masks(
        pseudo,
        core,
        valid,
        all_valid_queries=True,
        direct_valid_domain_only=True,
    )

    assert ce.tolist() == [True, False, False, False, False]
    assert direct.tolist() == [True, True, False, False, True]
    assert invariant.tolist() == [True, True, True, False, True]


def test_objective_gradient_budget_amplifies_small_source_group_and_protects_closed_gradient():
    model = torch.nn.Linear(2, 1, bias=False)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    weight = model.weight.reshape(-1)
    closed = weight.sum()
    boundary = -8.0 * weight[0] + weight[1]
    source = -0.2 * weight[0]
    invariant = 0.5 * weight[1]
    u_geometry = 0.1 * weight.sum()
    open_loss = boundary + source + invariant + u_geometry

    info = _backward_with_open_set_projection(
        model,
        scaler,
        closed,
        open_loss,
        project_conflicts=True,
        budget_controller=False,
        protect_closed_on_conflict=True,
        open_loss_groups={
            "boundary": boundary,
            "source": source,
            "invariant": invariant,
            "u_geometry": u_geometry,
        },
        open_group_shares={"boundary": 0.35, "source": 0.30, "invariant": 0.20, "u_geometry": 0.15},
        open_group_min_scale=0.1,
        open_group_max_scale=32.0,
    )

    assert info["nonfinite_gradient_bundle"] == 0.0
    assert info["objective_source_raw_norm"] > 0.0
    assert info["objective_source_scale"] > info["objective_boundary_scale"]
    assert info["objective_source_effective_norm"] > info["objective_source_raw_norm"]
    assert info["conflict_projection_priority_code"] == 1.0
    assert info["budget_scope_shared_trainable_params"] == 1.0
    assert info["budget_scope_shared_zid_path"] == 0.0
