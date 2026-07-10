import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import unlabeled_known_acceptance_quarantine_loss  # noqa: E402
from SSDG.train_ssdg import build_arg_parser  # noqa: E402


def test_unlabeled_quarantine_loss_penalizes_known_acceptance():
    anchors = torch.tensor(
        [
            [1.00, 0.00, 0.00],
            [0.98, 0.06, 0.00],
            [0.98, -0.06, 0.00],
            [0.00, 1.00, 0.00],
            [0.06, 0.98, 0.00],
            [-0.06, 0.98, 0.00],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    near_queries = torch.tensor([[0.99, 0.02, 0.00], [0.02, 0.99, 0.00]], dtype=torch.float32)
    far_queries = torch.tensor([[0.00, 0.00, 1.00], [0.00, 0.00, -1.00]], dtype=torch.float32)

    near_loss, near_metrics = unlabeled_known_acceptance_quarantine_loss(
        anchors,
        labels,
        near_queries,
        query_y=torch.tensor([0, 1]),
        accept_target=0.10,
        accept_quantile=0.90,
        cvar_alpha=1.0,
        min_samples_per_class=2,
    )
    far_loss, far_metrics = unlabeled_known_acceptance_quarantine_loss(
        anchors,
        labels,
        far_queries,
        query_y=torch.tensor([0, 1]),
        accept_target=0.10,
        accept_quantile=0.90,
        cvar_alpha=1.0,
        min_samples_per_class=2,
    )

    assert near_metrics["active"] == 1.0
    assert near_metrics["query_count"] == 2.0
    assert near_metrics["accept_rate"] > far_metrics["accept_rate"]
    assert near_loss.item() > far_loss.item()
    assert far_loss.item() == 0.0
    assert far_metrics["outside_known_negative_disabled"] == 1.0
    assert near_metrics["nearest_angle_p95_deg"] < far_metrics["nearest_angle_p95_deg"]


def test_unlabeled_quarantine_reports_geometry_tri_state_counts():
    anchors = torch.tensor(
        [
            [1.00, 0.00, 0.00],
            [0.98, 0.06, 0.00],
            [0.98, -0.06, 0.00],
            [0.00, 1.00, 0.00],
            [0.06, 0.98, 0.00],
            [-0.06, 0.98, 0.00],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    queries = torch.tensor(
        [
            [0.99, 0.02, 0.00],
            [0.02, 0.99, 0.00],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, -1.00],
        ],
        dtype=torch.float32,
    )

    _loss, metrics = unlabeled_known_acceptance_quarantine_loss(
        anchors,
        labels,
        queries,
        query_y=torch.tensor([0, 1, 0, 1]),
        accept_target=0.10,
        accept_quantile=0.90,
        cvar_alpha=1.0,
        min_samples_per_class=2,
    )

    trusted = metrics["tri_trusted_core_count"]
    ambiguous = metrics["tri_ambiguous_tail_count"]
    outside = metrics["tri_outside_reject_count"]
    assert trusted >= 1.0
    assert outside >= 1.0
    assert trusted + ambiguous + outside == metrics["query_count"]
    assert metrics["tri_trusted_core_rate"] == trusted / metrics["query_count"]
    assert metrics["tri_outside_reject_rate"] == outside / metrics["query_count"]


def test_train_parser_exposes_unlabeled_quarantine_args():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            "runs/tmp",
            "--lambda_u_quarantine_accept",
            "0.007",
            "--u_quarantine_start_epoch",
            "95",
            "--u_quarantine_accept_target",
            "0.18",
            "--u_quarantine_valid_domain_only",
            "true",
        ]
    )

    assert args.lambda_u_quarantine_accept == 0.007
    assert args.u_quarantine_start_epoch == 95
    assert args.u_quarantine_accept_target == 0.18
    assert args.u_quarantine_valid_domain_only is True


def test_receiver_local_tristate_never_marks_interclass_bridge_as_trusted_core():
    anchors = torch.tensor(
        [
            [1.00, 0.00], [0.99, 0.03], [0.97, 0.20], [0.96, 0.23],
            [0.00, 1.00], [0.03, 0.99], [0.20, 0.97], [0.23, 0.96],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    bridge = torch.tensor([[1.0, 1.0]], dtype=torch.float32, requires_grad=True)

    loss, metrics = unlabeled_known_acceptance_quarantine_loss(
        anchors,
        labels,
        bridge,
        anchor_d=domains,
        query_y=torch.tensor([0]),
        min_samples_per_class=2,
        component_margin_rad=0.10,
        accept_target=0.05,
    )
    loss.backward()

    assert metrics["local_component_count"] == 4.0
    assert metrics["tri_trusted_core_count"] == 0.0
    assert metrics["tri_ambiguous_tail_count"] + metrics["tri_outside_reject_count"] == 1.0
    assert bridge.grad is not None


def test_tristate_rejects_pseudo_label_that_disagrees_with_nearest_component():
    anchors = torch.tensor(
        [[1.0, 0.0], [0.99, 0.02], [0.0, 1.0], [0.02, 0.99]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    _loss, metrics = unlabeled_known_acceptance_quarantine_loss(
        anchors,
        labels,
        query,
        query_y=torch.tensor([0]),
        min_samples_per_class=2,
    )

    assert metrics["tri_trusted_core_count"] == 0.0
    assert metrics["tri_outside_reject_count"] == 1.0
    assert metrics["tri_pseudo_component_agreement_rate"] == 0.0
