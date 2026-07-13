import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import unlabeled_known_acceptance_quarantine_loss  # noqa: E402
from SSDG.train_ssdg import _route_unlabeled_known_geometry, build_arg_parser  # noqa: E402


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


def test_strict_local_quarantine_refuses_global_class_fallback_for_sparse_domains():
    anchors = torch.tensor(
        [[1.0, 0.0], [0.99, 0.02], [0.0, 1.0], [0.02, 0.99]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1])
    sparse_domains = torch.tensor([0, 1, 0, 1])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    _loss, metrics = unlabeled_known_acceptance_quarantine_loss(
        anchors,
        labels,
        query,
        anchor_d=sparse_domains,
        query_y=torch.tensor([0, 1]),
        min_samples_per_class=2,
        require_domain_local_components=True,
    )

    assert metrics["active"] == 0.0
    assert metrics["local_component_count"] == 0.0
    assert metrics["global_component_fallback"] == 0.0
    assert metrics["domain_local_components_required"] == 1.0


def test_unlabeled_router_uses_separate_clean_sat_local_components_without_fallback():
    args = build_arg_parser().parse_args(["--output_dir", "runs/tmp"])
    args.u_geometry_all_valid_queries = True
    args.u_quarantine_valid_domain_only = True
    args.u_quarantine_min_count = 1
    args.u_quarantine_include_sat_view = True
    args.direct_metric_require_domain_local_components = True
    args.direct_metric_min_samples_per_component = 2
    clean = torch.tensor(
        [
            [1.0, 0.00], [0.99, 0.03], [0.94, 0.34], [0.93, 0.36],
            [0.0, 1.00], [0.03, 0.99], [0.34, 0.94], [0.36, 0.93],
        ],
        dtype=torch.float32,
    )
    sat = torch.tensor(
        [
            [0.98, 0.18], [0.97, 0.20], [0.86, 0.51], [0.84, 0.54],
            [0.18, 0.98], [0.20, 0.97], [0.51, 0.86], [0.54, 0.84],
        ],
        dtype=torch.float32,
    )
    y_clean = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    d_clean = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    pseudo = torch.tensor([0, 1])
    d_u = torch.tensor([0, 0])

    _loss, info, _core, _direct = _route_unlabeled_known_geometry(
        args=args,
        z_id_l=torch.cat([clean, sat], dim=0),
        y_l=torch.cat([y_clean, y_clean], dim=0),
        d_l=torch.cat([d_clean, d_clean], dim=0),
        out_s={"z_id": torch.tensor([[1.0, 0.01], [0.01, 1.0]])},
        out_u_sat={"z_id": torch.tensor([[0.98, 0.18], [0.18, 0.98]])},
        pseudo=pseudo,
        d_u=d_u,
        pseudo_mask=torch.ones(2, dtype=torch.bool),
        valid_u_mask=torch.ones(2, dtype=torch.bool),
        labeled_view_count=8,
        labeled_sat_applied=True,
    )

    assert info["active"] == 1.0
    assert info["multiview_local_components"] == 1.0
    assert info["tri_direct_count"] >= info["tri_trusted_core_count"]
    assert info["global_component_fallback"] == 0.0
    assert info["clean_local_component_count"] == 4.0
    assert info["sat_local_component_count"] == 4.0
