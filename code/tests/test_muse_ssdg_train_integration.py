import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from torch import nn

from SSDG import train_ssdg
from cvsrffi.muse_ssdg import MUSEConfig, muse_schedule_for_epoch


class _TinyMUSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 4
        self.num_classes = 3
        self.num_domains = 2
        self.weight = nn.Parameter(torch.ones(1))


def _args(level="M1"):
    return train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true", "--muse_level", level]
    )


def _outputs():
    weak_z_id = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        requires_grad=True,
    )
    strong_z_id = torch.tensor(
        [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]],
        requires_grad=True,
    )
    weak = {
        "tx_logits": torch.tensor([[5.0, 0.0, -1.0], [0.0, 5.0, -1.0]]),
        "z_id": weak_z_id,
        "z_dom": torch.randn(2, 4, requires_grad=True),
        "dom_logits": torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True),
        "adv_dom_logits": torch.tensor([[1.5, -0.5], [-0.5, 1.5]], requires_grad=True),
    }
    strong = {
        "tx_logits": torch.tensor(
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True
        ),
        "z_id": strong_z_id,
        "z_dom": torch.randn(2, 4, requires_grad=True),
        "dom_logits": torch.tensor([[1.0, -0.5], [-0.5, 1.0]], requires_grad=True),
        "adv_dom_logits": torch.tensor([[0.8, -0.2], [-0.2, 0.8]], requires_grad=True),
    }
    return weak, strong


def _metadata():
    return {
        "domains": torch.tensor([0, 1]),
        "receivers": torch.tensor([0, 1]),
        "memory_keys": [
            (0, 0, 0, 10, 100),
            (1, 0, 0, 11, 101),
        ],
    }


def test_muse_unlabeled_path_never_passes_y_u_to_identity_losses():
    text = Path("code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    muse_block = text[text.index("def _compute_muse_unlabeled_losses") :]
    signature = muse_block.split("\n", 1)[0]
    assert "y_u" not in signature
    assert "proxy_unknown" not in muse_block.split("def ", 2)[0]


def test_muse_training_uses_true_u_s_labels_only_in_no_grad_diagnostics():
    text = Path("code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    start = text.index(
        '                if muse_state is not None:\n'
        '                    if muse_unlabeled_batch is None:'
    )
    end = text.index(
        '                elif phase == "pseudo" and bool(args.use_unlabeled):',
        start,
    )
    muse_train_block = text[start:end]
    assert "unlabeled_count = int(x_u.size(0))" in muse_train_block
    assert "int(y_u.numel())" not in muse_train_block
    assert muse_train_block.count("y_u") == 2

    tree = ast.parse(text)
    diagnostic = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_muse_unlabeled_label_diagnostics"
    )
    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "no_grad"
        for decorator in diagnostic.decorator_list
    )


def test_muse_parser_defaults_to_final_only_and_joint_epoch():
    args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true"]
    )
    assert args.checkpoint_selection == "final_only"
    assert args.muse_epoch_basis == "unlabeled_loader"


def test_muse_parser_exposes_schedule_routing_and_loss_hyperparameters():
    args = _args("M3")
    for field in fields(MUSEConfig):
        assert hasattr(args, f"muse_{field.name}")
    for name in (
        "muse_high_threshold",
        "muse_low_threshold",
        "muse_candidate_mass",
        "muse_candidate_max_classes",
        "muse_fusion_global_weight",
        "muse_fusion_local_weight",
        "muse_fusion_student_weight",
        "muse_unlabeled_prototype_weight",
        "muse_temporal_stability_steps",
        "muse_lambda_domain",
        "muse_lambda_adv",
        "muse_lambda_self",
        "muse_lambda_nuisance",
        "muse_lambda_satellite",
        "muse_lambda_cross_receiver",
    ):
        assert hasattr(args, name)


def test_muse_epoch_pairs_use_every_unlabeled_batch_and_cycle_labeled_batches():
    pairs = list(
        train_ssdg._muse_epoch_pairs(
            ["l0", "l1"],
            ["u0", "u1", "u2", "u3", "u4"],
            use_muse=True,
        )
    )
    assert pairs == [
        ("l0", "u0"),
        ("l1", "u1"),
        ("l0", "u2"),
        ("l1", "u3"),
        ("l0", "u4"),
    ]


def test_muse_levels_enable_capabilities_monotonically_and_m0_is_legacy():
    assert train_ssdg._muse_level_capabilities("M0") == {
        "base": False,
        "fusion": False,
        "satellite": False,
    }
    assert train_ssdg._muse_level_capabilities("M1") == {
        "base": True,
        "fusion": False,
        "satellite": False,
    }
    assert train_ssdg._muse_level_capabilities("M2") == {
        "base": True,
        "fusion": True,
        "satellite": False,
    }
    assert train_ssdg._muse_level_capabilities("M3") == {
        "base": True,
        "fusion": True,
        "satellite": True,
    }


def test_muse_initialization_adds_training_heads_to_optimizer_parameters():
    model = _TinyMUSEModel()
    state = train_ssdg._initialize_muse_training_state(
        _args("M1"), model, torch.device("cpu")
    )
    parameters = train_ssdg._optimizer_parameters(model, state)
    parameter_ids = {id(parameter) for parameter in parameters}
    assert state is not None
    assert all(id(parameter) in parameter_ids for parameter in model.parameters())
    assert all(id(parameter) in parameter_ids for parameter in state["heads"].parameters())


@pytest.mark.parametrize("level", ["M1", "M2", "M3"])
def test_muse_s1_reads_unlabeled_without_identity_classification_gradient(level):
    state = train_ssdg._initialize_muse_training_state(
        _args(level), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(1, state["config"])
    weak, strong = _outputs()
    for _ in range(3):
        losses = train_ssdg._compute_muse_unlabeled_losses(
            x_u=torch.randn(2, 2, 8),
            metadata=_metadata(),
            teacher_outputs=weak,
            student_outputs={"weak": weak, "strong": strong, "satellite": None},
            muse_state=state,
            simulator_metadata={
                "nuisance": torch.zeros(2, 6),
                "nuisance_valid": torch.ones(2, dtype=torch.bool),
            },
        )
    assert losses["stage"] == "S1"
    assert losses["identity"].item() == 0.0
    assert losses["base"].requires_grad
    losses["total"].backward()
    assert strong["tx_logits"].grad is None or torch.equal(
        strong["tx_logits"].grad, torch.zeros_like(strong["tx_logits"])
    )
    assert state["classification_prototypes"].state_dict()["counts"] == {}


def test_muse_m2_routes_are_a_partition_and_m3_updates_only_classification_bank():
    state = train_ssdg._initialize_muse_training_state(
        _args("M3"), _TinyMUSEModel(), torch.device("cpu")
    )
    weak, strong = _outputs()
    losses = None
    for epoch in (69, 70, 71):
        state["schedule_state"] = muse_schedule_for_epoch(epoch, state["config"])
        losses = train_ssdg._compute_muse_unlabeled_losses(
            x_u=torch.randn(2, 2, 8),
            metadata=_metadata(),
            teacher_outputs=weak,
            student_outputs={"weak": weak, "strong": strong, "satellite": strong},
            muse_state=state,
            simulator_metadata={},
        )
    assert losses is not None
    route = losses["route"]
    assert torch.equal(
        torch.stack([route.high, route.mid, route.low]).int().sum(0),
        torch.ones(2, dtype=torch.int),
    )
    assert state["classification_prototypes"].state_dict()["counts"]


def test_muse_m2_low_route_uses_unlabeled_entropy_not_true_identity_targets():
    args = _args("M2")
    args.muse_high_threshold = 1.0
    args.muse_low_threshold = 0.99
    args.muse_lambda_low_entropy = 0.0
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    weak, strong = _outputs()
    without_low = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        _metadata(),
        weak,
        {"weak": weak, "strong": strong, "satellite": None},
        state,
        {},
    )
    args.muse_lambda_low_entropy = 1.0
    with_low = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        _metadata(),
        weak,
        {"weak": weak, "strong": strong, "satellite": None},
        state,
        {},
    )
    assert without_low["route"].low.tolist() == [True, True]
    assert without_low["identity"].item() == 0.0
    assert with_low["identity"].item() < 0.0


@pytest.mark.parametrize("level", ["M1", "M2", "M3"])
def test_muse_s3c_freezes_temporal_and_classification_statistics(level):
    state = train_ssdg._initialize_muse_training_state(
        _args(level), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(181, state["config"])
    weak, strong = _outputs()
    train_ssdg._compute_muse_unlabeled_losses(
        x_u=torch.randn(2, 2, 8),
        metadata=_metadata(),
        teacher_outputs=weak,
        student_outputs={"weak": weak, "strong": strong, "satellite": strong},
        muse_state=state,
        simulator_metadata={},
    )
    assert state["temporal_memory"].state_dict()["frozen"] is True
    assert state["classification_prototypes"].state_dict()["frozen"] is True


def test_muse_rejects_any_u_s_open_geometry_update():
    with pytest.raises(
        RuntimeError, match="MUSE_PROTOCOL_U_S_OPEN_GEOMETRY_FORBIDDEN"
    ):
        train_ssdg._assert_muse_open_geometry_role("U_s")
    train_ssdg._assert_muse_open_geometry_role("L_s")
    with pytest.raises(
        RuntimeError, match="MUSE_PROTOCOL_U_S_OPEN_GEOMETRY_FORBIDDEN"
    ):
        train_ssdg._route_unlabeled_known_geometry(
            args=None,
            z_id_l=None,
            y_l=None,
            d_l=None,
            out_s=None,
            out_u_sat=None,
            pseudo=None,
            d_u=None,
            pseudo_mask=None,
            valid_u_mask=None,
            dataset_role="U_s",
        )


def test_muse_checkpoint_state_has_all_training_only_fields():
    state = train_ssdg._initialize_muse_training_state(
        _args("M2"), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(41, state["config"])
    checkpoint = train_ssdg._muse_checkpoint_state(state)
    assert set(checkpoint) == {
        "muse_training_heads",
        "muse_temporal_memory",
        "muse_classification_prototypes",
        "muse_schedule_state",
    }
    assert checkpoint["muse_training_heads"]
    assert checkpoint["muse_schedule_state"]["stage"] == "S2B"


def test_train_reaches_all_muse_integration_boundaries():
    source = Path("code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    called_names = {
        node.func.id
        for node in ast.walk(train_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "_initialize_muse_training_state",
        "_optimizer_parameters",
        "_muse_epoch_pairs",
        "_compute_muse_unlabeled_losses",
        "_muse_checkpoint_state",
    } <= called_names


def test_muse_unlabeled_total_is_not_rescaled_by_legacy_pseudo_weights():
    base = torch.tensor(2.0)
    muse_total = torch.tensor(3.0)
    legacy = {
        "identity": torch.tensor(5.0),
        "entropy": torch.tensor(7.0),
        "domain": torch.tensor(11.0),
        "adv": torch.tensor(13.0),
        "satellite": torch.tensor(17.0),
    }
    args = _args("M1")
    args.lambda_u = 0.0
    args.lambda_ent = 100.0
    args.lambda_u_domain = 100.0
    args.lambda_u_adv = 100.0
    args.lambda_u_sat_cons = 100.0
    combined = train_ssdg._compose_unlabeled_closed_loss(
        base,
        args=args,
        muse_state={"enabled": True},
        muse_total=muse_total,
        **legacy,
    )
    assert combined.item() == 5.0
