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


def test_muse_training_step_has_no_u_s_truth_or_in_loop_label_diagnostics():
    text = Path("code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    start = text.index(
        '                if muse_state is not None:\n'
        '                    if muse_unlabeled_batch is None:'
    )
    end = text.index(
        "                elif legacy_unlabeled_active:",
        start,
    )
    muse_train_block = text[start:end]
    assert "unlabeled_count = int(x_u.size(0))" in muse_train_block
    assert "int(y_u.numel())" not in muse_train_block
    assert "y_u" not in muse_train_block
    assert "_muse_unlabeled_label_diagnostics" not in text


def test_muse_unlabeled_dataset_view_removes_tx_truth_before_collation():
    class _SourceDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return (
                torch.ones(2, 8),
                17,
                3,
                {
                    "tx_i": 17,
                    "tx": "secret-tx",
                    "true_tx_i": 17,
                    "rx_i": 3,
                    "day_i": 1,
                    "eq_i": 0,
                    "sig_i": 9,
                    "base_index": 99,
                },
            )

    sample = train_ssdg._MUSEUnlabeledDatasetView(_SourceDataset())[0]
    assert len(sample) == 3
    x_u, domain_u, metadata_u = sample
    assert x_u.shape == (2, 8)
    assert domain_u == 3
    assert {"tx_i", "tx", "true_tx_i"}.isdisjoint(metadata_u)
    moved_x, moved_extra = train_ssdg._move_muse_unlabeled_batch(
        (
            x_u.unsqueeze(0),
            torch.tensor([domain_u]),
            {key: torch.tensor([value]) for key, value in metadata_u.items()},
        ),
        torch.device("cpu"),
    )
    assert moved_x.shape == (1, 2, 8)
    assert len(moved_extra) == 2


def test_muse_parser_defaults_to_final_only_and_joint_epoch():
    args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true"]
    )
    assert args.checkpoint_selection == "final_only"
    assert args.muse_epoch_basis == "unlabeled_loader"


def test_optional_component_gradient_accepts_graphless_zero():
    parameter = nn.Parameter(torch.tensor(2.0))

    graphless = train_ssdg._autograd_grad_or_none(
        torch.tensor(0.0), [parameter], retain_graph=True
    )
    connected = train_ssdg._autograd_grad_or_none(
        parameter.square(), [parameter], retain_graph=False
    )

    assert graphless == (None,)
    assert torch.allclose(connected[0], torch.tensor(4.0))


def test_muse_can_delegate_final_target_eval_without_changing_legacy(monkeypatch):
    calls = []

    def fake_internal(*_args, **_kwargs):
        calls.append("internal")
        return {"status": "COMPLETE"}

    monkeypatch.setattr(
        train_ssdg, "_evaluate_frozen_phase1_checkpoint", fake_internal
    )
    muse_args = _args("M3")
    muse_args.muse_external_final_eval = True
    delegated = train_ssdg._run_final_heldout_evaluation(
        muse_args, object(), {}, torch.device("cpu"), "final_ssdg.pth"
    )
    assert delegated["status"] == "DELEGATED_TO_MUSE_LAUNCHER"
    assert calls == []

    legacy_args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "false"]
    )
    completed = train_ssdg._run_final_heldout_evaluation(
        legacy_args, object(), {}, torch.device("cpu"), "final_ssdg.pth"
    )
    assert completed["status"] == "COMPLETE"
    assert calls == ["internal"]


def test_muse_external_eval_terminal_ignores_optional_legacy_promotion_gates():
    status = train_ssdg._resolve_phase1_terminal_status(
        tail_stopped=False,
        export_failed=False,
        final_blocked=False,
        selected_checkpoint_exists=True,
        heldout_eval_status="DELEGATED_TO_MUSE_LAUNCHER",
        external_final_eval=True,
        p0_mechanisms_ready=False,
        p1_mechanisms_ready=False,
        endpoint_export_ready=False,
        mechanism_gates_required=False,
        endpoint_export_required=False,
    )
    assert status == "COMPLETE"


def test_muse_enablement_resolves_and_validates_exact_four_role_source_protocol():
    args = _args("M1")
    train_ssdg._enforce_muse_source_protocol(args)
    assert args.split_mode == "tx_rx_day_1_7_2"
    assert args.phase1_source_role_protocol == "l_s_u_s_v_cal_v_select"
    assert (
        args.labeled_ratio,
        args.unlabeled_ratio,
        args.source_cal_ratio,
        args.source_select_ratio,
        args.source_val_ratio,
    ) == (0.07, 0.63, 0.15, 0.15, 0.30)

    mismatched = _args("M1")
    mismatched.source_cal_ratio = 0.10
    with pytest.raises(ValueError, match="MUSE source protocol mismatch"):
        train_ssdg._enforce_muse_source_protocol(mismatched)


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
        "muse_fusion_prototype_weight",
        "muse_prior_alignment_gamma",
        "muse_unlabeled_prototype_weight",
        "muse_temporal_stability_steps",
        "muse_lambda_domain",
        "muse_lambda_adv",
        "muse_lambda_self",
        "muse_lambda_nuisance",
        "muse_lambda_satellite",
        "muse_lambda_cross_receiver",
        "muse_enable_u_prototype_update",
        "muse_enable_u_satellite_identity",
        "muse_use_prototype_evidence",
        "muse_require_temporal_stability",
        "muse_class_balanced_cap",
        "muse_nuisance_detached",
    ):
        assert hasattr(args, name)


def test_u_prototype_update_override_can_enable_m2_and_disable_m3():
    m2 = _args("M2")
    m3 = _args("M3")
    assert not train_ssdg._muse_u_prototype_update_enabled(
        m2, train_ssdg._muse_level_capabilities("M2")
    )
    assert train_ssdg._muse_u_prototype_update_enabled(
        m3, train_ssdg._muse_level_capabilities("M3")
    )
    m2.muse_enable_u_prototype_update = True
    m3.muse_enable_u_prototype_update = False
    assert train_ssdg._muse_u_prototype_update_enabled(
        m2, train_ssdg._muse_level_capabilities("M2")
    )
    assert not train_ssdg._muse_u_prototype_update_enabled(
        m3, train_ssdg._muse_level_capabilities("M3")
    )


def test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment():
    args = _args("M2")
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    weak, strong = _outputs()
    train_ssdg._compute_muse_labeled_auxiliary_loss(
        weak["z_id"],
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        state,
    )

    first = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": 69},
        weak,
        {"weak": weak, "strong": strong, "satellite": None},
        state,
        {},
    )
    changed_student = {**strong, "tx_logits": -strong["tx_logits"]}
    second = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": 69},
        weak,
        {"weak": weak, "strong": changed_student, "satellite": None},
        state,
        {},
    )

    evidence = first["evidence_probabilities"]
    assert tuple(evidence) == ("global", "local", "prototype")
    assert torch.allclose(
        evidence["prototype"],
        state["classification_prototypes"].class_probabilities(
            weak["z_id"], num_classes=3
        ),
    )
    assert torch.allclose(first["fused_probability"], second["fused_probability"])
    assert not torch.allclose(
        evidence["prototype"],
        torch.softmax(strong["tx_logits"].detach().float(), dim=-1),
    )

    before_prior_change = first["fused_probability"].clone()
    state["source_global_class_counts"].copy_(torch.tensor([90.0, 5.0, 5.0]))
    state["source_domain_class_counts"][0].copy_(torch.tensor([5.0, 90.0, 5.0]))
    after = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": 69},
        weak,
        {"weak": weak, "strong": strong, "satellite": None},
        state,
        {},
    )
    assert not torch.allclose(before_prior_change[0], after["fused_probability"][0])


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


def test_m0_and_m1_share_unlabeled_length_optimizer_budget_without_m0_consuming_u_s():
    class _BudgetOnlyLoader:
        def __len__(self):
            return 5

        def __iter__(self):
            raise AssertionError("M0 must not fetch or consume U_s batches")

    m0_pairs = list(
        train_ssdg._muse_epoch_pairs(
            ["l0", "l1"],
            _BudgetOnlyLoader(),
            use_muse=False,
            use_unlabeled_step_budget=True,
        )
    )
    m1_pairs = list(
        train_ssdg._muse_epoch_pairs(
            ["l0", "l1"],
            ["u0", "u1", "u2", "u3", "u4"],
            use_muse=True,
            use_unlabeled_step_budget=True,
        )
    )

    assert len(m0_pairs) == len(m1_pairs) == 5
    assert m0_pairs == [
        ("l0", None),
        ("l1", None),
        ("l0", None),
        ("l1", None),
        ("l0", None),
    ]
    assert all(unlabeled_batch is None for _, unlabeled_batch in m0_pairs)
    assert train_ssdg._initialize_muse_training_state(
        _args("M0"), _TinyMUSEModel(), torch.device("cpu")
    ) is None


def test_m0_disables_legacy_u_s_loss_path_while_plain_legacy_keeps_it():
    m0_args = _args("M0")
    legacy_args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "false"]
    )

    assert not train_ssdg._legacy_unlabeled_active(
        m0_args,
        muse_state=None,
        phase="pseudo",
    )
    assert train_ssdg._legacy_unlabeled_active(
        legacy_args,
        muse_state=None,
        phase="pseudo",
    )
    assert not train_ssdg._legacy_unlabeled_active(
        m0_args,
        muse_state=None,
        phase="label",
    )


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


def test_muse_labeled_auxiliary_trains_local_head_and_updates_legal_prototypes():
    state = train_ssdg._initialize_muse_training_state(
        _args("M2"), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(17, state["config"])
    z_id_l = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        requires_grad=True,
    )
    loss = train_ssdg._compute_muse_labeled_auxiliary_loss(
        z_id_l,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        state,
    )
    loss.backward()
    local_parameters = [
        parameter
        for name, parameter in state["heads"].named_parameters()
        if name.startswith(("shared_projection", "shared_classifier", "domain_delta"))
    ]
    assert all(parameter.grad is not None for parameter in local_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in local_parameters)
    assert any(torch.count_nonzero(parameter.grad).item() > 0 for parameter in local_parameters)
    assert state["classification_prototypes"].state_dict()["counts"] == {0: 1.0, 1: 1.0}


def test_muse_nuisance_loss_uses_the_paired_simulated_view_z_dom():
    state = train_ssdg._initialize_muse_training_state(
        _args("M1"), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(1, state["config"])
    weak, strong = _outputs()
    nuisance_z_dom = torch.randn(2, 4, requires_grad=True)
    losses = train_ssdg._compute_muse_unlabeled_losses(
        x_u=torch.randn(2, 2, 8),
        metadata=_metadata(),
        teacher_outputs=weak,
        student_outputs={
            "weak": weak,
            "strong": strong,
            "nuisance": {"z_dom": nuisance_z_dom},
            "satellite": None,
        },
        muse_state=state,
        simulator_metadata={
            "nuisance": torch.ones(2, 6),
            "nuisance_valid": torch.ones(2, dtype=torch.bool),
        },
    )
    losses["nuisance"].backward()
    assert nuisance_z_dom.grad is not None
    assert torch.isfinite(nuisance_z_dom.grad).all()
    assert torch.count_nonzero(nuisance_z_dom.grad).item() > 0
    assert strong["z_dom"].grad is None


def test_nuisance_detached_ablation_blocks_backbone_view_gradient():
    args = _args("M1")
    args.muse_nuisance_detached = True
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(1, state["config"])
    weak, strong = _outputs()
    nuisance_z_dom = torch.randn(2, 4, requires_grad=True)
    losses = train_ssdg._compute_muse_unlabeled_losses(
        x_u=torch.randn(2, 2, 8),
        metadata=_metadata(),
        teacher_outputs=weak,
        student_outputs={
            "weak": weak,
            "strong": strong,
            "nuisance": {"z_dom": nuisance_z_dom},
            "satellite": None,
        },
        muse_state=state,
        simulator_metadata={
            "nuisance": torch.ones(2, 6),
            "nuisance_valid": torch.ones(2, dtype=torch.bool),
        },
    )
    losses["nuisance"].backward()
    assert nuisance_z_dom.grad is None
    assert any(
        parameter.grad is not None
        for name, parameter in state["heads"].named_parameters()
        if name.startswith("nuisance_head")
    )


def test_muse_nuisance_view_is_built_for_m1_without_satellite_identity_output(monkeypatch):
    x_u = torch.zeros(2, 2, 8)
    d_u = torch.tensor([0, 1])
    nuisance = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    valid = torch.ones(2, dtype=torch.bool)

    def fake_apply(x, scenario, args, gen, return_meta):
        assert return_meta is True
        return x + 3.0, {"scenario": scenario}

    def fake_normalize(raw, scenario, batch_size, device):
        assert raw == {"scenario": scenario}
        assert batch_size == 2
        return raw, nuisance.to(device), valid.to(device), ("a", "b", "c", "d", "e", "f")

    class _ViewModel(nn.Module):
        def forward(self, x, **kwargs):
            return {"z_dom": x.flatten(1)[:, :4], "tx_logits": torch.ones(2, 3)}

    monkeypatch.setattr(train_ssdg, "apply_sat_channel_for_scenario", fake_apply)
    monkeypatch.setattr(train_ssdg, "normalize_crra_nuisance_meta", fake_normalize)
    outputs, metadata = train_ssdg._build_muse_nuisance_view(
        _ViewModel(),
        x_u,
        d_u,
        _args("M1"),
        grl_lambda=0.05,
        generator=None,
    )
    assert torch.equal(outputs["z_dom"], torch.full((2, 4), 3.0))
    assert torch.equal(metadata["nuisance"], nuisance)
    assert torch.equal(metadata["nuisance_valid"], valid)


@pytest.mark.parametrize("level", ["M1", "M2", "M3"])
def test_muse_s1_reads_unlabeled_without_identity_graph_or_updates(level, monkeypatch):
    state = train_ssdg._initialize_muse_training_state(
        _args(level), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(1, state["config"])
    weak, strong = _outputs()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("S1 must not build the U identity graph")

    monkeypatch.setattr(state["heads"], "local_prob", forbidden)
    monkeypatch.setattr(
        state["classification_prototypes"], "class_probabilities", forbidden
    )
    monkeypatch.setattr(state["temporal_memory"], "observe", forbidden)
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
    assert losses["u_identity_selected_count"].item() == 0.0
    assert losses["u_satellite_identity_selected_count"].item() == 0.0
    assert losses["prototype_update_count"].item() == 0.0
    assert losses["evidence_probabilities"] == {}
    assert losses["base"].requires_grad
    losses["total"].backward()
    assert strong["tx_logits"].grad is None or torch.equal(
        strong["tx_logits"].grad, torch.zeros_like(strong["tx_logits"])
    )
    assert state["classification_prototypes"].state_dict()["counts"] == {}


def test_muse_m2_routes_are_a_partition_and_m3_updates_only_classification_bank():
    args = _args("M3")
    args.muse_hard_max_fraction = 1.0
    args.muse_identity_max_fraction = 1.0
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    weak, strong = _outputs()
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    train_ssdg._compute_muse_labeled_auxiliary_loss(
        weak["z_id"],
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        state,
    )
    agreed_probability = torch.softmax(weak["tx_logits"].detach().float(), dim=-1)
    state["heads"].local_prob = lambda features, domains: agreed_probability.to(
        features.device
    )
    state["classification_prototypes"].class_probabilities = (
        lambda features, num_classes: agreed_probability.to(features.device)
    )
    state["temporal_memory"].observe = lambda keys, pseudo, confidence, epoch: torch.ones(
        pseudo.numel(), dtype=torch.bool, device=pseudo.device
    )
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
    counts = state["classification_prototypes"].state_dict()["counts"]
    assert counts[0] > 1.0
    assert counts[1] > 1.0


def test_m3_satellite_identity_ce_uses_only_strict_hard_rows():
    args = _args("M3")
    args.muse_high_threshold = 0.0
    args.muse_low_threshold = 0.0
    args.muse_hard_max_fraction = 1.0
    args.muse_identity_max_fraction = 1.0
    args.muse_lambda_satellite = 0.68
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["temporal_memory"].observe = lambda keys, pseudo, confidence, epoch: torch.ones(
        pseudo.numel(), dtype=torch.bool, device=pseudo.device
    )
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    weak, strong = _outputs()
    train_ssdg._compute_muse_labeled_auxiliary_loss(
        weak["z_id"], torch.tensor([0, 1]), torch.tensor([0, 1]), state
    )
    agreed_probability = torch.softmax(weak["tx_logits"].detach().float(), dim=-1)
    state["heads"].local_prob = lambda features, domains: agreed_probability.to(
        features.device
    )
    state["classification_prototypes"].class_probabilities = (
        lambda features, num_classes: agreed_probability.to(features.device)
    )
    keys = [(0, 0, 0, 0, 100), (0, 0, 0, 1, 101)]
    satellite_mask = train_ssdg.select_satellite_student_mask(
        keys, epoch=69, probability=0.5, seed=392002
    )
    assert satellite_mask.tolist() == [False, True]
    satellite = {
        **strong,
        "tx_logits": torch.tensor(
            [[-4.0, 4.0, 0.0], [4.0, -4.0, 0.0]], requires_grad=True
        ),
        "z_id": torch.tensor(
            [[-1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            requires_grad=True,
        ),
    }
    losses = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {
            **_metadata(),
            "memory_keys": keys,
            "satellite_mask": satellite_mask,
            "epoch": 69,
        },
        weak,
        {"weak": weak, "strong": strong, "satellite": satellite},
        state,
        {},
    )

    selected = losses["identity_student"]
    assert torch.equal(selected["tx_logits"], strong["tx_logits"])
    assert torch.equal(selected["z_id"], strong["z_id"])
    assert losses["identity_student_satellite_mask"].tolist() == [False, True]
    assert torch.allclose(
        losses["self"],
        state["heads"].self_supervised_loss(weak["z_id"], selected["z_id"]),
    )

    losses["satellite"].backward()
    assert losses["u_satellite_identity_selected_count"].item() == 1.0
    assert strong["tx_logits"].grad is None or torch.count_nonzero(strong["tx_logits"].grad).item() == 0
    assert torch.count_nonzero(satellite["tx_logits"].grad[0]).item() == 0
    assert torch.count_nonzero(satellite["tx_logits"].grad[1]).item() > 0


def test_fasttrust_hard_target_remains_three_head_consensus_when_prior_flips_fusion():
    args = _args("M2")
    args.muse_high_threshold = 0.0
    args.muse_low_threshold = 0.0
    args.muse_hard_max_fraction = 1.0
    args.muse_identity_max_fraction = 1.0
    args.muse_prior_alignment_gamma = 1.0
    args.muse_reliability_confidence_weight = 1.0
    args.muse_reliability_margin_weight = 0.0
    args.muse_reliability_js_weight = 0.0
    args.muse_reliability_prototype_weight = 0.0
    args.muse_reliability_stability_weight = 0.0
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    state["temporal_memory"].observe = lambda keys, pseudo, confidence, epoch: torch.ones(
        pseudo.numel(), dtype=torch.bool, device=pseudo.device
    )
    state["source_global_class_counts"].copy_(torch.tensor([100.0, 100.0, 1.0]))
    state["source_domain_class_counts"][0].copy_(torch.tensor([1000.0, 1.0, 1.0]))
    state["source_domain_class_counts"][1].copy_(torch.tensor([1000.0, 1.0, 1.0]))
    consensus = torch.tensor([[0.51, 0.49, 1e-6], [0.51, 0.49, 1e-6]])
    consensus = consensus / consensus.sum(dim=-1, keepdim=True)
    state["heads"].local_prob = lambda features, domains: consensus.to(features.device)
    state["classification_prototypes"].class_probabilities = (
        lambda features, num_classes: consensus.to(features.device)
    )
    weak, strong = _outputs()
    weak["tx_logits"] = consensus.log()
    losses = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": 69},
        weak,
        {"weak": weak, "strong": strong, "nuisance": None, "satellite": None},
        state,
        {},
    )

    assert losses["fused_probability"].argmax(dim=-1).tolist() == [1, 1]
    assert losses["pseudo"].tolist() == [0, 0]
    assert torch.allclose(
        losses["hard"],
        torch.nn.functional.cross_entropy(strong["tx_logits"], torch.zeros(2, dtype=torch.long)),
    )


@pytest.mark.parametrize("level", ["M1", "M2"])
def test_m1_m2_never_enable_satellite_identity_student(level):
    state = train_ssdg._initialize_muse_training_state(
        _args(level), _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    weak, strong = _outputs()
    satellite = {
        **strong,
        "tx_logits": -strong["tx_logits"],
        "z_id": -strong["z_id"],
    }
    losses = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "satellite_mask": torch.tensor([True, True]), "epoch": 69},
        weak,
        {"weak": weak, "strong": strong, "satellite": satellite},
        state,
        {},
    )
    assert torch.equal(losses["identity_student"]["tx_logits"], strong["tx_logits"])
    assert torch.equal(losses["identity_student"]["z_id"], strong["z_id"])
    assert losses["identity_student_satellite_mask"].tolist() == [False, False]


def test_muse_m2_low_route_uses_candidate_set_not_true_identity_targets():
    args = _args("M2")
    args.muse_high_threshold = 1.0
    args.muse_low_threshold = 0.99
    args.muse_lambda_low_entropy = 0.0
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(69, state["config"])
    weak, strong = _outputs()
    without_candidate = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        _metadata(),
        weak,
        {"weak": weak, "strong": strong, "satellite": None},
        state,
        {},
    )
    args.muse_lambda_low_entropy = 1.0
    with_candidate = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        _metadata(),
        weak,
        {"weak": weak, "strong": strong, "satellite": None},
        state,
        {},
    )
    assert without_candidate["route"].low.tolist() == [True, True]
    assert without_candidate["identity"].item() == 0.0
    assert with_candidate["candidate"].item() > 0.0
    assert with_candidate["identity"].item() > 0.0


@pytest.mark.parametrize("epoch, candidate_active", [(17, False), (40, False), (41, True)])
def test_muse_candidate_supervision_starts_only_at_s2b(epoch, candidate_active):
    args = _args("M2")
    args.muse_high_threshold = 1.0
    args.muse_low_threshold = 1.0
    args.muse_candidate_mass = 0.75
    args.muse_candidate_max_classes = 1
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(epoch, state["config"])
    weak, strong = _outputs()
    losses = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": epoch},
        weak,
        {"weak": weak, "strong": strong, "nuisance": None, "satellite": None},
        state,
        {},
    )
    assert losses["route"].low.tolist() == [True, True]
    assert (losses["candidate"].item() > 0.0) is candidate_active


@pytest.mark.parametrize(
    "high_threshold,low_threshold,expected_key",
    [(0.0, 0.0, "soft"), (1.0, 0.0, "soft")],
)
def test_muse_s2a_first_observation_routes_unstable_high_as_soft(
    high_threshold, low_threshold, expected_key
):
    args = _args("M2")
    args.muse_high_threshold = high_threshold
    args.muse_low_threshold = low_threshold
    state = train_ssdg._initialize_muse_training_state(
        args, _TinyMUSEModel(), torch.device("cpu")
    )
    state["schedule_state"] = muse_schedule_for_epoch(17, state["config"])
    weak, strong = _outputs()
    losses = train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": 17},
        weak,
        {"weak": weak, "strong": strong, "nuisance": None, "satellite": None},
        state,
        {},
    )
    assert losses[expected_key].item() > 0.0
    assert losses["candidate"].item() == 0.0
    assert losses["hard"].item() == 0.0


def test_missing_classification_prototype_contributes_zero_reliability_evidence():
    state = train_ssdg._initialize_muse_training_state(
        _args("M2"), _TinyMUSEModel(), torch.device("cpu")
    )
    distances = train_ssdg._muse_prototype_distance(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([2]),
        state["classification_prototypes"],
    )
    reliability = train_ssdg.compute_muse_reliability(
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1),
        distances,
        torch.ones(1),
        weights=[0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert reliability.item() == 0.0


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


def test_epoch_181_freezes_muse_statistics_prior_and_local_teacher_state():
    state = train_ssdg._initialize_muse_training_state(
        _args("M3"), _TinyMUSEModel(), torch.device("cpu")
    )
    train_ssdg._configure_muse_epoch_state(state, 180)
    weak, strong = _outputs()
    train_ssdg._compute_muse_labeled_auxiliary_loss(
        weak["z_id"],
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        state,
    )
    state["temporal_memory"].observe(
        _metadata()["memory_keys"],
        torch.tensor([0, 1]),
        torch.tensor([0.9, 0.9]),
        180,
    )

    schedule = train_ssdg._configure_muse_epoch_state(state, 181)
    before_memory = state["temporal_memory"].state_dict()
    before_prototype = state["classification_prototypes"].state_dict()
    before_global_prior = state["source_global_class_counts"].clone()
    before_domain_prior = state["source_domain_class_counts"].clone()
    fixed_output = state["heads"].local_prob(
        weak["z_id"].detach(), torch.tensor([0, 1])
    ).detach().clone()

    state["heads"].train()
    train_ssdg._compute_muse_labeled_auxiliary_loss(
        strong["z_id"],
        torch.tensor([2, 2]),
        torch.tensor([0, 1]),
        state,
    )
    train_ssdg._compute_muse_unlabeled_losses(
        torch.randn(2, 2, 8),
        {**_metadata(), "epoch": 181},
        weak,
        {"weak": weak, "strong": strong, "satellite": strong},
        state,
        {},
    )

    assert schedule.freeze_statistics is True
    assert state["heads"].local_teacher_frozen is True
    assert state["temporal_memory"].state_dict() == before_memory
    after_prototype = state["classification_prototypes"].state_dict()
    assert after_prototype["counts"] == before_prototype["counts"]
    assert all(
        torch.equal(after_prototype["prototypes"][key], value)
        for key, value in before_prototype["prototypes"].items()
    )
    assert torch.equal(state["source_global_class_counts"], before_global_prior)
    assert torch.equal(state["source_domain_class_counts"], before_domain_prior)
    assert torch.equal(
        state["heads"].local_prob(weak["z_id"].detach(), torch.tensor([0, 1])),
        fixed_output,
    )


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
        "muse_source_global_class_counts",
            "muse_source_domain_class_counts",
            "muse_source_prior_frozen",
            "muse_schedule_state",
            "rc4_calibration",
            "sat_anchor_thresholds",
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


def test_muse_pseudo_gate_pass_rates_are_defined_for_first_batch_telemetry():
    domain_pass, temporal_pass, strong_pass = train_ssdg._pseudo_gate_pass_rates(
        domain_mask=torch.tensor([True, False, True, False]),
        temporal_mask=torch.tensor([True, True, True, False]),
        strong_mask=torch.tensor([False, True, False, False]),
    )

    assert domain_pass.item() == 0.5
    assert temporal_pass.item() == 0.75
    assert strong_pass.item() == 0.25


def test_final_only_source_contains_no_source_validation_checkpoint_branch():
    source = Path("code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    assert "best_source_validation_ssdg.pth" not in source
    assert '"checkpoint_selection": "source_validation_only"' not in source
    assert '"selection_source": "source_validation_only"' not in source
    assert '"checkpoint_role": "source_validation_selected"' not in source
