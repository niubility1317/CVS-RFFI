from __future__ import annotations

import torch
from torch import nn

import pytest

import cvsrffi.stage2_wiser_runner as runner_module
from cvsrffi.stage2_wiser_p3 import identity_fft_penalties
from cvsrffi.stage2_wiser_runner import (
    WISERP3TrainingConfig,
    WISERTrainingConfig,
    predict_wiser_representation_probes,
    select_support_safe_interpolation,
    train_wiser_p3_arm,
    train_wiser_arm,
)
from cvsrffi.wiser_source_summary import QuantizedSourceSummary


class _TinyIdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sinc = nn.Identity()
        self.time_fuse = nn.Linear(4, 4)
        self.t1 = nn.Linear(4, 4)
        self.t2 = nn.Linear(4, 4)
        self.t3 = nn.Linear(4, 4)
        self.f1 = nn.Linear(4, 4)
        self.f2 = nn.Linear(4, 4)
        self.f3 = nn.Linear(4, 4)
        self.t_proj = nn.Linear(4, 4)
        self.f_proj = nn.Linear(4, 4)
        self.freq_gate = nn.Linear(4, 4)
        self.fuse = nn.Linear(8, 4)
        self.cls_head = nn.Module()
        self.cls_head.id_proj = nn.Linear(4, 4)
        self.cls_head.id_gate = nn.Linear(8, 4)
        self.cls_head.joint_proj = nn.Linear(12, 4)
        self.cls_head.head = nn.Linear(4, 2, bias=False)


class _TinyDualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _TinyIdentityBackbone()
        self.dom_backbone = nn.Linear(4, 4)
        self.dom_head = nn.Linear(4, 2)
        self.adv_head = nn.Linear(4, 2)

    def forward(
        self,
        value: torch.Tensor,
        y_tx: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        flat = value.flatten(1)
        t = torch.relu(self.id_backbone.time_fuse(flat))
        t = torch.relu(self.id_backbone.t1(t))
        t = torch.relu(self.id_backbone.t2(t))
        t = torch.relu(self.id_backbone.t3(t))
        f = torch.relu(self.id_backbone.freq_gate(flat))
        f = torch.relu(self.id_backbone.f1(f))
        f = torch.relu(self.id_backbone.f2(f))
        f = torch.relu(self.id_backbone.f3(f))
        base = torch.relu(
            self.id_backbone.fuse(
                torch.cat(
                    (self.id_backbone.t_proj(t), self.id_backbone.f_proj(f)), dim=1
                )
            )
        )
        identity = torch.relu(self.id_backbone.cls_head.id_proj(base))
        zeros = torch.zeros_like(identity)
        joint = torch.relu(
            self.id_backbone.cls_head.joint_proj(
                torch.cat((identity, zeros, zeros), dim=1)
            )
        )
        logits = self.id_backbone.cls_head.head(joint)
        if return_aux:
            return {"tx_logits": logits, "z_id": joint, "aux_id": {}}
        return logits


def _summary() -> QuantizedSourceSummary:
    points = torch.tensor(
        [
            [[1.0, 0.0, 0.1, 0.0], [1.0, 0.0, -0.1, 0.0]],
            [[0.0, 1.0, 0.0, 0.1], [0.0, 1.0, 0.0, -0.1]],
        ]
    )
    centers = torch.nn.functional.normalize(points.mean(dim=1), dim=1)
    empty = torch.empty(0)
    return QuantizedSourceSummary(
        feature_schema="test4",
        class_registry=("c0", "c1"),
        centers=centers,
        basis=empty,
        coefficients=empty,
        radii=empty,
        direct_points=points,
    )


def test_train_b_arm_combines_dual_vsw_l2sp_without_query_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    seeds: list[int] = []
    original_vsw = runner_module.classwise_sliced_wasserstein

    def traced_vsw(*args, seed: int, **kwargs):
        seeds.append(seed)
        return original_vsw(*args, seed=seed, **kwargs)

    monkeypatch.setattr(runner_module, "classwise_sliced_wasserstein", traced_vsw)
    model = _TinyDualModel()
    support = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.1]],
            [[0.8, 0.2], [0.0, -0.1]],
            [[0.0, 1.0], [0.1, 0.0]],
            [[0.2, 0.8], [-0.1, 0.0]],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    audit = train_wiser_arm(
        model,
        support,
        labels,
        source_summary=_summary(),
        arm="B",
        config=WISERTrainingConfig(
            stage_steps=(1, 1, 1),
            lambda_sp=0.1,
            lambda_vsw=0.5,
            num_vsw_projections=4,
            seed=19,
        ),
    )

    assert audit.arm == "B"
    assert audit.optimizer_steps == 3
    assert audit.query_rows_used == 0
    assert audit.vsw_enabled is True
    assert audit.model_inversion_enabled is False
    assert len(audit.stage_audits) == 3
    assert seeds == [19, 19, 19]
    assert all(row["final_total_loss"] >= 0.0 for row in audit.stage_audits)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.training is False


def test_prediction_refuses_trainable_model_before_query_forward() -> None:
    model = _TinyDualModel()
    support = torch.zeros((4, 2, 2))
    query = torch.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="frozen eval model"):
        predict_wiser_representation_probes(
            model,
            support,
            torch.tensor([0, 0, 1, 1]),
            query,
            query_tokens=("q0", "q1"),
            source_summary=_summary(),
            seed=1,
        )


class _P3IdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sinc = nn.Identity()
        self.time_fuse = nn.Linear(160, 160)
        self.t1 = nn.Linear(160, 160)
        self.t2 = nn.Linear(160, 160)
        self.t3 = nn.Linear(160, 160)
        self.f1 = nn.Linear(160, 160)
        self.f2 = nn.Linear(160, 160)
        self.f3 = nn.Linear(160, 160)
        self.t_proj = nn.Linear(160, 160)
        self.f_proj = nn.Linear(160, 160)
        self.freq_gate = nn.Linear(160, 160)
        self.freq_stats_proj = nn.Linear(160, 160)
        self.fuse = nn.Linear(320, 160)
        self.identity_capacity = nn.Linear(160, 160)
        self.cls_head = nn.Module()
        self.cls_head.id_proj = nn.Linear(160, 160)
        self.cls_head.id_gate = nn.Linear(320, 160)
        self.cls_head.joint_proj = nn.Linear(480, 160)
        self.cls_head.head = nn.Linear(160, 6, bias=False)


class _P3DualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _P3IdentityBackbone()
        self.dom_backbone = nn.Linear(160, 160)
        self.dom_head = nn.Linear(160, 6)
        self.adv_head = nn.Linear(160, 6)
        self.meta_adapter_alpha = nn.Linear(160, 160)
        self.sat_anchor_identity_adapter = nn.Linear(160, 160)
        self.input_dropout = nn.Dropout(p=0.8)

    def forward(self, value: torch.Tensor, y_tx=None, return_aux: bool = False):
        flat = self.input_dropout(value.flatten(1)[:, :160])
        t = torch.tanh(self.id_backbone.t3(torch.tanh(self.id_backbone.t2(torch.tanh(self.id_backbone.t1(self.id_backbone.time_fuse(flat)))))))
        f = torch.tanh(self.id_backbone.f3(torch.tanh(self.id_backbone.f2(torch.tanh(self.id_backbone.f1(self.id_backbone.freq_gate(flat)))))))
        base = torch.tanh(self.id_backbone.fuse(torch.cat((self.id_backbone.t_proj(t), self.id_backbone.f_proj(f)), dim=1)))
        identity = torch.tanh(self.id_backbone.cls_head.id_proj(base))
        zeros = torch.zeros_like(identity)
        joint = torch.tanh(self.id_backbone.cls_head.joint_proj(torch.cat((identity, zeros, zeros), dim=1)))
        logits = self.id_backbone.cls_head.head(joint)
        return {"tx_logits": logits, "z_id": joint, "aux_id": {}} if return_aux else logits


def _p3_support_fixture() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], QuantizedSourceSummary]:
    generator = torch.Generator().manual_seed(713102)
    values = torch.randn((60, 2, 256), generator=generator) * 0.05
    labels = torch.arange(6).repeat_interleave(10)
    for class_id in range(6):
        values[labels == class_id, 0, class_id] += 1.0
    tokens = tuple(f"p3-support-{index:02d}" for index in range(60))
    points = torch.randn((6, 2, 160), generator=generator)
    empty = torch.empty(0)
    summary = QuantizedSourceSummary(
        feature_schema="p3-test160",
        class_registry=tuple(f"c{index}" for index in range(6)),
        centers=torch.nn.functional.normalize(points.mean(dim=1), dim=1),
        basis=empty, coefficients=empty, radii=empty, direct_points=points,
    )
    return values, labels, tokens, summary


@pytest.mark.parametrize("arm", ["N2", "N3", "N4", "N5", "N6"])
def test_p3_arm_uses_only_support_and_refreezes(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _P3DualModel()
    support, labels, tokens, summary = _p3_support_fixture()
    forward_modes: list[tuple[bool, bool]] = []
    original_forward = runner_module._forward_identity

    def traced_forward(current_model, values):
        forward_modes.append((current_model.training, any(parameter.requires_grad for parameter in current_model.parameters())))
        return original_forward(current_model, values)

    monkeypatch.setattr(runner_module, "_forward_identity", traced_forward)

    audit = train_wiser_p3_arm(
        model, support, labels, support_tokens=tokens, source_summary=summary, arm=arm,
        expected_source_class_registry=tuple(f"c{index}" for index in range(6)),
        expected_source_feature_schema="p3-test160",
        config=WISERP3TrainingConfig(
            stage_steps=(1, 0, 0), diagnostic_interval=1, interpolation_grid=(0.0,)
        ),
    )

    assert audit.query_rows_used == 0
    assert audit.optimizer_steps > 0
    assert audit.baseline_joint_condition_number >= 1.0
    assert forward_modes[0] == (False, False)
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert {row["branch"] for row in audit.stage_audits if row["branch"].startswith("stage2")} == {
        "stage2_time", "stage2_frequency", "stage2_joint"
    }
    stage3 = next(row for row in audit.stage_audits if row["branch"] == "stage3")
    assert stage3["parent_branch"] in {"stage2_time", "stage2_frequency", "stage2_joint"}
    stage1 = next(row for row in audit.stage_audits if row["branch"] == "stage1_time")
    stage2 = [row for row in audit.stage_audits if row["branch"].startswith("stage2")]
    assert all(row["input_duals"] == stage1["output_duals"] for row in stage2)
    selected_stage2 = next(row for row in stage2 if row["branch"] == stage3["parent_branch"])
    assert stage3["input_duals"] == selected_stage2["output_duals"]
    assert all(not name.startswith(("dom_", "adv_", "meta_adapter_", "sat_anchor_")) for name in audit.reached_parameter_names)


def test_p3_rejects_invalid_support_with_model_refrozen() -> None:
    model = _P3DualModel()
    bad_support = torch.zeros((6, 2, 255))

    with pytest.raises(ValueError, match=r"\[rows,2,256\]"):
        train_wiser_p3_arm(
            model, bad_support, torch.arange(6), support_tokens=tuple(str(index) for index in range(6)),
            source_summary=None, arm="N2", config=WISERP3TrainingConfig(stage_steps=(1, 0, 0)),
        )

    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_stage_branch_selection_is_support_only_and_falls_back_to_alpha_zero() -> None:
    base = {"weight": torch.tensor([1.0]), "frozen": torch.tensor([9.0])}
    candidate = {"weight": torch.tensor([5.0]), "frozen": torch.tensor([2.0])}

    result = select_support_safe_interpolation(
        base, candidate, evaluator=lambda _: {"safe": False}, grid=(1.0, 0.5, 0.0),
        trainable_parameter_names=("weight",),
    )

    assert result.alpha == 0.0
    assert result.query_rows_used == 0
    assert result.state["weight"].item() == 1.0
    assert result.state["frozen"].item() == 9.0


def test_interpolation_evaluates_base_even_when_grid_omits_zero() -> None:
    base = {"weight": torch.tensor([1.0])}
    candidate = {"weight": torch.tensor([5.0])}
    observed: list[float] = []

    result = select_support_safe_interpolation(
        base, candidate,
        evaluator=lambda state: observed.append(float(state["weight"])) or {"safe": False, "oof_p3_ba": 0.2},
        grid=(1.0,), trainable_parameter_names=("weight",),
    )

    assert result.alpha == 0.0
    assert observed == [1.0, 5.0]
    assert result.support_metrics["oof_p3_ba"] == 0.2


def test_p3_config_rejects_interpolation_grid_without_alpha_zero() -> None:
    with pytest.raises(ValueError, match="0.0"):
        WISERP3TrainingConfig(stage_steps=(1, 0, 0), interpolation_grid=(1.0,))


def test_n6_energy_guard_reaches_low_identity_gradients() -> None:
    identity = torch.full((60, 160), 1.0e-3, requires_grad=True)
    fft = torch.zeros((60, 96))
    labels = torch.arange(6).repeat_interleave(10)

    _, energy = identity_fft_penalties(
        identity, fft, labels, baseline_cross_covariance_frobenius=0.0, energy_floor=0.1
    )
    energy.backward()

    assert energy.item() > 0.0
    assert identity.grad is not None and torch.isfinite(identity.grad).all()
    assert identity.grad.abs().sum().item() > 0.0


def test_n4_source_binding_rejects_wrong_schema_or_registry() -> None:
    model = _P3DualModel()
    support, labels, tokens, summary = _p3_support_fixture()
    config = WISERP3TrainingConfig(stage_steps=(1, 0, 0), interpolation_grid=(0.0,))

    with pytest.raises(ValueError, match="explicit expected source class registry"):
        train_wiser_p3_arm(
            model, support, labels, support_tokens=tokens, source_summary=summary, arm="N4", config=config,
        )
    with pytest.raises(ValueError, match="registry"):
        train_wiser_p3_arm(
            model, support, labels, support_tokens=tokens, source_summary=summary, arm="N4", config=config,
            expected_source_class_registry=tuple(f"c{index}" for index in reversed(range(6))),
            expected_source_feature_schema="p3-test160",
        )
    with pytest.raises(ValueError, match="feature schema"):
        train_wiser_p3_arm(
            model, support, labels, support_tokens=tokens, source_summary=summary, arm="N4", config=config,
            expected_source_class_registry=tuple(f"c{index}" for index in range(6)),
            expected_source_feature_schema="wrong-schema",
        )
