from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

CALIBRATION_KWARGS = {"lr_min": 0.001, "lr_max": 0.01}


def _base_state() -> dict[str, torch.Tensor]:
    return {
        "id_backbone.t3.weight": torch.zeros(2),
        "id_backbone.fusion.bias": torch.zeros(1),
        "classifier.weight": torch.ones(1),
        "running_count": torch.tensor([3], dtype=torch.int64),
    }


def _bank():
    from cvsrffi.meta_weight_bank import DeltaTaskKey, fit_weight_delta_bank

    return fit_weight_delta_bank(
        "base-a",
        {
            DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10): {
                "id_backbone.t3.weight": torch.tensor([1.0, 2.0]),
                "id_backbone.fusion.bias": torch.tensor([3.0]),
            },
            DeltaTaskKey("rx-b", "d1", "leo_clear_weak", 10): {
                "id_backbone.t3.weight": torch.tensor([-1.0, -2.0]),
                "id_backbone.fusion.bias": torch.tensor([-3.0]),
            },
        },
    )


def _state(bank, *, q: torch.Tensor | None = None):
    from cvsrffi.meta_support_set_encoder import SupportDomainState

    return SupportDomainState(
        q=torch.ones(sum(entry.effective_rank for entry in bank.entries)) if q is None else q,
        uncertainty=torch.tensor(0.0),
        block_gates=torch.ones(len(bank.entries)),
        block_lrs=torch.full((len(bank.entries),), 0.01),
    )


def _assert_exact_base_copy(base: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> None:
    assert tuple(state) == tuple(base)
    for name, value in base.items():
        assert torch.equal(
            state[name].detach().contiguous().view(torch.uint8),
            value.detach().contiguous().view(torch.uint8),
        )
        assert state[name].data_ptr() != value.data_ptr()


def _assert_safe_fallback_metadata(plan, block_count: int) -> None:
    assert plan.uncertainty == 1.0
    assert plan.block_gates == (0.0,) * block_count
    assert plan.block_lrs == (0.001,) * block_count


def test_calibrator_applies_only_bank_allowlisted_blocks_and_never_mutates_base() -> None:
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    bank = _bank()
    plan = calibrate_weight_plan(base, "base-a", bank, _state(bank), **CALIBRATION_KWARGS)

    assert plan.applied is True
    assert plan.reason == "applied"
    assert not torch.equal(plan.state_dict["id_backbone.t3.weight"], base["id_backbone.t3.weight"])
    assert not torch.equal(plan.state_dict["id_backbone.fusion.bias"], base["id_backbone.fusion.bias"])
    assert torch.equal(plan.state_dict["classifier.weight"], base["classifier.weight"])
    assert torch.equal(plan.state_dict["running_count"], base["running_count"])
    assert torch.equal(base["id_backbone.t3.weight"], torch.zeros(2))
    assert plan.state_dict["classifier.weight"].data_ptr() != base["classifier.weight"].data_ptr()


@pytest.mark.parametrize("kind", ["binding", "geometry", "nonfinite"])
def test_calibrator_atomically_falls_back_to_complete_base_copy(kind: str) -> None:
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan
    from cvsrffi.meta_weight_bank import BlockSpec, DeltaBankEntry, WeightDeltaBank

    base = _base_state()
    bank = _bank()
    state = _state(bank)
    checkpoint_id = "base-a"
    if kind == "binding":
        checkpoint_id = "wrong-base"
    elif kind == "geometry":
        entry = bank.entries[0]
        bad_spec = BlockSpec(entry.spec.name, entry.spec.parameter_names, ((9,),), entry.spec.dtypes)
        bank = WeightDeltaBank(bank.schema, bank.base_checkpoint_id, bank.task_keys, (replace(entry, spec=bad_spec), *bank.entries[1:]))
    else:
        entry = bank.entries[0]
        bad_entry = DeltaBankEntry(entry.spec, torch.full_like(entry.basis, float("nan")), entry.task_coefficients, entry.effective_rank, entry.relative_error)
        bank = WeightDeltaBank(bank.schema, bank.base_checkpoint_id, bank.task_keys, (bad_entry, *bank.entries[1:]))

    plan = calibrate_weight_plan(base, checkpoint_id, bank, state, **CALIBRATION_KWARGS)

    assert plan.applied is False
    _assert_exact_base_copy(base, plan.state_dict)
    _assert_safe_fallback_metadata(plan, len(bank.entries))


def test_calibrator_rejects_mismatched_gate_or_learning_rate_geometry_atomically() -> None:
    from cvsrffi.meta_support_set_encoder import SupportDomainState
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    bank = _bank()
    state = SupportDomainState(
        q=torch.ones(bank.entries[0].effective_rank),
        uncertainty=torch.tensor(0.0),
        block_gates=torch.ones(len(bank.entries) - 1),
        block_lrs=torch.ones(len(bank.entries)),
    )

    plan = calibrate_weight_plan(base, "base-a", bank, state, **CALIBRATION_KWARGS)

    assert plan.applied is False
    _assert_exact_base_copy(base, plan.state_dict)
    _assert_safe_fallback_metadata(plan, len(bank.entries))


def test_calibrator_applies_heterogeneous_rank_segments_in_stable_bank_order() -> None:
    from cvsrffi.meta_support_set_encoder import SupportDomainState
    from cvsrffi.meta_weight_bank import BlockSpec, DeltaBankEntry, WeightDeltaBank, WEIGHT_DELTA_BANK_SCHEMA
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    bank = WeightDeltaBank(
        schema=WEIGHT_DELTA_BANK_SCHEMA,
        base_checkpoint_id="base-a",
        task_keys=(),
        entries=(
            DeltaBankEntry(
                BlockSpec("t3", ("id_backbone.t3.weight",), ((2,),), ("torch.float32",)),
                torch.tensor([[1.0], [0.5]], dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.float32),
                1,
                0.0,
            ),
            DeltaBankEntry(
                BlockSpec("fusion", ("id_backbone.fusion.bias",), ((1,),), ("torch.float32",)),
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                torch.ones((1, 2), dtype=torch.float32),
                2,
                0.0,
            ),
        ),
    )
    state = SupportDomainState(
        q=torch.tensor([2.0, 3.0, 4.0]),
        uncertainty=torch.tensor(0.0),
        block_gates=torch.ones(2),
        block_lrs=torch.full((2,), 0.005),
    )

    plan = calibrate_weight_plan(base, "base-a", bank, state, **CALIBRATION_KWARGS)

    assert plan.applied is True
    assert torch.equal(plan.state_dict["id_backbone.t3.weight"], torch.tensor([2.0, 1.0]))
    assert torch.equal(plan.state_dict["id_backbone.fusion.bias"], torch.tensor([11.0]))


@pytest.mark.parametrize("bad_lr", [0.0009, 0.0101])
def test_calibrator_falls_back_for_learning_rate_outside_frozen_interval(bad_lr: float) -> None:
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    bank = _bank()
    state = _state(bank)
    state = replace(state, block_lrs=torch.full((len(bank.entries),), bad_lr))

    plan = calibrate_weight_plan(base, "base-a", bank, state, **CALIBRATION_KWARGS)

    assert plan.applied is False
    _assert_exact_base_copy(base, plan.state_dict)
    _assert_safe_fallback_metadata(plan, len(bank.entries))


def test_calibrator_normalizes_invalid_failure_metadata_before_returning_fallback() -> None:
    from cvsrffi.meta_support_set_encoder import SupportDomainState
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    bank = _bank()
    state = SupportDomainState(
        q=torch.full((sum(entry.effective_rank for entry in bank.entries),), float("nan")),
        uncertainty=torch.tensor(float("nan")),
        block_gates=torch.full((len(bank.entries),), float("inf")),
        block_lrs=torch.full((len(bank.entries),), float("nan")),
    )

    plan = calibrate_weight_plan(base, "base-a", bank, state, **CALIBRATION_KWARGS)

    assert plan.applied is False
    _assert_exact_base_copy(base, plan.state_dict)
    _assert_safe_fallback_metadata(plan, len(bank.entries))


@pytest.mark.parametrize("name", ["classifier.weight", "running_count"])
def test_calibrator_falls_back_when_any_base_tensor_is_nonfinite(name: str) -> None:
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    base[name] = torch.tensor([float("nan")])
    bank = _bank()

    plan = calibrate_weight_plan(base, "base-a", bank, _state(bank), **CALIBRATION_KWARGS)

    assert plan.applied is False
    _assert_exact_base_copy(base, plan.state_dict)
    _assert_safe_fallback_metadata(plan, len(bank.entries))


def test_calibrator_falls_back_for_duplicate_block_name_even_with_disjoint_parameters() -> None:
    from cvsrffi.meta_weight_bank import BlockSpec, WeightDeltaBank
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    base = _base_state()
    bank = _bank()
    second = bank.entries[1]
    duplicate_name_spec = BlockSpec(
        bank.entries[0].spec.name,
        second.spec.parameter_names,
        second.spec.shapes,
        second.spec.dtypes,
    )
    duplicate_name_bank = WeightDeltaBank(
        bank.schema,
        bank.base_checkpoint_id,
        bank.task_keys,
        (bank.entries[0], replace(second, spec=duplicate_name_spec)),
    )

    plan = calibrate_weight_plan(base, "base-a", duplicate_name_bank, _state(duplicate_name_bank), **CALIBRATION_KWARGS)

    assert plan.applied is False
    assert "duplicate block name" in plan.reason
    _assert_exact_base_copy(base, plan.state_dict)
    _assert_safe_fallback_metadata(plan, len(bank.entries))
