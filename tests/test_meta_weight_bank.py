from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def test_extract_block_delta_changes_only_allowlisted_parameters() -> None:
    from cvsrffi.meta_weight_bank import extract_block_delta

    base = {
        "id_backbone.t3.weight": torch.zeros(2, 2),
        "classifier.weight": torch.ones(2, 2),
    }
    adapted = {
        "id_backbone.t3.weight": torch.eye(2),
        "classifier.weight": torch.full((2, 2), 7.0),
    }
    adapted["classifier.weight"] = base["classifier.weight"].clone()

    delta = extract_block_delta(base, adapted, prefixes=("id_backbone.t3.",))

    assert tuple(delta) == ("id_backbone.t3.weight",)
    assert torch.equal(delta["id_backbone.t3.weight"], torch.eye(2))


def test_extract_block_delta_rejects_classifier_prefix_even_when_changed() -> None:
    from cvsrffi.meta_weight_bank import extract_block_delta

    base = {"classifier.weight": torch.zeros(2, 2)}
    adapted = {"classifier.weight": torch.eye(2)}

    with pytest.raises(ValueError, match="canonical"):
        extract_block_delta(base, adapted, prefixes=("classifier.",))


@pytest.mark.parametrize(
    ("base", "adapted", "match"),
    [
        (
            {"id_backbone.t3.weight": torch.zeros(2, 2)},
            {"id_backbone.t3.weight": torch.zeros(3, 2)},
            "shape",
        ),
        (
            {"id_backbone.t3.weight": torch.zeros(2, dtype=torch.float32)},
            {"id_backbone.t3.weight": torch.zeros(2, dtype=torch.float64)},
            "dtype",
        ),
        (
            {"id_backbone.t3.weight": torch.zeros(2)},
            {"id_backbone.t3.weight": torch.tensor([float("inf"), 0.0])},
            "non-finite",
        ),
    ],
)
def test_extract_block_delta_rejects_allowlisted_geometry_or_value_drift(
    base: dict[str, torch.Tensor], adapted: dict[str, torch.Tensor], match: str
) -> None:
    from cvsrffi.meta_weight_bank import extract_block_delta

    with pytest.raises(ValueError, match=match):
        extract_block_delta(base, adapted, prefixes=("id_backbone.t3.",))


@pytest.mark.parametrize(
    ("name", "base_value", "adapted_value"),
    [
        ("running_count", torch.tensor([7], dtype=torch.int64), torch.tensor([8], dtype=torch.int64)),
        ("classifier.weight", torch.tensor([0.0]), torch.tensor([-0.0])),
    ],
)
def test_extract_block_delta_rejects_any_unallowlisted_tensor_change_bitwise(
    name: str, base_value: torch.Tensor, adapted_value: torch.Tensor
) -> None:
    from cvsrffi.meta_weight_bank import extract_block_delta

    base = {
        "id_backbone.t3.weight": torch.zeros(2, 2),
        name: base_value,
    }
    adapted = {
        "id_backbone.t3.weight": torch.eye(2),
        name: adapted_value,
    }

    with pytest.raises(ValueError, match="unallowlisted"):
        extract_block_delta(base, adapted, prefixes=("id_backbone.t3.",))


def test_parameter_block_key_routes_only_canonical_identity_blocks() -> None:
    from cvsrffi.meta_weight_bank import parameter_block_key

    assert parameter_block_key("id_backbone.t3.weight") == "t3"
    assert parameter_block_key("id_backbone.frequency_projection.weight") == "frequency_projection"
    assert parameter_block_key("id_backbone.freq_stats_proj.weight") == "frequency_projection"
    assert parameter_block_key("classifier.weight") is None


def test_package_exports_canonical_weight_delta_bank_interfaces() -> None:
    from cvsrffi import BlockSpec, WeightDeltaBank, parameter_block_key

    assert BlockSpec.__name__ == "BlockSpec"
    assert WeightDeltaBank.__name__ == "WeightDeltaBank"
    assert parameter_block_key("id_backbone.t1.weight") == "t1"


def test_fit_weight_delta_bank_rank_one_reconstructs_and_has_canonical_sign() -> None:
    from cvsrffi.meta_weight_bank import DeltaTaskKey, compose_weight_delta, fit_weight_delta_bank

    task_a = DeltaTaskKey(receiver="rx-b", day="d2", scene="leo_clear_weak", k_shot=10)
    task_b = DeltaTaskKey(receiver="rx-a", day="d1", scene="leo_rain_weak", k_shot=10)
    bank = fit_weight_delta_bank(
        "base-123",
        {
            task_a: {"id_backbone.t3.weight": torch.tensor([[-1.0, 0.0]])},
            task_b: {"id_backbone.t3.weight": torch.tensor([[1.0, 0.0]])},
        },
    )

    assert bank.schema == "cvs.marc_ot.weight_delta_bank.v1"
    assert bank.task_keys == (task_b, task_a)
    entry = bank.entries[0]
    assert entry.effective_rank == 1
    assert entry.relative_error == pytest.approx(0.0, abs=1e-7)
    assert entry.basis[0, 0] > 0.0
    reconstructed = compose_weight_delta(entry, entry.task_coefficients[0])
    assert torch.allclose(reconstructed["id_backbone.t3.weight"], torch.tensor([[1.0, 0.0]]))


def test_fit_weight_delta_bank_keeps_capture_blocks_as_distinct_deterministic_tasks() -> None:
    """Collapsing capture blocks would silently overwrite one physical-domain delta."""
    from cvsrffi.meta_weight_bank import DeltaTaskKey, fit_weight_delta_bank

    later_capture = DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10, "capture-b")
    earlier_capture = DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10, "capture-a")
    bank = fit_weight_delta_bank(
        "base-123",
        {
            later_capture: {"id_backbone.t3.weight": torch.tensor([0.0, 1.0])},
            earlier_capture: {"id_backbone.t3.weight": torch.tensor([1.0, 0.0])},
        },
    )

    assert bank.task_keys == (earlier_capture, later_capture)
    assert bank.entries[0].task_coefficients.shape[0] == 2


def test_fit_weight_delta_bank_respects_rank_cap_unless_error_threshold_requires_full_rank() -> None:
    from cvsrffi.meta_weight_bank import DeltaTaskKey, fit_weight_delta_bank

    deltas = {
        DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10): {
            "id_backbone.t3.weight": torch.tensor([1.0, 0.0])
        },
        DeltaTaskKey("rx-b", "d1", "leo_clear_weak", 10): {
            "id_backbone.t3.weight": torch.tensor([0.0, 1.0])
        },
    }
    capped = fit_weight_delta_bank("base-123", deltas, max_rank=1)
    exact = fit_weight_delta_bank("base-123", deltas, max_rank=1, max_relative_error=0.1)

    assert capped.entries[0].effective_rank == 1
    assert capped.entries[0].relative_error == pytest.approx(2.0**-0.5, rel=1e-6)
    assert exact.entries[0].effective_rank == 2
    assert exact.entries[0].relative_error == pytest.approx(0.0, abs=1e-7)


def test_fit_weight_delta_bank_preserves_single_task_delta() -> None:
    from cvsrffi.meta_weight_bank import DeltaTaskKey, compose_weight_delta, fit_weight_delta_bank

    task = DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10)
    bank = fit_weight_delta_bank(
        "base-123", {task: {"id_backbone.t3.weight": torch.tensor([1.0, -2.0])}}
    )

    entry = bank.entries[0]
    assert entry.effective_rank == 1
    assert entry.relative_error == pytest.approx(0.0, abs=2e-7)
    composed = compose_weight_delta(entry, entry.task_coefficients[0])
    assert torch.allclose(composed["id_backbone.t3.weight"], torch.tensor([1.0, -2.0]))


def test_compose_weight_delta_rejects_bad_coefficients() -> None:
    from cvsrffi.meta_weight_bank import DeltaTaskKey, compose_weight_delta, fit_weight_delta_bank

    bank = fit_weight_delta_bank(
        "base-123",
        {
            DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10): {
                "id_backbone.t3.weight": torch.tensor([1.0, 0.0])
            },
            DeltaTaskKey("rx-b", "d1", "leo_clear_weak", 10): {
                "id_backbone.t3.weight": torch.tensor([-1.0, 0.0])
            },
        },
    )
    entry = bank.entries[0]

    with pytest.raises(ValueError, match="shape"):
        compose_weight_delta(entry, torch.ones(2))
    with pytest.raises(ValueError, match="non-finite"):
        compose_weight_delta(entry, torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="floating"):
        compose_weight_delta(entry, torch.tensor([1], dtype=torch.int64))


def test_compose_weight_delta_rejects_non_fp32_basis_and_overflow() -> None:
    from cvsrffi.meta_weight_bank import BlockSpec, DeltaBankEntry, compose_weight_delta

    spec = BlockSpec(
        name="t3",
        parameter_names=("id_backbone.t3.weight",),
        shapes=((1,),),
        dtypes=("torch.float32",),
    )
    non_fp32 = DeltaBankEntry(
        spec=spec,
        basis=torch.ones((1, 1), dtype=torch.float16),
        task_coefficients=torch.ones((1, 1), dtype=torch.float32),
        effective_rank=1,
        relative_error=0.0,
    )
    multiplication_overflow = DeltaBankEntry(
        spec=spec,
        basis=torch.tensor([[torch.finfo(torch.float32).max]], dtype=torch.float32),
        task_coefficients=torch.ones((1, 1), dtype=torch.float32),
        effective_rank=1,
        relative_error=0.0,
    )

    with pytest.raises(ValueError, match="float32"):
        compose_weight_delta(non_fp32, torch.ones(1))
    with pytest.raises(ValueError, match="non-finite"):
        compose_weight_delta(multiplication_overflow, torch.tensor([2.0]))


def test_compose_weight_delta_rejects_low_precision_conversion_overflow() -> None:
    from cvsrffi.meta_weight_bank import BlockSpec, DeltaBankEntry, compose_weight_delta

    entry = DeltaBankEntry(
        spec=BlockSpec(
            name="t3",
            parameter_names=("id_backbone.t3.weight",),
            shapes=((1,),),
            dtypes=("torch.float16",),
        ),
        basis=torch.tensor([[65520.0]], dtype=torch.float32),
        task_coefficients=torch.ones((1, 1), dtype=torch.float32),
        effective_rank=1,
        relative_error=0.0,
    )

    with pytest.raises(ValueError, match="non-finite"):
        compose_weight_delta(entry, torch.ones(1))


@pytest.mark.parametrize(
    ("raw_task_key", "task_values"),
    [
        pytest.param("not-a-task-key", None, id="wrong-type"),
        pytest.param(None, ("", "d1", "leo_clear_weak", 10), id="empty-receiver"),
        pytest.param(None, ("rx-a", 1, "leo_clear_weak", 10), id="non-string-day"),
        pytest.param(None, ("rx-a", "d1", "leo_clear_weak", 0), id="zero-k"),
        pytest.param(None, ("rx-a", "d1", "leo_clear_weak", 1.5), id="non-integer-k"),
        pytest.param(None, ("rx-a", "d1", "leo_clear_weak", 10, ""), id="empty-capture"),
        pytest.param(None, ("rx-a", "d1", "leo_clear_weak", 10, 9), id="non-string-capture"),
    ],
)
def test_fit_weight_delta_bank_rejects_invalid_task_keys_before_sorting(
    raw_task_key: object, task_values: tuple[object, ...] | None
) -> None:
    from cvsrffi.meta_weight_bank import DeltaTaskKey, fit_weight_delta_bank

    task_key = raw_task_key if task_values is None else DeltaTaskKey(*task_values)

    with pytest.raises(ValueError, match="task key"):
        fit_weight_delta_bank(
            "base-123", {task_key: {"id_backbone.t3.weight": torch.tensor([1.0])}}
        )


@pytest.mark.parametrize("max_rank", [1.5, True, "1"])
def test_fit_weight_delta_bank_rejects_non_integer_max_rank(max_rank: object) -> None:
    from cvsrffi.meta_weight_bank import DeltaTaskKey, fit_weight_delta_bank

    with pytest.raises(ValueError, match="max_rank"):
        fit_weight_delta_bank(
            "base-123",
            {
                DeltaTaskKey("rx-a", "d1", "leo_clear_weak", 10): {
                    "id_backbone.t3.weight": torch.tensor([1.0])
                }
            },
            max_rank=max_rank,
        )
