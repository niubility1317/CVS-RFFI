from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _base_state():
    return {
        "id_backbone.t3.weight": torch.zeros(2, 2),
        "id_backbone.t3.bias": torch.zeros(2),
        "id_backbone.fusion.bias": torch.zeros(2),
        "classifier.weight": torch.ones(2, 2),
        "running_count": torch.tensor([4], dtype=torch.int64),
    }


def _bank():
    from cvsrffi.meta_weight_bank import DeltaTaskKey, fit_weight_delta_bank

    return fit_weight_delta_bank(
        "base-checkpoint-7",
        {
            DeltaTaskKey("rx-1", "day-a", "leo_clear_weak", 10, "capture-a"): {
                "id_backbone.t3.weight": torch.tensor([[0.3, -0.1], [0.2, 0.4]]),
                "id_backbone.t3.bias": torch.tensor([0.1, -0.2]),
                "id_backbone.fusion.bias": torch.tensor([0.2, -0.3]),
            },
            DeltaTaskKey("rx-2", "day-a", "leo_rain_weak", 10, "capture-b"): {
                "id_backbone.t3.weight": torch.tensor([[-0.2, 0.5], [0.1, -0.4]]),
                "id_backbone.t3.bias": torch.tensor([-0.3, 0.2]),
                "id_backbone.fusion.bias": torch.tensor([-0.1, 0.4]),
            },
        },
        max_rank=1,
    )


def _encoder(bank, *, feature_dim: int = 685):
    from cvsrffi.meta_support_set_encoder import SupportSetEncoder

    torch.manual_seed(101)
    return SupportSetEncoder(
        feature_dim=feature_dim,
        coefficient_dim=sum(entry.effective_rank for entry in bank.entries),
        block_count=len(bank.entries),
        hidden_dim=6,
        lr_min=0.002,
        lr_max=0.02,
    )


def _task_domain_descriptors(bank):
    from cvsrffi.meta_weight_bank_checkpoint import TaskDomainDescriptor

    return {
        task_key: TaskDomainDescriptor(
            values=torch.linspace(-1.0, 1.0, 685) + float(index),
            aggregation_count=20 + index,
        )
        for index, task_key in enumerate(bank.task_keys)
    }


def _expected_block_specs():
    from cvsrffi.meta_weight_bank import BlockSpec

    return (
        BlockSpec(
            "fusion",
            ("id_backbone.fusion.bias",),
            ((2,),),
            ("torch.float32",),
        ),
        BlockSpec(
            "t3",
            ("id_backbone.t3.bias", "id_backbone.t3.weight"),
            ((2,), (2, 2)),
            ("torch.float32", "torch.float32"),
        ),
    )


def _load_raw(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _save_bundle(path: Path):
    from cvsrffi.meta_weight_bank_checkpoint import save_meta_weight_bundle

    bank = _bank()
    encoder = _encoder(bank)
    save_meta_weight_bundle(
        path,
        base_checkpoint_id="base-checkpoint-7",
        base_state=_base_state(),
        bank=bank,
        support_encoder=encoder,
        expected_block_specs=_expected_block_specs(),
        task_domain_descriptors=_task_domain_descriptors(bank),
    )
    return bank, encoder


def test_meta_weight_bundle_round_trip_preserves_support_composition_bitwise(tmp_path: Path) -> None:
    """Serialization drift in bank or encoder state would change composed weights."""
    from cvsrffi.meta_weight_bank_checkpoint import (
        META_WEIGHT_BUNDLE_SCHEMA,
        load_meta_weight_bundle,
    )
    from cvsrffi.meta_weight_calibrator import calibrate_weight_plan

    path = tmp_path / "marc_ot.pt"
    original_bank, original_encoder = _save_bundle(path)
    raw = _load_raw(path)
    assert raw["schema"] == "marc_ot_weight_bank_v1"
    assert raw["support_feature"]["schema"] == "marc_ot.support.row.v1"
    assert raw["support_feature"]["dim"] == 685
    assert raw["support_feature"]["config"]["rf_lite_dim"] == 10
    assert raw["task_domain_bank"]["schema"] == "marc_ot.task_domain_bank.int8.v1"
    assert raw["task_domain_bank"]["values"].dtype == torch.int8
    assert raw["task_domain_bank"]["values"].shape == (2, 685)
    assert raw["task_domain_bank"]["aggregation_counts"].tolist() == [20, 21]
    assert raw["bank"]["task_keys"][0]["capture_block"] == "capture-a"
    assert raw["task_domain_bank"]["task_keys"][1]["capture_block"] == "capture-b"
    assert set(raw["task_domain_bank"]) == {
        "schema",
        "task_keys",
        "values",
        "scales",
        "zero_points",
        "aggregation_counts",
        "feature_schema",
        "feature_dim",
        "feature_config",
        "aggregation",
    }
    loaded = load_meta_weight_bundle(
        path,
        expected_base_checkpoint_id="base-checkpoint-7",
        base_state=_base_state(),
        expected_block_specs=_expected_block_specs(),
    )
    assert loaded.schema == META_WEIGHT_BUNDLE_SCHEMA
    assert loaded.feature_schema == "marc_ot.support.row.v1"
    assert loaded.feature_dim == 685
    assert loaded.feature_config["psd_bins"] == 16
    assert loaded.task_domain_bank.task_keys == loaded.bank.task_keys
    assert loaded.task_domain_bank.values.dtype == torch.int8
    assert loaded.task_domain_bank.aggregation_counts.tolist() == [20, 21]
    assert loaded.bank.task_keys[0].capture_block == "capture-a"
    assert loaded.task_domain_bank.task_keys[1].capture_block == "capture-b"

    features = torch.zeros(3, 685)
    features[:, :3] = torch.tensor(
        [[1.0, 0.2, -0.1], [0.4, 0.8, 0.3], [-0.2, 0.5, 1.1]]
    )
    labels = torch.tensor([4, 4, 9])
    tokens = ("s0", "s1", "s2")
    mask = torch.ones(3)
    original_support = original_encoder(features, labels, tokens, mask)
    loaded_support = loaded.support_encoder(features, labels, tokens, mask)
    for original, restored in zip(
        (
            original_support.q,
            original_support.uncertainty,
            original_support.block_gates,
            original_support.block_lrs,
        ),
        (
            loaded_support.q,
            loaded_support.uncertainty,
            loaded_support.block_gates,
            loaded_support.block_lrs,
        ),
        strict=True,
    ):
        assert torch.equal(
            original.detach().contiguous().reshape(-1).view(torch.uint8),
            restored.detach().contiguous().reshape(-1).view(torch.uint8),
        )

    original_plan = calibrate_weight_plan(
        _base_state(), "base-checkpoint-7", original_bank, original_support, lr_min=0.002, lr_max=0.02
    )
    loaded_plan = calibrate_weight_plan(
        _base_state(), "base-checkpoint-7", loaded.bank, loaded_support, lr_min=0.002, lr_max=0.02
    )
    assert original_plan.applied and loaded_plan.applied
    for name in original_plan.state_dict:
        assert torch.equal(
            original_plan.state_dict[name].detach().contiguous().reshape(-1).view(torch.uint8),
            loaded_plan.state_dict[name].detach().contiguous().reshape(-1).view(torch.uint8),
        )


@pytest.mark.parametrize("tamper", ["schema", "base", "block_name", "shape", "dtype"])
def test_meta_weight_bundle_rejects_schema_binding_and_block_geometry(
    tmp_path: Path, tamper: str
) -> None:
    """A mismatched deployment bundle must fail closed instead of partially loading."""
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    if tamper == "schema":
        raw["schema"] = "marc_ot_weight_bank_v0"
    elif tamper == "base":
        raw["base_checkpoint_id"] = "other-base"
    elif tamper == "block_name":
        raw["bank"]["entries"][0]["name"] = "classifier"
    elif tamper == "shape":
        raw["bank"]["entries"][0]["shapes"][0] = [99]
    else:
        raw["bank"]["entries"][0]["dtypes"][0] = "torch.float64"
    tampered = tmp_path / f"{tamper}.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


@pytest.mark.parametrize("tamper", ["forbidden_member", "basis_nan", "encoder_inf"])
def test_meta_weight_bundle_rejects_forbidden_or_nonfinite_members(
    tmp_path: Path, tamper: str
) -> None:
    """Forbidden payload members and nonfinite learned state are never repaired."""
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    if tamper == "forbidden_member":
        raw["source_sample_embeddings"] = torch.ones(2, 2)
    elif tamper == "basis_nan":
        raw["bank"]["entries"][0]["basis"][0, 0] = float("nan")
    else:
        first_key = next(iter(raw["support_encoder"]["state_dict"]))
        raw["support_encoder"]["state_dict"][first_key].view(-1)[0] = float("inf")
    tampered = tmp_path / f"{tamper}.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


def test_meta_weight_bundle_rejects_swapped_block_entry_order(tmp_path: Path) -> None:
    """Payload entry order must be bound independently, not accepted as self-authenticating."""
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    raw["bank"]["entries"] = list(reversed(raw["bank"]["entries"]))
    tampered = tmp_path / "swapped-entries.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


def test_meta_weight_bundle_rejects_synchronized_parameter_geometry_reorder(
    tmp_path: Path,
) -> None:
    """Swapping names with their shapes/dtypes must still violate canonical block order."""
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    t3_entry = next(entry for entry in raw["bank"]["entries"] if entry["name"] == "t3")
    assert len(t3_entry["parameter_names"]) == 2
    for key in ("parameter_names", "shapes", "dtypes"):
        t3_entry[key] = list(reversed(t3_entry[key]))
    tampered = tmp_path / "swapped-parameter-order.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


@pytest.mark.parametrize(
    "tamper",
    ("feature_schema", "feature_dim", "feature_config", "legacy160", "missing_feature_binding"),
)
def test_meta_weight_bundle_rejects_old_or_mismatched_support_feature_abi(
    tmp_path: Path, tamper: str
) -> None:
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    if tamper == "feature_schema":
        raw["support_feature"]["schema"] = "marc_ot.support.row.v0"
    elif tamper == "feature_dim":
        raw["support_feature"]["dim"] = 160
    elif tamper == "feature_config":
        raw["support_feature"]["config"]["psd_bins"] = 8
    elif tamper == "legacy160":
        raw["support_encoder"]["config"]["feature_dim"] = 160
    else:
        raw.pop("support_feature")
    tampered = tmp_path / f"{tamper}.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError, match="support feature|member|config"):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


def test_meta_weight_bundle_save_rejects_old_160d_encoder(tmp_path: Path) -> None:
    from cvsrffi.meta_weight_bank_checkpoint import save_meta_weight_bundle

    bank = _bank()
    with pytest.raises(ValueError, match="685|support feature"):
        save_meta_weight_bundle(
            tmp_path / "legacy160.pt",
            base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            bank=bank,
            support_encoder=_encoder(bank, feature_dim=160),
            expected_block_specs=_expected_block_specs(),
            task_domain_descriptors=_task_domain_descriptors(bank),
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "missing",
        "task_order",
        "geometry",
        "dtype",
        "scale",
        "zero_point",
        "count",
        "forbidden_source_sample",
        "duplicate_key",
    ),
)
def test_meta_weight_bundle_rejects_invalid_task_domain_bank(
    tmp_path: Path, tamper: str
) -> None:
    """The Phase2 OT bank must be a complete task-key-bound int8 aggregate."""
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    task_domain = raw["task_domain_bank"]
    if tamper == "missing":
        raw.pop("task_domain_bank")
    elif tamper == "task_order":
        task_domain["task_keys"] = list(reversed(task_domain["task_keys"]))
    elif tamper == "geometry":
        task_domain["values"] = task_domain["values"][:, :-1]
    elif tamper == "dtype":
        task_domain["values"] = task_domain["values"].to(torch.float32)
    elif tamper == "scale":
        task_domain["scales"][0] = float("inf")
    elif tamper == "zero_point":
        task_domain["zero_points"][0] = 1
    elif tamper == "count":
        task_domain["aggregation_counts"][0] = 1
    elif tamper == "forbidden_source_sample":
        task_domain["source_sample_embeddings"] = torch.ones(2, 685)
    else:
        raw["bank"]["task_keys"][1] = dict(raw["bank"]["task_keys"][0])
        task_domain["task_keys"][1] = dict(task_domain["task_keys"][0])
    tampered = tmp_path / f"task-domain-{tamper}.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError, match="task.domain|member|aggregate"):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


@pytest.mark.parametrize(
    ("payload_name", "tamper"),
    (
        ("bank", "missing_capture"),
        ("bank", "extra_capture"),
        ("task_domain_bank", "missing_capture"),
        ("task_domain_bank", "extra_capture"),
        ("task_domain_bank", "changed_capture"),
    ),
)
def test_meta_weight_bundle_fails_closed_on_task_key_capture_payload_drift(
    tmp_path: Path, payload_name: str, tamper: str
) -> None:
    """MARC task-key payloads must retain an exact capture-block binding."""
    from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle

    path = tmp_path / "valid.pt"
    _save_bundle(path)
    raw = _load_raw(path)
    task_key = raw[payload_name]["task_keys"][0]
    if tamper == "missing_capture":
        task_key.pop("capture_block")
    elif tamper == "extra_capture":
        task_key["unexpected"] = "forbidden"
    else:
        task_key["capture_block"] = "forged-capture"
    tampered = tmp_path / f"capture-{payload_name}-{tamper}.pt"
    torch.save(raw, tampered)

    with pytest.raises(ValueError, match="task_keys|task.domain|member"):
        load_meta_weight_bundle(
            tampered,
            expected_base_checkpoint_id="base-checkpoint-7",
            base_state=_base_state(),
            expected_block_specs=_expected_block_specs(),
        )


def test_task_domain_bank_dequantizes_only_frozen_685d_aggregates(tmp_path: Path) -> None:
    from cvsrffi.meta_weight_bank_checkpoint import (
        dequantize_task_domain_features,
        load_meta_weight_bundle,
    )

    path = tmp_path / "valid.pt"
    bank, _encoder_value = _save_bundle(path)
    loaded = load_meta_weight_bundle(
        path,
        expected_base_checkpoint_id="base-checkpoint-7",
        base_state=_base_state(),
        expected_block_specs=_expected_block_specs(),
    )
    restored = dequantize_task_domain_features(loaded.task_domain_bank)

    assert restored.shape == (len(bank.task_keys), 685)
    assert restored.dtype == torch.float32
    assert restored.requires_grad is False
    expected = torch.stack(
        [_task_domain_descriptors(bank)[key].values for key in bank.task_keys]
    )
    assert torch.allclose(restored, expected, atol=0.02, rtol=0.0)
