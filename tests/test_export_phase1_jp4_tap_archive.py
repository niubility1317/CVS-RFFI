from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import export_phase1_jp4_tap_archive as tap


def _arrays() -> dict[str, np.ndarray]:
    count = 6
    generator = np.random.default_rng(7)
    hidden = generator.normal(size=(count, tap.HIDDEN_DIM)).astype(np.float32)
    pre = generator.normal(size=(count, tap.Z_DIM)).astype(np.float32)
    classes = np.asarray([f"c{index}" for index in range(6)], dtype=np.str_)
    return {
        "z_id": np.maximum(pre, 0.0).astype(np.float32),
        "hidden": hidden,
        "pre_relu": pre,
        "joint_weight": generator.normal(
            size=(tap.Z_DIM, tap.HIDDEN_DIM)
        ).astype(np.float32),
        "labels": classes.copy(),
        "receiver_ids": np.asarray(["r0"] * count, dtype=np.str_),
        "day_ids": np.asarray(["d0"] * count, dtype=np.str_),
        "physical_ids": np.asarray([f"p{index}" for index in range(count)]),
        "scenario_names": np.asarray(["leo_clear_weak"] * count),
        "class_ids": classes,
        "observation_ids": np.asarray([f"o{index}" for index in range(count)]),
    }


def test_validate_arrays_accepts_exact_tap_and_rejects_relu_or_identity_drift():
    arrays = _arrays()
    tap._validate_arrays(arrays)

    changed = {name: value.copy() for name, value in arrays.items()}
    changed["z_id"][0, 0] += np.float32(1.0)
    with pytest.raises(tap.Phase1JP4TapArchiveError, match="ReLU binding"):
        tap._validate_arrays(changed)

    duplicate = {name: value.copy() for name, value in arrays.items()}
    duplicate["physical_ids"][1] = duplicate["physical_ids"][0]
    with pytest.raises(tap.Phase1JP4TapArchiveError, match="registry/physical-ID"):
        tap._validate_arrays(duplicate)


def test_forward_taps_preserves_batch_order_and_counts(monkeypatch):
    rows = np.arange(10 * 2 * 8, dtype=np.float32).reshape(10, 2, 8)

    def fake_forward(_model, tensor):
        first = tensor[:, 0, 0].detach().cpu().numpy().astype(np.float32)
        z_id = np.repeat(first[:, None], tap.Z_DIM, axis=1)
        hidden = np.repeat(first[:, None], tap.HIDDEN_DIM, axis=1)
        pre = z_id.copy()
        return SimpleNamespace(z_id=z_id, hidden=hidden, pre_relu=pre)

    monkeypatch.setattr(tap, "strict_zid_with_hook", fake_forward)
    monkeypatch.setattr(
        tap.torch,
        "from_numpy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Torch NumPy C bridge must not be used")
        ),
    )
    z_id, hidden, pre, calls = tap._forward_taps(
        object(), rows, device=torch.device("cpu"), batch_size=4
    )
    expected = rows[:, 0, 0]
    assert calls == 3
    assert np.array_equal(z_id[:, 0], expected)
    assert np.array_equal(hidden[:, 0], expected)
    assert np.array_equal(pre[:, 0], expected)


def test_joint_linear_requires_exact_real_layer_shape():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.joint_proj = torch.nn.Sequential(
                torch.nn.Linear(tap.HIDDEN_DIM, tap.Z_DIM), torch.nn.ReLU()
            )

    model = SimpleNamespace(
        id_backbone=SimpleNamespace(cls_head=Head())
    )
    assert tap._joint_linear(model).weight.shape == (tap.Z_DIM, tap.HIDDEN_DIM)
    model.id_backbone.cls_head.joint_proj[0] = torch.nn.Linear(12, 7)
    with pytest.raises(tap.Phase1JP4TapArchiveError, match="weight contract"):
        tap._joint_linear(model)


def test_exporter_source_has_no_torch_numpy_bridge():
    source = Path(tap.__file__).read_text(encoding="utf-8")
    assert ".numpy(" not in source


def test_bound_file_rejects_hash_drift(tmp_path):
    value = tmp_path / "value.bin"
    value.write_bytes(b"abc")
    with pytest.raises(tap.Phase1JP4TapArchiveError, match="SHA256 drift"):
        tap._regular_bound(value, "0" * 64, "value")


def test_exact_sha_bound_checkpoint_has_audited_torch21_legacy_path(
    monkeypatch, tmp_path
):
    calls = []
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"frozen")

    def fake_load(path, *, map_location, weights_only=False):
        calls.append((Path(path), map_location, weights_only))
        return {"state_dict": {}}

    monkeypatch.delattr(tap.torch.serialization, "safe_globals", raising=False)
    monkeypatch.setattr(tap.torch, "load", fake_load)
    monkeypatch.setattr(tap, "_sha_file", lambda _path: tap.BASE_CHECKPOINT_SHA256)
    checkpoint, audit = tap._load_exact_sha_bound_checkpoint(
        source, tap.BASE_CHECKPOINT_SHA256
    )
    assert checkpoint == {"state_dict": {}}
    assert calls == [(source, "cpu", False)]
    assert audit["policy"] == "legacy_pickle_exact_frozen_sha_only"
    assert audit["safe_globals_available"] is False
    assert audit["weights_only"] is False
    assert audit["caller_selected_checkpoint_allowed"] is False
    assert (
        audit["exact_frozen_checkpoint_sha256_required"]
        == tap.BASE_CHECKPOINT_SHA256
    )


def test_legacy_checkpoint_path_rejects_nonfrozen_sha(monkeypatch, tmp_path):
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"not-frozen")
    monkeypatch.setattr(tap, "_sha_file", lambda _path: "0" * 64)
    with pytest.raises(
        tap.Phase1JP4TapArchiveError, match="exact frozen SHA256"
    ):
        tap._load_exact_sha_bound_checkpoint(source, "0" * 64)
