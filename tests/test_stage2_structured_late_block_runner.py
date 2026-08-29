from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn


_SUT = importlib.import_module("cvsrffi.stage2_structured_late_block_runner")


class _FakeFrozenCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_weight = nn.Parameter(torch.ones(4, 4))


def _write_row_inputs(tmp_path: Path) -> dict[str, Path]:
    support_path = tmp_path / "support.npz"
    query_path = tmp_path / "query.npz"
    prototype_path = tmp_path / "frozen_prototypes.npz"
    checkpoint_path = tmp_path / "ADV3B02_CORE90_SOFT_E200.pth"
    checkpoint_path.write_bytes(b"fake-checkpoint-loaded-only-through-monkeypatch")
    np.savez(
        support_path,
        received_iq=np.asarray(
            [
                [[1.0, 0.8, 0.6], [0.1, 0.0, 0.2]],
                [[0.9, 0.7, 0.5], [0.0, 0.2, 0.1]],
                [[0.1, 0.0, 0.2], [1.0, 0.8, 0.6]],
                [[0.0, 0.2, 0.1], [0.9, 0.7, 0.5]],
            ],
            dtype=np.float32,
        ),
        support_labels=np.asarray([10, 10, 20, 20], dtype=np.int64),
    )
    np.savez(
        query_path,
        received_iq=np.asarray(
            [
                [[0.8, 0.7, 0.5], [0.1, 0.0, 0.2]],
                [[0.1, 0.2, 0.0], [0.8, 0.7, 0.5]],
            ],
            dtype=np.float32,
        ),
        query_ids=np.asarray(["query-fixed-001", "query-fixed-002"]),
    )
    np.savez(
        prototype_path,
        prototypes=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        class_ids=np.asarray([10, 20], dtype=np.int64),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "support_path": support_path,
        "query_path": query_path,
        "prototype_path": prototype_path,
    }


def _row_config(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed-received-iq",
        "split_id": "split-support-query-disjoint",
        "row_id": "target5-rx20-1-k2-seed392002",
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "seed": 392002,
        "k_shot": 2,
        "checkpoint_path": str(paths["checkpoint_path"]),
        "support_path": str(paths["support_path"]),
        "query_path": str(paths["query_path"]),
        "prototype_path": str(paths["prototype_path"]),
        "candidate": "freq_f3_proj",
        "steps": 2,
        "learning_rate": 0.02,
        "decision_rule": "frozen_prototype_cosine_v1",
    }


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    paths: dict[str, Path],
) -> tuple[list[str], _FakeFrozenCheckpoint]:
    events: list[str] = []
    model = _FakeFrozenCheckpoint()
    real_load_npz = _SUT._load_npz

    def fake_load_checkpoint(*args, **kwargs):
        events.append("load_checkpoint")
        return model

    def traced_load_npz(path, *args, **kwargs):
        resolved = Path(path)
        if resolved == paths["support_path"]:
            events.append("open_support")
        elif resolved == paths["prototype_path"]:
            events.append("open_prototypes")
        elif resolved == paths["query_path"]:
            assert "adapt" in events
            assert model.training is False
            assert all(not parameter.requires_grad for parameter in model.parameters())
            events.append("open_query")
        return real_load_npz(path, *args, **kwargs)

    def fake_adapt(
        adapted_model,
        support_iq,
        support_labels,
        *,
        frozen_prototypes,
        prototype_class_ids,
        context,
        config,
    ):
        assert adapted_model is model
        assert "open_support" in events
        assert "open_query" not in events
        assert support_iq.shape == (4, 2, 3)
        assert support_labels.tolist() == [10, 10, 20, 20]
        assert frozen_prototypes.ndim == 2
        assert frozen_prototypes.shape == (2, 4)
        assert frozen_prototypes.requires_grad is False
        assert prototype_class_ids.ndim == 1
        assert prototype_class_ids.tolist() == [10, 20]
        assert context == {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "capsule-fixed-received-iq",
            "split_id": "split-support-query-disjoint",
        }
        assert config.candidate == "freq_f3_proj"
        assert config.steps == 2
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        events.append("adapt")
        return {
            "candidate": "freq_f3_proj",
            "gradient_updates": 2,
            "trainable_fraction": 0.08,
        }

    def fake_predict(
        predicted_model,
        received_iq,
        *,
        frozen_prototypes,
        prototype_class_ids,
    ):
        assert predicted_model is model
        assert events[-1] == "open_query"
        assert model.training is False
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert received_iq.shape == (2, 2, 3)
        assert frozen_prototypes.ndim == 2
        assert frozen_prototypes.requires_grad is False
        assert prototype_class_ids.tolist() == [10, 20]
        events.append("predict")
        return (
            torch.tensor([10, 20], dtype=torch.long),
            torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
        )

    monkeypatch.setattr(_SUT, "_load_frozen_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(_SUT, "_load_npz", traced_load_npz)
    monkeypatch.setattr(
        _SUT,
        "adapt_on_target_support_with_frozen_prototypes",
        fake_adapt,
    )
    monkeypatch.setattr(
        _SUT,
        "predict_query_with_frozen_prototypes",
        fake_predict,
    )
    return events, model


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "source_path",
        "source_cache_path",
        "source_features_path",
        "clean_path",
        "query_truth_path",
        "query_labels_path",
    ],
)
def test_runner_config_is_an_exhaustive_allowlist(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    paths = _write_row_inputs(tmp_path)
    config = _row_config(paths)
    config[forbidden_key] = str(tmp_path / "forbidden")

    with pytest.raises(ValueError, match="allowlist"):
        _SUT.run_stage2_row(config, output_dir=tmp_path / "output", device="cpu")
    assert not (tmp_path / "output").exists()


def test_runner_opens_query_only_after_support_adaptation_is_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_row_inputs(tmp_path)
    events, model = _install_fake_execution(monkeypatch, paths=paths)

    receipt = _SUT.run_stage2_row(
        _row_config(paths),
        output_dir=tmp_path / "output",
        device="cpu",
    )

    assert events.index("open_support") < events.index("adapt")
    assert events.index("adapt") < events.index("open_query")
    assert events.index("open_query") < events.index("predict")
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert receipt["query_truth_opened"] is False
    assert receipt["gradient_updates"] == 2


def test_prediction_is_not_published_when_npz_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a half-written final prediction being mistaken for completion."""

    paths = _write_row_inputs(tmp_path)
    _install_fake_execution(monkeypatch, paths=paths)

    def interrupted_save(target, **_payload):
        if hasattr(target, "write"):
            target.write(b"partial")
        else:
            Path(target).write_bytes(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(_SUT.np, "savez", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        _SUT.run_stage2_row(
            _row_config(paths),
            output_dir=tmp_path / "output",
            device="cpu",
        )
    assert not (tmp_path / "output" / "predictions.npz").exists()


def test_prediction_artifact_contains_no_truth_or_query_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_row_inputs(tmp_path)
    _install_fake_execution(monkeypatch, paths=paths)

    receipt = _SUT.run_stage2_row(
        _row_config(paths),
        output_dir=tmp_path / "output",
        device="cpu",
    )

    prediction_path = Path(receipt["prediction_path"])
    assert prediction_path == tmp_path / "output" / "predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "query_ids",
            "predicted_class_ids",
            "scores",
        }
        np.testing.assert_array_equal(
            artifact["query_ids"],
            np.asarray(["query-fixed-001", "query-fixed-002"]),
        )
        np.testing.assert_array_equal(
            artifact["predicted_class_ids"], np.asarray([10, 20])
        )
        assert artifact["scores"].shape == (2, 2)
        assert not {
            "truth",
            "query_truth",
            "query_labels",
            "query_role",
        } & set(artifact.files)


def test_prototype_payload_rejects_covariance_or_nonmatrix_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_row_inputs(tmp_path)
    monkeypatch.setattr(
        _SUT,
        "_load_frozen_checkpoint",
        lambda *args, **kwargs: _FakeFrozenCheckpoint(),
    )
    np.savez(
        paths["prototype_path"],
        prototypes=np.eye(2, 4, dtype=np.float32),
        class_ids=np.asarray([10, 20], dtype=np.int64),
        covariance=np.stack([np.eye(4), np.eye(4)]).astype(np.float32),
    )
    with pytest.raises(ValueError, match="prototype.*allowlist"):
        _SUT.run_stage2_row(
            _row_config(paths), output_dir=tmp_path / "extra", device="cpu"
        )

    np.savez(
        paths["prototype_path"],
        prototypes=np.zeros((2, 2, 4), dtype=np.float32),
        class_ids=np.asarray([10, 20], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="2D"):
        _SUT.run_stage2_row(
            _row_config(paths), output_dir=tmp_path / "rank", device="cpu"
        )


def test_forbidden_npz_member_is_rejected_before_value_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_row_inputs(tmp_path)
    monkeypatch.setattr(
        _SUT,
        "_load_frozen_checkpoint",
        lambda *args, **kwargs: _FakeFrozenCheckpoint(),
    )
    np.savez(
        paths["support_path"],
        received_iq=np.zeros((4, 2, 3), dtype=np.float32),
        support_labels=np.asarray([10, 10, 20, 20], dtype=np.int64),
        source_cache=np.asarray([object()], dtype=object),
    )
    with pytest.raises(ValueError, match="support payload allowlist"):
        _SUT.run_stage2_row(
            _row_config(paths), output_dir=tmp_path / "forbidden", device="cpu"
        )


def test_runner_enforces_exact_k_per_support_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_row_inputs(tmp_path)
    _install_fake_execution(monkeypatch, paths=paths)
    config = _row_config(paths)
    config["k_shot"] = 1
    with pytest.raises(ValueError, match="K-shot"):
        _SUT.run_stage2_row(
            config, output_dir=tmp_path / "wrong-k", device="cpu"
        )


def test_runner_and_adaptation_surface_have_no_source_or_query_truth_inputs() -> None:
    runner_parameters = set(inspect.signature(_SUT.run_stage2_row).parameters)
    adaptation_parameters = set(
        inspect.signature(
            _SUT.adapt_on_target_support_with_frozen_prototypes
        ).parameters
    )
    forbidden = {
        "source",
        "source_samples",
        "source_features",
        "source_cache",
        "clean_samples",
        "query_labels",
        "query_truth",
        "query_role",
    }
    assert not forbidden & runner_parameters
    assert not forbidden & adaptation_parameters
