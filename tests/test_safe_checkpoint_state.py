from __future__ import annotations

import json

import pytest
import torch

from baseline_origin_sat_view import SatViewStage
from cvsrffi.safe_checkpoint_state import (
    SCHEMA,
    SafeCheckpointError,
    _sha_file,
    export_safe_checkpoint,
    verify_safe_checkpoint_receipt,
)


def _source(path):
    model = {
        "bias": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "count": torch.tensor(3, dtype=torch.int64),
    }
    torch.save(
        {
            "model": model,
            "ema_model": {key: value.clone() for key, value in model.items()},
            "sat_schedule": (
                SatViewStage(
                    start_epoch=1,
                    scenarios=("leo_clear_weak",),
                    view_prob=1.0,
                ),
            ),
        },
        path,
    )


def test_export_roundtrip_is_pure_tensor_and_lineage_bound(tmp_path):
    source = tmp_path / "source.pth"
    safe = tmp_path / "safe.pth"
    receipt = tmp_path / "safe.receipt.json"
    _source(source)
    source_sha = _sha_file(source)

    exported = export_safe_checkpoint(
        source_path=source,
        source_sha256=source_sha,
        output_path=safe,
        receipt_path=receipt,
    )
    raw = torch.load(safe, map_location="cpu", weights_only=True)
    assert set(raw) == {
        "schema",
        "source_checkpoint_sha256",
        "state_sha256",
        "model",
        "ema_model",
    }
    assert raw["schema"] == SCHEMA
    assert raw["source_checkpoint_sha256"] == source_sha
    assert exported["weights_only_roundtrip"] is True

    payload, audit = verify_safe_checkpoint_receipt(
        checkpoint_path=safe,
        checkpoint_sha256=_sha_file(safe),
        receipt_path=receipt,
        receipt_sha256=_sha_file(receipt),
        expected_source_sha256=source_sha,
    )
    assert tuple(payload["model"]) == tuple(payload["ema_model"])
    assert audit["tensor_count"] == 4
    assert audit["parameter_count_model_plus_ema"] == 6
    assert audit["weights_only"] is True


def test_safe_checkpoint_receipt_tamper_fails_closed(tmp_path):
    source = tmp_path / "source.pth"
    safe = tmp_path / "safe.pth"
    receipt = tmp_path / "safe.receipt.json"
    _source(source)
    source_sha = _sha_file(source)
    export_safe_checkpoint(
        source_path=source,
        source_sha256=source_sha,
        output_path=safe,
        receipt_path=receipt,
    )
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["tensor_count"] += 1
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SafeCheckpointError, match="receipt content"):
        verify_safe_checkpoint_receipt(
            checkpoint_path=safe,
            checkpoint_sha256=_sha_file(safe),
            receipt_path=receipt,
            receipt_sha256=_sha_file(receipt),
            expected_source_sha256=source_sha,
        )


def test_export_refuses_wrong_source_sha(tmp_path):
    source = tmp_path / "source.pth"
    _source(source)
    with pytest.raises(SafeCheckpointError, match="source checkpoint SHA256 drift"):
        export_safe_checkpoint(
            source_path=source,
            source_sha256="0" * 64,
            output_path=tmp_path / "safe.pth",
            receipt_path=tmp_path / "safe.receipt.json",
        )
