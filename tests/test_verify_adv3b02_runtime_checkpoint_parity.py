from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle
from scripts import export_phase1_singleobs_feature_archive as exporter
from scripts import verify_adv3b02_runtime_checkpoint_parity as parity


class TinyRuntime(nn.Module):
    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = rows[:, 0, :160]
        logits = feature[:, :3]
        return feature, logits


def _assets(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"diagnostic-checkpoint")
    runtime = tmp_path / "runtime.ts"
    traced = torch.jit.trace(TinyRuntime(), torch.randn(2, 2, 160))
    torch.jit.save(traced, runtime)
    return checkpoint, runtime


def _patch_checkpoint(monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
    monkeypatch.setattr(parity, "BASE_CHECKPOINT_SHA256", parity._sha256_file(checkpoint))
    monkeypatch.setattr(parity, "_load_checkpoint_bytes", lambda value: {})
    monkeypatch.setattr(
        parity,
        "build_exact_ssdg_model_from_checkpoint",
        lambda *args, **kwargs: (TinyRuntime(), {"diagnostic": True}),
    )
    monkeypatch.setattr(parity, "ADV3B02IdentityRuntime", lambda model: model)


def test_fresh_numerical_parity_writes_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime = _assets(tmp_path)
    _patch_checkpoint(monkeypatch, checkpoint)
    output = tmp_path / "receipt.json"
    vector_output = tmp_path / "vectors.json"
    result = parity.verify_runtime_checkpoint(
        checkpoint_path=checkpoint,
        runtime_path=runtime,
        receipt_out=output,
        vector_audit_out=vector_output,
        input_len=160,
        parity_seed=7,
        parity_rows=2,
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    vectors = json.loads(vector_output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["max_abs_output_delta"] == 0.0
    assert result["resolved_device"] == "cpu"
    assert result["batch_sizes"] == [1, 2, 256]
    assert [row["batch_size"] for row in vectors["rows"]] == [1, 2, 256]
    assert vectors["parity_vector_root_sha256"] == payload["parity_vector_root_sha256"]
    vector_preimage = {
        key: value
        for key, value in vectors.items()
        if key not in {"parity_vector_root_sha256", "authority_scope"}
    }
    assert deployment_bundle.sha256_bytes(
        deployment_bundle.canonical_json_bytes(vector_preimage)
    ) == payload["parity_vector_root_sha256"]
    assert payload["schema"] == parity.RECEIPT_SCHEMA
    assert set(payload) == {
        "schema",
        "checkpoint_lineage_sha256",
        "runtime_sha256",
        "parity_status",
        "max_abs_output_delta",
        "parity_vector_root_sha256",
        "runtime_archive_member_root_sha256",
        "runtime_state_schema_root_sha256",
        "runtime_state_bytes",
        "runtime_structure_sha256",
    }
    with pytest.raises(parity.ADV3B02RuntimeParityError, match="overwrite"):
        parity.verify_runtime_checkpoint(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            receipt_out=output,
            vector_audit_out=vector_output,
            input_len=160,
            parity_seed=7,
            parity_rows=2,
            device="cpu",
        )


def test_parity_failure_does_not_write_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime = _assets(tmp_path)
    _patch_checkpoint(monkeypatch, checkpoint)

    class Different(TinyRuntime):
        def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            feature, logits = super().forward(rows)
            return feature + 0.01, logits

    monkeypatch.setattr(
        parity,
        "build_exact_ssdg_model_from_checkpoint",
        lambda *args, **kwargs: (Different(), {}),
    )
    output = tmp_path / "failed.json"
    vector_output = tmp_path / "failed_vectors.json"
    with pytest.raises(parity.ADV3B02RuntimeParityError, match="exceeds tolerance"):
        parity.verify_runtime_checkpoint(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            receipt_out=output,
            vector_audit_out=vector_output,
            input_len=160,
            parity_seed=9,
            parity_rows=2,
            device="cpu",
        )
    assert not output.exists()
    assert not vector_output.exists()


def test_checkpoint_and_limits_fail_closed(tmp_path: Path) -> None:
    checkpoint, runtime = _assets(tmp_path)
    with pytest.raises(parity.ADV3B02RuntimeParityError, match="strict ADV3B02"):
        parity.verify_runtime_checkpoint(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            receipt_out=tmp_path / "none.json",
            vector_audit_out=tmp_path / "none_vectors.json",
            input_len=160,
            parity_seed=1,
            parity_rows=2,
            device="cpu",
        )
    with pytest.raises(parity.ADV3B02RuntimeParityError, match="parity_rows"):
        parity.verify_runtime_checkpoint(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            receipt_out=tmp_path / "none2.json",
            vector_audit_out=tmp_path / "none2_vectors.json",
            input_len=160,
            parity_seed=1,
            parity_rows=257,
            device="cpu",
        )


def test_requested_cuda_never_silently_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime = _assets(tmp_path)
    _patch_checkpoint(monkeypatch, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(parity.ADV3B02RuntimeParityError, match="fallback is forbidden"):
        parity.verify_runtime_checkpoint(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            receipt_out=tmp_path / "none.json",
            vector_audit_out=tmp_path / "none_vectors.json",
            input_len=160,
            parity_seed=7,
            parity_rows=8,
            device="cuda:0",
        )


def test_input_files_cannot_drift_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime = _assets(tmp_path)
    _patch_checkpoint(monkeypatch, checkpoint)
    original_builder = parity.build_exact_ssdg_model_from_checkpoint

    def drifting_builder(*args, **kwargs):
        runtime.write_bytes(runtime.read_bytes() + b"drift")
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        parity, "build_exact_ssdg_model_from_checkpoint", drifting_builder
    )
    output = tmp_path / "none.json"
    vector_output = tmp_path / "none_vectors.json"
    with pytest.raises(parity.ADV3B02RuntimeParityError, match="changed during"):
        parity.verify_runtime_checkpoint(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            receipt_out=output,
            vector_audit_out=vector_output,
            input_len=160,
            parity_seed=7,
            parity_rows=8,
            device="cpu",
        )
    assert not output.exists()
    assert not vector_output.exists()


def test_receipt_is_accepted_by_exact_development_runtime_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime = _assets(tmp_path)
    _patch_checkpoint(monkeypatch, checkpoint)
    receipt = tmp_path / "receipt.json"
    vectors = tmp_path / "vectors.json"
    parity.verify_runtime_checkpoint(
        checkpoint_path=checkpoint,
        runtime_path=runtime,
        receipt_out=receipt,
        vector_audit_out=vectors,
        input_len=160,
        parity_seed=7,
        parity_rows=8,
        device="cpu",
    )
    checkpoint_sha = parity._sha256_file(checkpoint)
    runtime_sha = parity._sha256_file(runtime)
    receipt_sha = parity._sha256_file(receipt)
    monkeypatch.setattr(exporter, "BASE_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(
        exporter, "KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256", (runtime_sha,)
    )
    manifest = {
        "schema": exporter.RUNTIME_MANIFEST_SCHEMA,
        "artifact_stage": "phase1_offline_before_target_access",
        "bundle_id": "a" * 64,
        "phase1_checkpoint_sha256": checkpoint_sha,
        "feature_runtime": {
            "path": runtime.name,
            "sha256": runtime_sha,
            "schema": exporter.RUNTIME_SCHEMA,
        },
        "runtime_export_receipt": {
            "path": receipt.name,
            "sha256": receipt_sha,
            "schema": exporter.RUNTIME_EXPORT_RECEIPT_SCHEMA,
        },
        "feature_dims": {
            "input_channels": 2,
            "z160": 160,
            "checkpoint_reference_logits": 3,
            "features": 288,
        },
        "class_ids": ["c0", "c1", "c2"],
    }
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    bound = exporter._load_runtime_binding(
        manifest_path,
        parity._sha256_file(manifest_path),
        require_known_development_runtime=True,
        expected_runtime_sha256=runtime_sha,
        expected_parity_receipt_sha256=receipt_sha,
    )
    assert bound["runtime_sha256"] == runtime_sha
    assert bound["runtime_receipt_sha256"] == receipt_sha
    assert bound["authority_mode"] == "development_known_adv3b02_runtime_sha"
