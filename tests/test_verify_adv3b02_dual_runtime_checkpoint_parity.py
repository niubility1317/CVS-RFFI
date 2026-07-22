from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle
from scripts import export_adv3b02_dual_feature_torchscript as dual_export
from scripts import verify_adv3b02_dual_runtime_checkpoint_parity as parity


class _Backbone(nn.Module):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def forward(self, rows, y=None, return_aux=True, domain_labels=None):
        del y, return_aux, domain_labels
        if self.key == "feat_joint":
            feature = rows[:, 0, :160]
            logits = (rows[:, 0, :160] + rows[:, 1, :160])[:, :3]
            return {self.key: feature, "logits": logits}
        return {self.key: rows[:, 1, :160]}


class _Enhancer(nn.Module):
    def forward(self, feature, rows):
        del rows
        return feature, None


class _TinyDualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_feature_key = "feat_joint"
        self.dom_feature_key = "feat_imp"
        self.id_backbone = _Backbone("feat_joint")
        self.dom_backbone = _Backbone("feat_imp")
        self.dom_enhancer = _Enhancer()

    @staticmethod
    def _pick_z_id(aux):
        return aux["feat_joint"]

    @staticmethod
    def _pick_z_dom(aux):
        return aux["feat_imp"]


class _DifferentRuntime(nn.Module):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def forward(self, rows: torch.Tensor):
        z_id = rows[:, 0, :160]
        z_dom = rows[:, 1, :160]
        logits = (z_id + z_dom)[:, :3]
        if self.field == "z_id":
            z_id = z_id + 0.01
        elif self.field == "z_dom":
            z_dom = z_dom + 0.01
        else:
            logits = logits + 0.01
        return z_id, z_dom, logits


class _DormantForbidden(nn.Module):
    def __init__(self, runtime: nn.Module) -> None:
        super().__init__()
        self.runtime = runtime
        self.dom_head = nn.Linear(1, 1)

    def forward(self, rows: torch.Tensor):
        return self.runtime(rows)


def _sha(path: Path) -> str:
    return parity._sha256_file(path)


def _write_export_receipt(
    path: Path,
    *,
    checkpoint: Path,
    adapter: Path,
    runtime: Path,
) -> None:
    execution_contract = dual_export._seal_graph_executor_optimize_false(torch.device("cpu"))
    payload = {
        "schema": dual_export.EXPORT_SCHEMA,
        "status": "PASS",
        "runtime_output_schema": dual_export.RUNTIME_OUTPUT_SCHEMA,
        "feature_keys": {"z_id": "feat_joint", "z_dom": "feat_imp"},
        "checkpoint_sha256": _sha(checkpoint),
        "adapter_state_sha256": _sha(adapter),
        "base_runtime_sha256": _sha(runtime),
        "candidate_runtime_sha256": _sha(runtime),
        "expected_input_len": 160,
        "runtime_batch_capacity": 256,
        "feature_dimensions": {"z_id": 160, "z_dom": 160, "tx_logits": 3},
        "runtime_component_allowlist": list(
            dual_export.RUNTIME_COMPONENT_ALLOWLIST
        ),
        "forbidden_runtime_tokens": list(
            dual_export.FORBIDDEN_RUNTIME_TOKENS
        ),
        "effective8_target_modules": list(dual_export.EFFECTIVE8_TARGET_MODULES),
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_contract["contract_sha256"],
        "max_abs_tolerance": 1.0e-5,
        "formal_phase2_eligible": False,
        "bundle_created": False,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8"
    )


def _assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint.pth"
    adapter = tmp_path / "adapter.pth"
    torch.save({"diagnostic": True}, checkpoint)
    torch.save({"unused": torch.zeros(1)}, adapter)
    runtime = tmp_path / "dual_runtime.ts"
    wrapper = dual_export.ADV3B02DualFeatureRuntime(
        _TinyDualModel(),
        expected_input_len=160,
        expected_tx_classes=3,
        runtime_batch_size=256,
    ).eval()
    dual_export._trace_and_save(wrapper, torch.randn(2, 2, 160), runtime)
    export_receipt = tmp_path / "export.json"
    _write_export_receipt(
        export_receipt,
        checkpoint=checkpoint,
        adapter=adapter,
        runtime=runtime,
    )
    return checkpoint, adapter, runtime, export_receipt


def _patch_candidate(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> None:
    monkeypatch.setattr(parity, "BASE_CHECKPOINT_SHA256", _sha(checkpoint))
    monkeypatch.setattr(parity, "_load_checkpoint_bytes", lambda _value: {})
    monkeypatch.setattr(
        parity,
        "build_exact_ssdg_model_from_checkpoint",
        lambda *args, **kwargs: (_TinyDualModel(), {"strict": True}),
    )
    monkeypatch.setattr(
        parity,
        "apply_fp16_lora_state",
        lambda *args, **kwargs: {
            "element_count": 44_048,
            "tensor_bytes_fp16": 88_096,
            "target_modules": list(parity.EFFECTIVE8_TARGET_MODULES),
        },
    )
    monkeypatch.setattr(
        parity,
        "merge_feat_joint_lora",
        lambda _model: {
            "merged_module_count": 8,
            "remaining_lora_wrappers": [],
            "algebraic_probe_parity_pass": True,
        },
    )


def _verify(
    *,
    checkpoint: Path,
    adapter: Path,
    runtime: Path,
    export_receipt: Path,
    receipt: Path,
    vectors: Path,
    expected_export_sha: str | None = None,
):
    return parity.verify_dual_runtime_checkpoint(
        checkpoint_path=checkpoint,
        adapter_state_path=adapter,
        runtime_path=runtime,
        export_receipt_path=export_receipt,
        expected_export_receipt_sha256=(
            _sha(export_receipt)
            if expected_export_sha is None
            else expected_export_sha
        ),
        runtime_role="candidate",
        receipt_out=receipt,
        vector_audit_out=vectors,
        input_len=160,
        parity_seed=5,
        parity_rows=8,
        device="cpu",
    )


def test_candidate_parity_binds_export_and_closes_three_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, adapter, runtime, export_receipt = _assets(tmp_path)
    _patch_candidate(monkeypatch, checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    adapter_bytes = adapter.read_bytes()
    original_read = parity._read_regular_bytes
    counts = {
        "checkpoint": 0,
        "adapter state": 0,
        "runtime": 0,
        "export receipt": 0,
    }

    def _counted_read(path: Path, name: str) -> bytes:
        counts[name] += 1
        return original_read(path, name)

    def _checkpoint_snapshot(value: bytes):
        checkpoint.write_bytes(b"swapped-checkpoint")
        try:
            assert value == checkpoint_bytes
            return {}
        finally:
            checkpoint.write_bytes(checkpoint_bytes)

    def _adapter_snapshot(value: bytes):
        adapter.write_bytes(b"swapped-adapter")
        try:
            assert value == adapter_bytes
            return {"unused": torch.zeros(1)}
        finally:
            adapter.write_bytes(adapter_bytes)

    monkeypatch.setattr(parity, "_read_regular_bytes", _counted_read)
    monkeypatch.setattr(parity, "_load_checkpoint_bytes", _checkpoint_snapshot)
    monkeypatch.setattr(parity, "_load_tensor_state_bytes", _adapter_snapshot)
    receipt = tmp_path / "receipt.json"
    vectors = tmp_path / "vectors.json"
    result = _verify(
        checkpoint=checkpoint,
        adapter=adapter,
        runtime=runtime,
        export_receipt=export_receipt,
        receipt=receipt,
        vectors=vectors,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    vector_payload = json.loads(vectors.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["batch_sizes"] == [1, 8, 256]
    assert result["max_abs_output_delta"] == 0.0
    assert payload["expected_input_len"] == 160
    assert payload["expected_tx_classes"] == 3
    assert payload["runtime_batch_capacity"] == 256
    assert payload["runtime_component_audit"][
        "forbidden_runtime_components_absent"
    ] is True
    assert payload["export_receipt_sha256"] == _sha(export_receipt)
    assert payload["schema"] == parity.RECEIPT_SCHEMA
    assert payload["execution_contract_sha256"] == payload["execution_contract"]["contract_sha256"]
    assert payload["runtime_invocations_per_parity_batch"] == 3
    assert payload["runtime_calls_per_batch"] == 3
    assert counts == {
        "checkpoint": 1,
        "adapter state": 1,
        "runtime": 1,
        "export receipt": 1,
    }
    assert [row["batch_size"] for row in vector_payload["rows"]] == [1, 8, 256]
    assert all(len(row["runtime_calls"]) == 3 for row in vector_payload["rows"])
    preimage = {
        key: value
        for key, value in vector_payload.items()
        if key not in {"parity_vector_root_sha256", "authority_scope"}
    }
    assert deployment_bundle.sha256_bytes(
        deployment_bundle.canonical_json_bytes(preimage)
    ) == payload["parity_vector_root_sha256"]


@pytest.mark.parametrize("field", ["z_id", "z_dom", "tx_logits"])
def test_candidate_parity_detects_each_numeric_mismatch_before_writes(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, adapter, runtime, export_receipt = _assets(tmp_path)
    _patch_candidate(monkeypatch, checkpoint)
    monkeypatch.setattr(
        parity,
        "ADV3B02DualFeatureRuntime",
        lambda *args, **kwargs: _DifferentRuntime(field),
    )
    receipt = tmp_path / "failed.json"
    vectors = tmp_path / "failed_vectors.json"
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="exceeds tolerance"):
        _verify(
            checkpoint=checkpoint,
            adapter=adapter,
            runtime=runtime,
            export_receipt=export_receipt,
            receipt=receipt,
            vectors=vectors,
        )
    assert not receipt.exists()
    assert not vectors.exists()


@pytest.mark.parametrize(
    "bad_field", ["shape", "dtype", "nonfinite", "tx_width", "arity"]
)
def test_runtime_output_contract_rejects_drift(bad_field: str) -> None:
    rows = 2
    z_id = torch.zeros(rows, 160)
    z_dom = torch.zeros(rows, 160)
    logits = torch.zeros(rows, 3)
    value = (z_id, z_dom, logits)
    if bad_field == "shape":
        value = (z_id[:, :159], z_dom, logits)
    elif bad_field == "dtype":
        value = (z_id, z_dom.double(), logits)
    elif bad_field == "nonfinite":
        z_dom[0, 0] = float("inf")
    elif bad_field == "tx_width":
        value = (z_id, z_dom, torch.zeros(rows, 4))
    else:
        value = (z_id, logits)
    with pytest.raises(parity.ADV3B02DualRuntimeParityError):
        parity._runtime_outputs(value, rows=rows, expected_tx_classes=3)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema", "cvs.phase1.adv3b02_dual_feature_torchscript_export.v1"),
        ("candidate_runtime_sha256", "0" * 64),
        ("feature_keys", {"z_id": "feat", "z_dom": "feat_imp"}),
        ("checkpoint_sha256", "0" * 64),
        ("adapter_state_sha256", "0" * 64),
        ("runtime_component_allowlist", ["runtime.id_backbone"]),
        ("forbidden_runtime_tokens", []),
        ("effective8_target_modules", ["wrong"]),
    ],
)
def test_verifier_rejects_export_binding_drift(
    field: str,
    replacement,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, adapter, runtime, export_receipt = _assets(tmp_path)
    _patch_candidate(monkeypatch, checkpoint)
    payload = json.loads(export_receipt.read_text(encoding="utf-8"))
    payload[field] = replacement
    export_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="binding drift"):
        _verify(
            checkpoint=checkpoint,
            adapter=adapter,
            runtime=runtime,
            export_receipt=export_receipt,
            receipt=tmp_path / "none.json",
            vectors=tmp_path / "none_vectors.json",
        )


def test_verifier_rejects_expected_export_receipt_sha_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, adapter, runtime, export_receipt = _assets(tmp_path)
    _patch_candidate(monkeypatch, checkpoint)
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="SHA256"):
        _verify(
            checkpoint=checkpoint,
            adapter=adapter,
            runtime=runtime,
            export_receipt=export_receipt,
            expected_export_sha="0" * 64,
            receipt=tmp_path / "none.json",
            vectors=tmp_path / "none_vectors.json",
        )


def test_verifier_rejects_dormant_forbidden_module_even_with_matching_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, adapter, runtime, export_receipt = _assets(tmp_path)
    _patch_candidate(monkeypatch, checkpoint)
    inner = torch.jit.load(str(runtime)).eval()
    malicious = torch.jit.script(_DormantForbidden(inner).eval())
    torch.jit.save(malicious, runtime)
    _write_export_receipt(
        export_receipt,
        checkpoint=checkpoint,
        adapter=adapter,
        runtime=runtime,
    )
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="forbidden"):
        _verify(
            checkpoint=checkpoint,
            adapter=adapter,
            runtime=runtime,
            export_receipt=export_receipt,
            receipt=tmp_path / "none.json",
            vectors=tmp_path / "none_vectors.json",
        )


def test_verifier_rejects_checkpoint_cuda_and_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, adapter, runtime, export_receipt = _assets(tmp_path)
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="strict ADV3B02"):
        _verify(
            checkpoint=checkpoint,
            adapter=adapter,
            runtime=runtime,
            export_receipt=export_receipt,
            receipt=tmp_path / "none.json",
            vectors=tmp_path / "none_vectors.json",
        )

    _patch_candidate(monkeypatch, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="fallback is forbidden"):
        parity.verify_dual_runtime_checkpoint(
            checkpoint_path=checkpoint,
            adapter_state_path=adapter,
            runtime_path=runtime,
            export_receipt_path=export_receipt,
            expected_export_receipt_sha256=_sha(export_receipt),
            runtime_role="candidate",
            receipt_out=tmp_path / "cuda.json",
            vector_audit_out=tmp_path / "cuda_vectors.json",
            input_len=160,
            parity_seed=1,
            parity_rows=8,
            device="cuda:0",
        )

    receipt = tmp_path / "exists.json"
    receipt.write_text("occupied", encoding="utf-8")
    with pytest.raises(parity.ADV3B02DualRuntimeParityError, match="overwrite"):
        _verify(
            checkpoint=checkpoint,
            adapter=adapter,
            runtime=runtime,
            export_receipt=export_receipt,
            receipt=receipt,
            vectors=tmp_path / "exists_vectors.json",
        )
