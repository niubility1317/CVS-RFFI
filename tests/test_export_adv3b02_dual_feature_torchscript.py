from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

from model_dual_cvsincnet import DualCVSincNetDisentangle


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import export_adv3b02_dual_feature_torchscript as dual_export  # noqa: E402
import verify_adv3b02_dual_runtime_checkpoint_parity as dual_verify  # noqa: E402


class _Backbone(torch.nn.Module):
    def __init__(self, key: str, offset: float) -> None:
        super().__init__()
        self.key = key
        self.offset = float(offset)

    def forward(self, rows, y=None, return_aux=True, domain_labels=None):
        del y, return_aux, domain_labels
        feature = rows.mean(dim=1)[:, :160] + self.offset
        return {self.key: feature, "logits": feature[:, :3]}


class _Enhancer(torch.nn.Module):
    def forward(self, feature, rows):
        return feature + rows.std(dim=1, unbiased=False)[:, :160], None


class _Bomb(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden_marker = torch.nn.Parameter(torch.ones(1))

    def forward(self, _rows):
        raise AssertionError("dom_head must not execute")


class _TinyDual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_feature_key = "feat_joint"
        self.dom_feature_key = "feat_imp"
        self.id_backbone = _Backbone("feat_joint", 1.0)
        self.dom_backbone = _Backbone("feat_imp", 2.0)
        self.dom_enhancer = _Enhancer()
        self.dom_head = _Bomb()

    @staticmethod
    def _pick_z_id(aux):
        return aux["feat_joint"]

    @staticmethod
    def _pick_z_dom(aux):
        return aux["feat_imp"]


def test_dual_runtime_trace_preserves_all_three_outputs(tmp_path: Path) -> None:
    wrapper = dual_export.ADV3B02DualFeatureRuntime(
        _TinyDual(),
        expected_input_len=192,
        expected_tx_classes=3,
        runtime_batch_size=256,
    ).eval()
    rows = torch.randn(4, 2, 192)
    runtime = dual_export._trace_and_save(
        wrapper, rows[:2], tmp_path / "dual.ts"
    )
    eager = wrapper(rows)
    loaded = runtime(rows)
    assert len(eager) == len(loaded) == 3
    for actual, expected in zip(loaded, eager):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert all("dom_head" not in name for name, _ in runtime.named_modules())
    assert all("dom_head" not in name for name, _ in runtime.named_parameters())
    with pytest.raises((RuntimeError, torch.jit.Error), match="float32"):
        runtime(rows.double())
    with pytest.raises((RuntimeError, torch.jit.Error), match="shape"):
        runtime(rows[:, :1])
    with pytest.raises((RuntimeError, torch.jit.Error), match="length"):
        runtime(rows[:, :, :-1])
    with pytest.raises((RuntimeError, torch.jit.Error), match="length"):
        runtime(torch.randn(4, 2, 193))
    with pytest.raises((RuntimeError, torch.jit.Error), match="capacity"):
        runtime(torch.randn(257, 2, 192))
    nonfinite = rows.clone()
    nonfinite[0, 0, 0] = torch.nan
    with pytest.raises((RuntimeError, torch.jit.Error), match="finite"):
        runtime(nonfinite)


def test_dual_runtime_rejects_non_float32_and_batch_overflow() -> None:
    wrapper = dual_export.ADV3B02DualFeatureRuntime(
        _TinyDual(),
        expected_input_len=160,
        expected_tx_classes=3,
        runtime_batch_size=256,
    ).eval()
    with pytest.raises(RuntimeError, match="float32"):
        wrapper(torch.randn(2, 2, 160).double())
    with pytest.raises(RuntimeError, match="batch"):
        wrapper(torch.randn(257, 2, 160))
    with pytest.raises(RuntimeError, match="length"):
        wrapper(torch.randn(2, 2, 159))
    with pytest.raises(RuntimeError, match="length"):
        wrapper(torch.randn(2, 2, 161))
    with pytest.raises(ValueError, match="exactly 256"):
        dual_export.ADV3B02DualFeatureRuntime(
            _TinyDual(),
            expected_input_len=160,
            expected_tx_classes=3,
            runtime_batch_size=255,
        )
    wrong_tx_width = dual_export.ADV3B02DualFeatureRuntime(
        _TinyDual(),
        expected_input_len=160,
        expected_tx_classes=4,
        runtime_batch_size=256,
    ).eval()
    with pytest.raises(RuntimeError, match="TX class width"):
        wrong_tx_width(torch.randn(2, 2, 160))


def test_real_lite_d_runtime_trace_matches_training_path_and_strips_heads(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    model = DualCVSincNetDisentangle(
        num_classes=4,
        num_domains=3,
        model_size="S",
        input_len=256,
        model_variant="lite_d",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        fast_infer_when_no_aux=False,
    ).eval()
    rows = torch.randn(2, 2, 256)
    with torch.no_grad():
        expected = model(rows, return_aux=True)
    wrapper = dual_export.ADV3B02DualFeatureRuntime(
        model,
        expected_input_len=256,
        expected_tx_classes=4,
        runtime_batch_size=256,
    ).eval()
    runtime = dual_export._trace_and_save(
        wrapper, rows, tmp_path / "real_lite_d_dual.ts"
    )
    actual = runtime(rows)
    torch.testing.assert_close(actual[0], expected["z_id"], rtol=0.0, atol=1.0e-5)
    torch.testing.assert_close(actual[1], expected["z_dom"], rtol=0.0, atol=1.0e-5)
    torch.testing.assert_close(
        actual[2], expected["tx_logits"], rtol=0.0, atol=1.0e-5
    )
    chunked = tuple(
        torch.cat([runtime(rows[:1])[index], runtime(rows[1:])[index]], dim=0)
        for index in range(3)
    )
    permuted = runtime(rows[[1, 0]])
    for index in range(3):
        torch.testing.assert_close(
            chunked[index], actual[index], rtol=0.0, atol=1.0e-5
        )
        torch.testing.assert_close(
            permuted[index], actual[index][[1, 0]], rtol=0.0, atol=1.0e-5
        )
    forbidden = ("dom_head", "adv_head", "tx_adv_head")
    assert all(
        not any(token in name for token in forbidden)
        for name, _ in runtime.named_modules()
    )
    assert all(
        not any(token in name for token in forbidden)
        for name, _ in runtime.named_parameters()
    )
    component_audit = dual_verify._audit_runtime_components(runtime)
    assert component_audit["forbidden_runtime_components_absent"] is True


def test_export_writes_bound_nonformal_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    adapter = tmp_path / "adapter.pth"
    torch.save({"diagnostic": True}, checkpoint)
    torch.save({"unused": torch.zeros(1)}, adapter)
    checkpoint_bytes = checkpoint.read_bytes()
    adapter_bytes = adapter.read_bytes()
    read_counts = {"checkpoint": 0, "adapter state": 0}
    original_read = dual_export._read_regular_bytes

    def _counted_read(path: Path, name: str) -> bytes:
        read_counts[name] += 1
        return original_read(path, name)

    def _load_checkpoint_snapshot(value: bytes):
        checkpoint.write_bytes(b"swapped-checkpoint")
        try:
            assert value == checkpoint_bytes
            return {}
        finally:
            checkpoint.write_bytes(checkpoint_bytes)

    def _load_adapter_snapshot(value: bytes):
        adapter.write_bytes(b"swapped-adapter")
        try:
            assert value == adapter_bytes
            return {"unused": torch.zeros(1)}
        finally:
            adapter.write_bytes(adapter_bytes)

    monkeypatch.setattr(dual_export, "_read_regular_bytes", _counted_read)
    monkeypatch.setattr(
        dual_export, "BASE_CHECKPOINT_SHA256", dual_export.sha256_file(checkpoint)
    )
    monkeypatch.setattr(
        dual_export,
        "build_exact_ssdg_model_from_checkpoint",
        lambda *args, **kwargs: (_TinyDual(), {"strict": True}),
    )
    monkeypatch.setattr(
        dual_export, "_load_checkpoint_bytes", _load_checkpoint_snapshot
    )
    monkeypatch.setattr(
        dual_export,
        "_load_tensor_state_bytes",
        _load_adapter_snapshot,
    )
    monkeypatch.setattr(
        dual_export,
        "apply_fp16_lora_state",
        lambda *args, **kwargs: {
            "element_count": 44_048,
            "tensor_bytes_fp16": 88_096,
            "target_modules": list(dual_export.EFFECTIVE8_TARGET_MODULES),
        },
    )
    monkeypatch.setattr(
        dual_export,
        "merge_feat_joint_lora",
        lambda _model: {
            "merged_module_count": 8,
            "remaining_lora_wrappers": [],
            "algebraic_probe_parity_pass": True,
        },
    )
    args = argparse.Namespace(
        checkpoint=checkpoint,
        adapter_state=adapter,
        input_len=192,
        base_runtime_out=tmp_path / "base.ts",
        candidate_runtime_out=tmp_path / "candidate.ts",
        export_receipt_out=tmp_path / "export.json",
        device="cpu",
        parity_seed=17,
        parity_rows=4,
        runtime_batch_size=256,
        max_abs_tolerance=1.0e-4,
    )
    result = dual_export.export(args)
    payload = json.loads(args.export_receipt_out.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert payload["status"] == "PASS"
    assert payload["runtime_output_schema"] == dual_export.RUNTIME_OUTPUT_SCHEMA
    assert payload["feature_keys"] == {"z_id": "feat_joint", "z_dom": "feat_imp"}
    assert payload["feature_dimensions"] == {
        "z_id": 160,
        "z_dom": 160,
        "tx_logits": 3,
    }
    assert payload["expected_input_len"] == 192
    assert payload["runtime_batch_capacity"] == 256
    assert payload["runtime_invocations_per_prediction"] == 1
    assert payload["component_forward_counts_per_invocation"] == {
        "id_backbone": 1,
        "dom_backbone": 1,
        "dom_enhancer": 1,
    }
    assert payload["runtime_component_allowlist"] == list(
        dual_export.RUNTIME_COMPONENT_ALLOWLIST
    )
    assert payload["effective8_target_modules"] == list(
        dual_export.EFFECTIVE8_TARGET_MODULES
    )
    assert payload["formal_phase2_eligible"] is False
    assert payload["bundle_created"] is False
    assert payload["checkpoint_sha256"] == dual_export.sha256_file(checkpoint)
    assert payload["adapter_state_sha256"] == dual_export.sha256_file(adapter)
    assert payload["base_runtime_sha256"] == dual_export.sha256_file(args.base_runtime_out)
    assert payload["candidate_runtime_sha256"] == dual_export.sha256_file(
        args.candidate_runtime_out
    )
    assert read_counts == {"checkpoint": 1, "adapter state": 1}
    assert payload["resource_audit"]["trainable_parameters"] == 0
    assert payload["resource_audit"]["optimizer_steps"] == 0


def test_export_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "exists.ts"
    target.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="overwrite"):
        dual_export._trace_and_save(
            dual_export.ADV3B02DualFeatureRuntime(
                _TinyDual(),
                expected_input_len=160,
                expected_tx_classes=3,
                runtime_batch_size=256,
            ).eval(),
            torch.randn(2, 2, 160),
            target,
        )


def test_export_requires_sealed_runtime_capacity_256() -> None:
    with pytest.raises(ValueError, match="exactly 256"):
        dual_export.export(argparse.Namespace(runtime_batch_size=255))
