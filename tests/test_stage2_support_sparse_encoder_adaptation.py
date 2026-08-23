from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


def _subject():
    try:
        return importlib.import_module(
            "cvsrffi.stage2_support_sparse_encoder_adaptation"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"support-only sparse encoder adaptation is missing: {exc}")


class _NormBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)


class _FrozenClassifierHead(nn.Module):
    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(channels, classes, bias=True)


class _ToyIdentityBackbone(nn.Module):
    def __init__(self, *, reserve: int = 20_000) -> None:
        super().__init__()
        self.t1 = _NormBlock(4)
        self.t2 = _NormBlock(4)
        self.t3 = _NormBlock(4)
        self.f1 = _NormBlock(4)
        self.f2 = _NormBlock(4)
        self.f3 = _NormBlock(4)
        self.pa_b1 = _NormBlock(4)
        self.pa_b2 = _NormBlock(4)
        self.pa_b3 = _NormBlock(4)
        self.t_proj = nn.Linear(4, 4)
        self.stem = nn.Conv1d(2, 4, kernel_size=1, bias=False)
        self.cls_head = _FrozenClassifierHead(4, 2)
        self.reserve = nn.Parameter(torch.zeros(reserve), requires_grad=False)

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        rows = self.stem(rows)
        branches = (
            self.t1.norm(rows),
            self.t2.norm(rows),
            self.t3.norm(rows),
            self.f1.norm(rows),
            self.f2.norm(rows),
            self.f3.norm(rows),
            self.pa_b1.norm(rows),
            self.pa_b2.norm(rows),
            self.pa_b3.norm(rows),
        )
        return torch.stack(branches).mean(dim=0).mean(dim=-1)


class _ToyADV3B02(nn.Module):
    def __init__(self, *, reserve: int = 20_000) -> None:
        super().__init__()
        self.id_backbone = _ToyIdentityBackbone(reserve=reserve)
        self.dom_backbone = nn.Linear(4, 3)
        with torch.no_grad():
            self.id_backbone.cls_head.head.weight.copy_(
                torch.tensor([[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]])
            )
            self.id_backbone.cls_head.head.bias.zero_()

    def forward(self, rows: torch.Tensor, **_kwargs):
        z_id = self.id_backbone.encode(rows)
        logits = self.id_backbone.cls_head.head(z_id)
        return {"z_id": z_id, "tx_logits": logits}


def _support() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.tensor(
        [
            [[0.2, 2.0, 0.1], [2.0, 0.1, 0.2]],
            [[0.1, 1.8, 0.2], [1.9, 0.2, 0.1]],
            [[2.0, 0.1, 0.2], [0.2, 2.1, 0.1]],
            [[1.8, 0.2, 0.1], [0.1, 1.9, 0.2]],
        ],
        dtype=torch.float32,
    )
    return rows, torch.tensor([0, 0, 1, 1], dtype=torch.long)


def _context() -> dict[str, str]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed-iq",
        "split_id": "split-support-query-disjoint",
    }


def test_c1_candidate_updates_only_original_nonclassification_encoder() -> None:
    subject = _subject()
    model = _ToyADV3B02()
    support_iq, support_labels = _support()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    audit = subject.adapt_on_target_support(
        model,
        support_iq,
        support_labels,
        context=_context(),
        config=subject.SparseEncoderAdaptationConfig(
            candidate="c1_norm_affine",
            steps=3,
            learning_rate=0.05,
            feature_anchor_weight=0.05,
        ),
    )

    changed = {
        name
        for name, value in model.named_parameters()
        if not torch.equal(value.detach(), before[name])
    }
    assert changed
    assert changed == set(audit.trainable_parameter_names)
    assert all(name.startswith("id_backbone.") for name in changed)
    assert all(".cls_head." not in name and ".classifier." not in name for name in changed)
    assert all(".norm." in name for name in changed)
    assert audit.gradient_updates == 3
    assert audit.trainable_fraction <= 0.01
    assert audit.classifier_parameters_changed == 0
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_candidate_parameter_allowlists_are_exact_and_nested() -> None:
    subject = _subject()
    c1 = {
        f"id_backbone.{block}.norm.{affine}"
        for block in (
            "t1",
            "t2",
            "t3",
            "f1",
            "f2",
            "f3",
            "pa_b1",
            "pa_b2",
            "pa_b3",
        )
        for affine in ("weight", "bias")
    }
    c2_increment = {
        "id_backbone.freq_gate.conv.weight",
        "id_backbone.freq_gate.conv.bias",
        "id_backbone.pa_gate.net.weight",
        "id_backbone.pa_gate.net.bias",
    }
    c3_increment = {
        "id_backbone.f_proj.weight",
        "id_backbone.f_proj.bias",
    }
    assert set(subject._CANDIDATE_PARAMETER_NAMES["c1_norm_affine"]) == c1
    assert (
        set(subject._CANDIDATE_PARAMETER_NAMES["c2_norm_gates"]) - c1
        == c2_increment
    )
    assert (
        set(subject._CANDIDATE_PARAMETER_NAMES["c3_norm_gates_fproj"])
        - set(subject._CANDIDATE_PARAMETER_NAMES["c2_norm_gates"])
        == c3_increment
    )
    for forbidden in (
        "id_backbone.t_proj.bias",
        "id_backbone.pa_proj.0.bias",
        "id_backbone.dac_b3.norm.weight",
        "id_backbone.cls_head.weight",
        "id_backbone.classifier.bias",
        "dom_backbone.norm.weight",
    ):
        assert not any(
            subject._candidate_allows(candidate, forbidden)
            for candidate in subject._CANDIDATE_PARAMETER_NAMES
        )


def test_candidate_fails_closed_when_a_declared_parameter_is_missing() -> None:
    subject = _subject()
    model = _ToyADV3B02()
    del model.id_backbone.t1
    support_iq, support_labels = _support()
    with pytest.raises(subject.SparseEncoderAdaptationError, match="missing declared"):
        subject.adapt_on_target_support(
            model,
            support_iq,
            support_labels,
            context=_context(),
            config=subject.SparseEncoderAdaptationConfig(
                candidate="c1_norm_affine", steps=1
            ),
        )


def test_candidate_fails_closed_when_any_declared_parameter_has_no_gradient() -> None:
    subject = _subject()
    model = _ToyADV3B02()
    backbone = model.id_backbone

    def encode_without_t1(rows: torch.Tensor) -> torch.Tensor:
        rows = backbone.stem(rows)
        branches = (
            backbone.t2.norm(rows),
            backbone.t3.norm(rows),
            backbone.f1.norm(rows),
            backbone.f2.norm(rows),
            backbone.f3.norm(rows),
            backbone.pa_b1.norm(rows),
            backbone.pa_b2.norm(rows),
            backbone.pa_b3.norm(rows),
        )
        return torch.stack(branches).mean(dim=0).mean(dim=-1)

    backbone.encode = encode_without_t1
    support_iq, support_labels = _support()
    with pytest.raises(
        subject.SparseEncoderAdaptationError, match="received no support gradient"
    ):
        subject.adapt_on_target_support(
            model,
            support_iq,
            support_labels,
            context=_context(),
            config=subject.SparseEncoderAdaptationConfig(
                candidate="c1_norm_affine", steps=1
            ),
        )


def test_resource_caps_and_forbidden_phase2_inputs_fail_closed() -> None:
    subject = _subject()
    support_iq, support_labels = _support()
    with pytest.raises(subject.SparseEncoderAdaptationError, match="steps.*40"):
        subject.adapt_on_target_support(
            _ToyADV3B02(),
            support_iq,
            support_labels,
            context=_context(),
            config=subject.SparseEncoderAdaptationConfig(steps=41),
        )
    with pytest.raises(subject.SparseEncoderAdaptationError, match="1%"):
        subject.adapt_on_target_support(
            _ToyADV3B02(reserve=0),
            support_iq,
            support_labels,
            context=_context(),
            config=subject.SparseEncoderAdaptationConfig(steps=1),
        )
    for forbidden in (
        "source_cache_path",
        "clean_samples",
        "query_truth",
        "query_role",
    ):
        context = _context()
        context[forbidden] = "forbidden"
        with pytest.raises(subject.SparseEncoderAdaptationError, match="forbidden"):
            subject.adapt_on_target_support(
                _ToyADV3B02(),
                support_iq,
                support_labels,
                context=context,
                config=subject.SparseEncoderAdaptationConfig(steps=1),
            )


def test_query_forward_is_read_only_and_has_no_truth_or_role_interface() -> None:
    subject = _subject()
    model = _ToyADV3B02()
    support_iq, support_labels = _support()
    subject.adapt_on_target_support(
        model,
        support_iq,
        support_labels,
        context=_context(),
        config=subject.SparseEncoderAdaptationConfig(steps=2),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    predictions, scores = subject.predict_query_read_only(model, support_iq[:2])
    after = model.state_dict()

    assert predictions.shape == (2,)
    assert scores.shape == (2, 2)
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    adapt_parameters = set(inspect.signature(subject.adapt_on_target_support).parameters)
    query_parameters = set(inspect.signature(subject.predict_query_read_only).parameters)
    assert not {"query_iq", "query_labels", "query_truth", "query_role"} & adapt_parameters
    assert not {"query_labels", "query_truth", "query_role"} & query_parameters


def _write_enrollment(root: Path) -> None:
    root.mkdir()
    (root / "package_manifest.json").write_text(
        json.dumps(
            {
                "profile": "enrollment_only",
                "stage": "stage2b",
                "k_shot": 3,
                "support_pool_max_k": 3,
                "registered_class_count": 2,
                "target_channel_scenarios": ["leo_clear_weak"],
                "phase1_checkpoint_sha256": "a" * 64,
                **_context(),
            }
        ),
        encoding="utf-8",
    )
    iq = np.arange(6 * 2 * 8, dtype=np.float32).reshape(6, 2, 8)
    np.savez(
        root / "support_leo_clear_weak.npz",
        support_leo_weak_iq=iq,
        support_class_indices=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        support_rank_within_class=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        support_tokens=np.asarray([f"opaque-{index}" for index in range(6)]),
    )


def test_enrollment_loader_selects_support_prefix_without_query_surface(
    tmp_path: Path,
) -> None:
    subject = _subject()
    root = tmp_path / "enrollment_only"
    _write_enrollment(root)
    loaded = subject.load_validated_enrollment_support(
        root, scenario="leo_clear_weak", k_shot=2, context=_context()
    )
    assert loaded.iq.shape == (4, 2, 8)
    assert loaded.class_indices.tolist() == [0, 0, 1, 1]
    assert loaded.rank_within_class.tolist() == [0, 1, 0, 1]
    assert loaded.tokens == ("opaque-0", "opaque-1", "opaque-3", "opaque-4")
    assert loaded.checkpoint_sha256 == "a" * 64


def test_enrollment_loader_rejects_non_enrollment_and_forbidden_manifest(
    tmp_path: Path,
) -> None:
    subject = _subject()
    wrong_root = tmp_path / "query_bundle"
    _write_enrollment(wrong_root)
    with pytest.raises(subject.SparseEncoderAdaptationError, match="enrollment_only"):
        subject.load_validated_enrollment_support(
            wrong_root, scenario="leo_clear_weak", k_shot=2, context=_context()
        )
    root = tmp_path / "enrollment_only"
    _write_enrollment(root)
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_cache_path"] = "forbidden"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(subject.SparseEncoderAdaptationError, match="forbidden"):
        subject.load_validated_enrollment_support(
            root, scenario="leo_clear_weak", k_shot=2, context=_context()
        )


def test_enrollment_loader_binds_all_validated_once_handles(tmp_path: Path) -> None:
    subject = _subject()
    root = tmp_path / "enrollment_only"
    _write_enrollment(root)
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_id"] = "wrong-split"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(subject.SparseEncoderAdaptationError, match="split_id"):
        subject.load_validated_enrollment_support(
            root, scenario="leo_clear_weak", k_shot=2, context=_context()
        )


def test_runner_rejects_same_state_and_audit_path(tmp_path: Path) -> None:
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_stage2_support_sparse_encoder_adaptation.py"
    )
    spec = importlib.util.spec_from_file_location("sofesa_support_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    same = tmp_path / "same-output"
    with pytest.raises(ValueError, match="different paths"):
        runner._validate_output_paths(same, same)
