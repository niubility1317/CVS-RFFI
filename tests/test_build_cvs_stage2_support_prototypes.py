from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cvsrffi.stage2_structured_late_block_adaptation import StructuredLateBlockAudit
from scripts import build_cvs_stage2_support_prototypes as subject
from scripts import smoke_adv3b02_structured_lateblock_no_query as no_query_smoke


CAPSULE_ID = "536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2"
SPLIT_ID = "260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25"
CHECKPOINT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/"
    "best_joint_safe_ssdg.pth"
)
CLASS_IDS = list(range(26))
TARGET_NEW_CLASS_IDS = list(range(6, 26))
K_SHOT = 20
REPO_ROOT = Path(__file__).resolve().parents[1]


class _ToyCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()


def _write_support(path: Path, *, extra: dict[str, np.ndarray] | None = None) -> None:
    labels = np.repeat(np.asarray(CLASS_IDS, dtype=np.int64), K_SHOT)
    shots = np.tile(np.arange(K_SHOT, dtype=np.float32), len(CLASS_IDS))
    classes = labels.astype(np.float32)
    received_iq = np.stack(
        [
            np.stack([classes + 1.0, shots + 1.0], axis=1),
            np.stack([np.ones_like(classes), classes + shots + 1.0], axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    payload = {
        "received_iq": received_iq,
        "support_labels": labels,
    }
    payload.update(extra or {})
    np.savez(path, **payload)


def _audit_payload() -> dict[str, object]:
    row_count = len(CLASS_IDS) * K_SHOT
    return {
        "schema": "cvs.stage2.target_row_export.v1",
        "mode": "support_only_no_query_smoke",
        "k_shot": K_SHOT,
        "support_input_rows": row_count,
        "support_output_rows": row_count,
        "support_class_count": len(CLASS_IDS),
        "support_class_ids": CLASS_IDS,
        "support_per_class_counts": {
            str(class_id): K_SHOT for class_id in CLASS_IDS
        },
        "support_selected_ids": [f"physical-{index:04d}" for index in range(row_count)],
        "support_ids_preserved": True,
        "query_input_opened": False,
        "query_input_rows": 0,
        "query_output_rows": 0,
        "query_ids": [],
        "query_ids_preserved": False,
        "query_truth_opened": False,
        "query_role_opened": False,
    }


def _write_audit(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(payload or _audit_payload(), sort_keys=True),
        encoding="utf-8",
    )


def _config(support_path: Path, prototype_path: Path) -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": CAPSULE_ID,
        "split_id": SPLIT_ID,
        "checkpoint_path": CHECKPOINT,
        "support_path": str(support_path),
        "prototype_path": str(prototype_path),
        "candidate": "freq_f3_proj",
        "steps": 1,
        "learning_rate": 0.0005,
        "seed": 20260828,
        "k_shot": K_SHOT,
    }


def _patch_checkpoint_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject,
        "_load_frozen_checkpoint",
        lambda _path, *, device: _ToyCheckpoint().to(device),
    )

    def identity_features(_model: nn.Module, rows: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                rows[:, 0, 0],
                rows[:, 0, 1],
                rows[:, 1, 0] + rows[:, 1, 1],
            ),
            dim=1,
        )

    monkeypatch.setattr(subject, "_identity_features", identity_features)


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], Path, Path]:
    support_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"
    audit_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.audit.json"
    output_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
    _write_support(support_path)
    _write_audit(audit_path)
    _patch_checkpoint_embedding(monkeypatch)
    audit = subject.build_support_prototypes(
        _config(support_path, output_path),
        support_audit_path=audit_path,
        scene="leo_clear_weak",
        receiver="1-1",
        device="cpu",
    )
    return audit, support_path, output_path


def test_builds_all_26_normalized_class_means_and_target_new_from_own_k20(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, support_path, output_path = _build(tmp_path, monkeypatch)

    with np.load(output_path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"prototypes", "class_ids"}
        prototypes = artifact["prototypes"]
        class_ids = artifact["class_ids"]
    assert class_ids.tolist() == CLASS_IDS
    assert prototypes.shape[0] == 26
    assert np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-6)
    with np.load(support_path, allow_pickle=False) as support:
        rows = support["received_iq"]
        labels = support["support_labels"]
    features = np.stack(
        (
            rows[:, 0, 0],
            rows[:, 0, 1],
            rows[:, 1, 0] + rows[:, 1, 1],
        ),
        axis=1,
    ).astype(np.float64)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    expected = np.stack(
        [features[labels == class_id].mean(axis=0) for class_id in CLASS_IDS]
    )
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(prototypes, expected, rtol=1e-6, atol=1e-6)
    assert len({tuple(row.tolist()) for row in prototypes[6:]}) == 20
    assert audit["target_new_class_ids"] == TARGET_NEW_CLASS_IDS
    assert audit["target_new_prototypes_from_own_support"] is True
    assert audit["support_rows"] == 520
    assert audit["support_physical_ids_unique"] is True
    assert audit["query_opened"] is False
    assert audit["source_opened"] is False
    assert audit["clean_opened"] is False


def test_same_seed_and_input_are_bitwise_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _audit, support_path, first_path = _build(tmp_path, monkeypatch)
    second_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20_second.npz"
    subject.build_support_prototypes(
        _config(support_path, second_path),
        support_audit_path=tmp_path / "support_leo_clear_weak_rx1-1_k20.audit.json",
        scene="leo_clear_weak",
        receiver="1-1",
        device="cpu",
    )
    with np.load(first_path, allow_pickle=False) as first, np.load(
        second_path, allow_pickle=False
    ) as second:
        np.testing.assert_array_equal(first["class_ids"], second["class_ids"])
        np.testing.assert_array_equal(first["prototypes"], second["prototypes"])


def test_numpy2_pytorch21_bridge_uses_buffers_in_both_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_from_numpy(_value: object) -> torch.Tensor:
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    def reject_tensor_numpy(_value: torch.Tensor) -> np.ndarray:
        raise TypeError("expected 0 arguments, got 1")

    monkeypatch.setattr(torch, "from_numpy", reject_from_numpy)
    monkeypatch.setattr(torch.Tensor, "numpy", reject_tensor_numpy)

    audit, _support_path, output_path = _build(tmp_path, monkeypatch)

    with np.load(output_path, allow_pickle=False) as artifact:
        assert artifact["class_ids"].tolist() == CLASS_IDS
        assert artifact["prototypes"].shape[0] == len(CLASS_IDS)
    assert audit["support_rows"] == len(CLASS_IDS) * K_SHOT


def test_no_query_smoke_uses_buffers_when_numpy_bridge_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"
    prototype_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
    config_path = tmp_path / "smoke_r2.json"
    output_path = tmp_path / "smoke.json"
    _write_support(support_path)
    np.savez(
        prototype_path,
        prototypes=np.ones((len(CLASS_IDS), 3), dtype=np.float32),
        class_ids=np.asarray(CLASS_IDS, dtype=np.int64),
    )
    config_path.write_text(
        json.dumps(_config(support_path, prototype_path), sort_keys=True),
        encoding="utf-8",
    )

    def reject_from_numpy(_value: object) -> torch.Tensor:
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    monkeypatch.setattr(torch, "from_numpy", reject_from_numpy)
    monkeypatch.setattr(
        no_query_smoke,
        "_load_frozen_checkpoint",
        lambda _path, *, device: _ToyCheckpoint().to(device),
    )
    monkeypatch.setattr(
        no_query_smoke,
        "adapt_on_target_support_with_frozen_prototypes",
        lambda *_args, **_kwargs: StructuredLateBlockAudit(
            method_id="SCLBA_V1",
            candidate="freq_f3_proj",
            gradient_updates=1,
            support_samples=len(CLASS_IDS) * K_SHOT,
            support_class_count=len(CLASS_IDS),
            trainable_parameters=0,
            total_parameters=1,
            trainable_fraction=0.0,
            trainable_parameter_names=(),
            structural_trainable_parameters=0,
            classifier_parameters_changed=0,
            prototypes_changed=False,
            loss_trace=(),
        ),
    )
    monkeypatch.setattr(
        no_query_smoke.sys,
        "argv",
        [
            "smoke_adv3b02_structured_lateblock_no_query.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--output-json",
            str(output_path),
        ],
    )

    assert no_query_smoke.main() == 0
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS"
    assert receipt["query_opened"] is False


@pytest.mark.parametrize(
    "forbidden_member",
    ["query_iq", "query_truth", "query_role", "source_iq", "clean_iq"],
)
def test_support_npz_rejects_query_truth_source_and_clean_members_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_member: str,
) -> None:
    support_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"
    audit_path = tmp_path / "support.audit.json"
    output_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
    _write_support(
        support_path,
        extra={forbidden_member: np.zeros(520, dtype=np.float32)},
    )
    _write_audit(audit_path)
    monkeypatch.setattr(
        subject,
        "_load_frozen_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("checkpoint opened before support allowlist"),
    )

    with pytest.raises(ValueError, match="support.*allowlist"):
        subject.build_support_prototypes(
            _config(support_path, output_path),
            support_audit_path=audit_path,
            scene="leo_clear_weak",
            receiver="1-1",
            device="cpu",
        )
    assert not output_path.exists()


@pytest.mark.parametrize("field", ["capsule_id", "split_id"])
def test_wrong_handle_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    support_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"
    audit_path = tmp_path / "support.audit.json"
    output_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
    _write_support(support_path)
    _write_audit(audit_path)
    config = _config(support_path, output_path)
    config[field] = "f" * 64
    _patch_checkpoint_embedding(monkeypatch)

    with pytest.raises(ValueError, match=field):
        subject.build_support_prototypes(
            config,
            support_audit_path=audit_path,
            scene="leo_clear_weak",
            receiver="1-1",
            device="cpu",
        )


def test_wrong_k_class_scene_and_duplicate_physical_ids_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"
    audit_path = tmp_path / "support.audit.json"
    output_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
    _write_support(support_path)
    _write_audit(audit_path)
    _patch_checkpoint_embedding(monkeypatch)

    wrong_k = _config(support_path, output_path)
    wrong_k["k_shot"] = 19
    with pytest.raises(ValueError, match="k_shot"):
        subject.build_support_prototypes(
            wrong_k,
            support_audit_path=audit_path,
            scene="leo_clear_weak",
            receiver="1-1",
            device="cpu",
        )

    wrong_class = _audit_payload()
    wrong_class["support_class_ids"] = CLASS_IDS[:-1]
    _write_audit(audit_path, wrong_class)
    with pytest.raises(ValueError, match="class"):
        subject.build_support_prototypes(
            _config(support_path, output_path),
            support_audit_path=audit_path,
            scene="leo_clear_weak",
            receiver="1-1",
            device="cpu",
        )

    _write_audit(audit_path)
    with pytest.raises(ValueError, match="scene"):
        subject.build_support_prototypes(
            _config(support_path, output_path),
            support_audit_path=audit_path,
            scene="leo_rain_weak",
            receiver="1-1",
            device="cpu",
        )

    duplicate_ids = _audit_payload()
    duplicate_ids["support_selected_ids"][1] = duplicate_ids["support_selected_ids"][0]
    _write_audit(audit_path, duplicate_ids)
    with pytest.raises(ValueError, match="physical"):
        subject.build_support_prototypes(
            _config(support_path, output_path),
            support_audit_path=audit_path,
            scene="leo_clear_weak",
            receiver="1-1",
            device="cpu",
        )


def test_config_and_cli_have_no_query_source_clean_or_truth_path(
    tmp_path: Path,
) -> None:
    parameters = set(inspect.signature(subject.build_support_prototypes).parameters)
    parser_destinations = {
        action.dest for action in subject._parser()._actions  # noqa: SLF001
    }
    forbidden = {
        "query",
        "query_path",
        "query_truth",
        "query_role",
        "source_path",
        "clean_path",
    }
    assert not parameters & forbidden
    assert not parser_destinations & forbidden

    support_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"
    output_path = tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
    config = _config(support_path, output_path)
    config["query_path"] = str(tmp_path / "forbidden.npz")
    with pytest.raises(ValueError, match="config allowlist"):
        subject.build_support_prototypes(
            config,
            support_audit_path=tmp_path / "missing.audit.json",
            scene="leo_clear_weak",
            receiver="1-1",
            device="cpu",
        )


def test_committed_smoke_config_is_directly_consumable_by_bridge_and_no_query_smoke() -> None:
    for name, run_id in (
        (
            "phase2_canonical_union_k20_smoke_v1.json",
            "P2_CANONICAL_UNION_SMOKE_V1_20260828",
        ),
        (
            "phase2_canonical_union_k20_smoke_v1_r2.json",
            "P2_CANONICAL_UNION_SMOKE_V1_20260828_R2",
        ),
        (
            "phase2_canonical_union_k20_smoke_v1_r3_support.json",
            "P2_CANONICAL_UNION_SMOKE_V1_20260828_R3",
        ),
    ):
        config_path = REPO_ROOT / "configs" / name
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))

        resolved = subject._validate_config(config)  # noqa: SLF001
        subject._validate_row_binding(  # noqa: SLF001
            resolved,
            scene="leo_clear_weak",
            receiver="1-1",
        )
        assert set(config) == subject._CONFIG_ALLOWLIST  # noqa: SLF001
        assert set(config) == no_query_smoke._LEGACY_CONFIG_ALLOWLIST  # noqa: SLF001
        assert not any("query" in key or "truth" in key for key in config)
        assert f"/runs/{run_id}/input/" in config["support_path"]
        assert config["support_path"].endswith(
            "/support_only_leo_clear_weak_rx1-1_k20.npz"
        )
        assert f"/runs/{run_id}/input/" in config["prototype_path"]
        assert config["prototype_path"].endswith(
            "/prototypes_leo_clear_weak_rx1-1_k20.npz"
        )

    r3_path = (
        REPO_ROOT / "configs" / "phase2_canonical_union_k20_smoke_v1_r3.json"
    )
    r3_config = json.loads(r3_path.read_text(encoding="utf-8-sig"))
    assert set(r3_config) == no_query_smoke._CONFIG_ALLOWLIST  # noqa: SLF001
    assert not any("query" in key or "truth" in key for key in r3_config)
    structured = no_query_smoke._structured_config(r3_config)  # noqa: SLF001
    assert structured.candidate == "freq_f3_proj"
    assert structured.min_trainable_fraction == 0.03
    assert structured.max_trainable_fraction == 0.15
    assert structured.min_trainable_fraction <= 0.034342 <= structured.max_trainable_fraction
    assert "P2_CANONICAL_UNION_SMOKE_V1_20260828_R3" in r3_config["support_path"]
    assert "P2_CANONICAL_UNION_SMOKE_V1_20260828_R3" in r3_config["prototype_path"]
