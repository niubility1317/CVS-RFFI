from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_m26_spectral_anchor import load_m26_spectral_anchor
from scripts import build_m26_phase1_spectral_anchor as builder


def _write_binding(path: Path, registry: tuple[str, ...], tx_order: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.d20_adv3b02_class_binding.v2",
                "checkpoint_sha256": "e" * 64,
                "entries": [
                    {
                        "class_index": index,
                        "direct_logit_index": index,
                        "phase1_tx": tx_order[index],
                        "registered_class_handle": registry[index],
                    }
                    for index in range(6)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_builder_reads_source_only_npz_and_maps_raw_labels(tmp_path: Path) -> None:
    rng = np.random.default_rng(82630)
    registry = tuple(f"old-{index}" for index in range(6))
    raw_order = tuple(str(index) for index in range(6))
    tx_order = tuple(f"tx-{index}" for index in range(6))
    manifest = {
        "source_checkpoint_sha256": "e" * 64,
        "class_id_to_tx": list(tx_order),
        "logit_class_order": list(range(6)),
    }
    binding = tmp_path / "class_binding.json"
    _write_binding(binding, registry, tx_order)
    identity = []
    fft = []
    labels = []
    for index, raw in enumerate(raw_order):
        for _ in range(4):
            left = 0.01 * rng.normal(size=160)
            right = 0.01 * rng.normal(size=96)
            left[index * 9] += 1.0
            right[index * 7] += 1.0
            identity.append(left)
            fft.append(right)
            labels.append(raw)
    source = tmp_path / "source_only.npz"
    target_identity = rng.normal(size=(2, 160)).astype(np.float32)
    target_fft = rng.normal(size=(2, 96)).astype(np.float32)
    np.savez(
        source,
        features=np.concatenate([np.asarray(identity, dtype=np.float32), target_identity]),
        fft_logmag_features=np.concatenate([np.asarray(fft, dtype=np.float32), target_fft]),
        raw_labels=np.asarray(labels + ["90", "91"]),
        dataset_role=np.asarray(["source"] * len(labels) + ["target_old", "target_new"]),
        manifest_json=np.asarray(json.dumps(manifest)),
    )
    output = tmp_path / "anchor.npz"
    audit = tmp_path / "anchor_audit.json"
    result = builder.build_from_source_npz(
        source_npz=source,
        output_path=output,
        audit_path=audit,
        checkpoint_sha256="e" * 64,
        class_binding_json=binding,
    )
    loaded = load_m26_spectral_anchor(output, expected_checkpoint_sha256="e" * 64)
    assert loaded.class_registry == registry
    assert result["source_row_count"] == 24
    assert result["input_target_row_count"] == 2
    assert result["target_rows_used"] == 0
    assert result["query_rows_used"] == 0
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "VERIFIED"

    changed_target_source = tmp_path / "source_with_changed_target.npz"
    np.savez(
        changed_target_source,
        features=np.concatenate(
            [np.asarray(identity, dtype=np.float32), 1.0e6 * target_identity]
        ),
        fft_logmag_features=np.concatenate(
            [np.asarray(fft, dtype=np.float32), -1.0e6 * target_fft]
        ),
        raw_labels=np.asarray(labels + ["190", "191"]),
        dataset_role=np.asarray(["source"] * len(labels) + ["target_old", "target_new"]),
        manifest_json=np.asarray(
            json.dumps(manifest)
        ),
    )
    changed_output = tmp_path / "anchor_changed_target.npz"
    builder.build_from_source_npz(
        source_npz=changed_target_source,
        output_path=changed_output,
        audit_path=tmp_path / "anchor_changed_target_audit.json",
        checkpoint_sha256="e" * 64,
        class_binding_json=binding,
    )
    changed = load_m26_spectral_anchor(
        changed_output, expected_checkpoint_sha256="e" * 64
    )
    np.testing.assert_array_equal(changed.identity_q, loaded.identity_q)
    np.testing.assert_array_equal(changed.identity_scale, loaded.identity_scale)
    np.testing.assert_array_equal(changed.fft_q, loaded.fft_q)
    np.testing.assert_array_equal(changed.fft_scale, loaded.fft_scale)


def test_builder_rejects_missing_manifest_identity_and_class_mapping_drift(
    tmp_path: Path,
) -> None:
    registry = tuple(f"old-{index}" for index in range(6))
    tx_order = tuple(f"tx-{index}" for index in range(6))
    binding = tmp_path / "binding.json"
    _write_binding(binding, registry, tx_order)
    rows = np.eye(6, 256, dtype=np.float32)
    common = {
        "features": rows[:, :160],
        "fft_logmag_features": rows[:, 160:],
        "raw_labels": np.asarray([str(index) for index in range(6)]),
        "dataset_role": np.asarray(["source"] * 6),
    }
    missing = tmp_path / "missing_manifest.npz"
    np.savez(missing, **common, source_checkpoint_sha256=np.asarray("e" * 64))
    with pytest.raises(ValueError, match="manifest_json"):
        builder.build_from_source_npz(
            source_npz=missing,
            output_path=tmp_path / "missing_anchor.npz",
            audit_path=tmp_path / "missing_audit.json",
            checkpoint_sha256="e" * 64,
            class_binding_json=binding,
        )

    drift_binding = tmp_path / "drift_binding.json"
    _write_binding(drift_binding, registry, tuple(reversed(tx_order)))
    valid = tmp_path / "valid_manifest.npz"
    np.savez(
        valid,
        **common,
        manifest_json=np.asarray(
            json.dumps(
                {
                    "source_checkpoint_sha256": "e" * 64,
                    "class_id_to_tx": list(tx_order),
                    "logit_class_order": list(range(6)),
                }
            )
        ),
    )
    with pytest.raises(ValueError, match="class binding"):
        builder.build_from_source_npz(
            source_npz=valid,
            output_path=tmp_path / "drift_anchor.npz",
            audit_path=tmp_path / "drift_audit.json",
            checkpoint_sha256="e" * 64,
            class_binding_json=drift_binding,
        )
