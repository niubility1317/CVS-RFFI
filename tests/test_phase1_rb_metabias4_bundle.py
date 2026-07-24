from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase1_rb_metabias4_bundle import (
    MANIFEST_NAME,
    NPZ_NAME,
    PAYLOAD_MEMBERS,
    RBMetaBias4BundleError,
    RBMetaBias4Config,
    apply_metabias4,
    build_phase1_rb_metabias4_bundle,
    infer_metabias4_coefficient,
    load_phase1_rb_metabias4_bundle,
    merge_verified_phase1_tap_and_dual_archives,
    save_phase1_rb_metabias4_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def synthetic_tap_archive(
    *, receivers: int = 3, classes: int = 3, days: int = 2, per_cell: int = 6
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(102)
    class_ids = tuple(f"class-{index}" for index in range(classes))
    class_pre = rng.normal(scale=0.15, size=(classes, 160))
    class_dom = rng.normal(scale=0.2, size=(classes, 160))
    rx_pre = rng.normal(scale=0.12, size=(receivers, 160))
    rx_dom = rng.normal(scale=0.3, size=(receivers, 160))
    day_pre = rng.normal(scale=0.04, size=(days, 160))
    day_dom = rng.normal(scale=0.08, size=(days, 160))
    pre, zdom, labels, rx_ids, day_ids, physical = [], [], [], [], [], []
    for receiver in range(receivers):
        for day in range(days):
            for class_index, class_id in enumerate(class_ids):
                for sample in range(per_cell):
                    pre.append(
                        0.6
                        + class_pre[class_index]
                        + rx_pre[receiver]
                        + day_pre[day]
                        + rng.normal(scale=0.03, size=160)
                    )
                    zdom.append(
                        class_dom[class_index]
                        + rx_dom[receiver]
                        + day_dom[day]
                        + rng.normal(scale=0.03, size=160)
                    )
                    labels.append(class_id)
                    rx_ids.append(f"rx-{receiver}")
                    day_ids.append(f"day-{day}")
                    physical.append(f"p-{receiver}-{day}-{class_index}-{sample}")
    return {
        "pre_relu": np.asarray(pre, dtype=np.float32),
        "z_dom": np.asarray(zdom, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.str_),
        "receiver_ids": np.asarray(rx_ids, dtype=np.str_),
        "day_ids": np.asarray(day_ids, dtype=np.str_),
        "physical_ids": np.asarray(physical, dtype=np.str_),
        "class_ids": np.asarray(class_ids, dtype=np.str_),
    }


def _build(arrays: dict[str, np.ndarray]):
    return build_phase1_rb_metabias4_bundle(
        arrays,
        checkpoint_sha256=SHA_A,
        runtime_sha256=SHA_B,
        method_lock_sha256=SHA_C,
        config=RBMetaBias4Config(),
    )


def test_bundle_is_deterministic_class_free_int8_and_joint_bound(tmp_path: Path) -> None:
    arrays = synthetic_tap_archive()
    first, second = _build(arrays), _build(arrays)
    assert first.content_root_sha256 == second.content_root_sha256
    assert first.basis().shape == (160, 4)
    assert first.domain_encoder().shape == (32, 160)
    assert first.bank_g().shape == (6, 32)
    assert first.bank_t().shape == (6, 4)
    assert first.numeric_state_bytes < 80 * 1024
    assert first.aggregation_receipt["minimum_observed_class_cell_physical_count"] == 6
    assert first.aggregation_receipt["class_free_payload"] is True
    assert first.quantization_receipt["persistent_fp32_sidecar"] is False
    result = save_phase1_rb_metabias4_bundle(tmp_path / "bundle", first)
    loaded = load_phase1_rb_metabias4_bundle(
        tmp_path / "bundle",
        expected_checkpoint_sha256=SHA_A,
        expected_runtime_sha256=SHA_B,
        expected_method_lock_sha256=SHA_C,
    )
    assert loaded.content_root_sha256 == first.content_root_sha256
    assert result["formal_phase2_eligible"] is False
    with np.load(tmp_path / "bundle" / NPZ_NAME, allow_pickle=False) as archive:
        assert tuple(archive.files) == PAYLOAD_MEMBERS
        assert not any(
            token in name
            for name in archive.files
            for token in ("class", "receiver", "day", "physical", "member")
        )
    text = (tmp_path / "bundle" / MANIFEST_NAME).read_text(encoding="utf-8")
    assert all(token not in text for token in ("rx-0", "day-0", "class-0", "p-0-"))


def test_bundle_rejects_single_physical_class_cell_and_tamper(tmp_path: Path) -> None:
    arrays = synthetic_tap_archive(per_cell=2)
    mask = ~(
        (arrays["receiver_ids"] == "rx-0")
        & (arrays["day_ids"] == "day-0")
        & (arrays["labels"] == "class-0")
        & (np.arange(len(arrays["labels"])) % 2 == 0)
    )
    broken = {
        name: value if name == "class_ids" else value[mask]
        for name, value in arrays.items()
    }
    with pytest.raises(RBMetaBias4BundleError, match="at least two"):
        _build(broken)
    bundle = _build(synthetic_tap_archive())
    save_phase1_rb_metabias4_bundle(tmp_path / "bundle", bundle)
    manifest = tmp_path / "bundle" / MANIFEST_NAME
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    parsed["checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(RBMetaBias4BundleError, match="seal"):
        load_phase1_rb_metabias4_bundle(tmp_path / "bundle")


def test_closed_form_is_class_permutation_equal_and_changes_pre_relu_geometry() -> None:
    arrays = synthetic_tap_archive()
    bundle = _build(arrays)
    indices = []
    for class_id in arrays["class_ids"]:
        indices.extend(np.flatnonzero(arrays["labels"] == class_id)[:2].tolist())
    indices = np.asarray(indices)
    labels = arrays["labels"][indices].tolist()
    coefficient, audit = infer_metabias4_coefficient(
        bundle, arrays["z_dom"][indices], labels
    )
    mapping = {value: f"renamed-{index}" for index, value in enumerate(reversed(sorted(set(labels))))}
    renamed = [mapping[value] for value in labels]
    other, other_audit = infer_metabias4_coefficient(
        bundle, arrays["z_dom"][indices], renamed
    )
    np.testing.assert_array_equal(coefficient, other)
    assert audit["all_classes_equal_outer_weight"] is True
    assert audit["old_new_role_access"] is False
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["coverage_hard_gate"] is False
    assert other_audit["class_support_counts"] == audit["class_support_counts"]
    adapted = apply_metabias4(bundle, arrays["pre_relu"][indices], coefficient)
    assert adapted.shape == (len(indices), 160)
    np.testing.assert_allclose(np.linalg.norm(adapted, axis=1), 1.0, atol=1.0e-6)


def test_tap_dual_merge_requires_same_rows_and_zid_parity() -> None:
    arrays = synthetic_tap_archive()
    z_id = np.maximum(arrays["pre_relu"], 0.0).astype(np.float32)
    tap = {
        "z_id": z_id,
        "pre_relu": arrays["pre_relu"],
        **{name: arrays[name] for name in (
            "labels", "receiver_ids", "day_ids", "physical_ids", "class_ids"
        )},
    }
    dual = {
        "z_id": z_id.copy(),
        "z_dom": arrays["z_dom"],
        **{name: arrays[name] for name in (
            "labels", "receiver_ids", "day_ids", "physical_ids", "class_ids"
        )},
    }
    merged = merge_verified_phase1_tap_and_dual_archives(tap, dual)
    np.testing.assert_array_equal(merged["pre_relu"], arrays["pre_relu"])
    np.testing.assert_array_equal(merged["z_dom"], arrays["z_dom"])
    broken = dict(dual)
    broken["physical_ids"] = dual["physical_ids"].copy()
    broken["physical_ids"][0] = "wrong"
    with pytest.raises(RBMetaBias4BundleError, match="physical_ids"):
        merge_verified_phase1_tap_and_dual_archives(tap, broken)
