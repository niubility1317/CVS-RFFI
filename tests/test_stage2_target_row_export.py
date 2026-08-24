from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_target_row_export import export_target_row, main


def _write_support(path: Path) -> dict[str, np.ndarray]:
    iq = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
    payload = {
        "support_pool_leo_weak_iq": iq,
        "support_pool_class_indices": np.asarray(
            [20, 10, 20, 10, 10, 20], dtype=np.int64
        ),
        "support_pool_rank_within_class": np.asarray(
            [1, 0, 0, 2, 1, 2], dtype=np.int64
        ),
        "support_pool_tokens": np.asarray(
            ["c20-r1", "c10-r0", "c20-r0", "c10-r2", "c10-r1", "c20-r2"]
        ),
    }
    np.savez(path, **payload)
    return payload


def _write_query(path: Path) -> dict[str, np.ndarray]:
    payload = {
        "query_leo_weak_iq": np.arange(3 * 2 * 4, dtype=np.float32).reshape(
            3, 2, 4
        ),
        "query_tokens": np.asarray(["q-001", "q-002", "q-003"]),
    }
    np.savez(path, **payload)
    return payload


def test_support_export_selects_rank_prefix_and_preserves_physical_ids(
    tmp_path: Path,
) -> None:
    support_input = tmp_path / "support_pool.npz"
    source = _write_support(support_input)
    support_output = tmp_path / "support.npz"
    audit_output = tmp_path / "audit.json"

    audit = export_target_row(
        support_input=support_input,
        support_output=support_output,
        audit_output=audit_output,
        k_shot=2,
    )

    selected = np.asarray([0, 1, 2, 4], dtype=np.int64)
    with np.load(support_output, allow_pickle=False) as exported:
        assert set(exported.files) == {
            "received_iq",
            "support_labels",
            "support_physical_ids",
        }
        np.testing.assert_array_equal(
            exported["received_iq"], source["support_pool_leo_weak_iq"][selected]
        )
        np.testing.assert_array_equal(
            exported["support_labels"],
            source["support_pool_class_indices"][selected],
        )
        np.testing.assert_array_equal(
            exported["support_physical_ids"],
            source["support_pool_tokens"][selected],
        )
    assert audit == json.loads(audit_output.read_text(encoding="utf-8"))
    assert audit["support_input_rows"] == 6
    assert audit["support_output_rows"] == 4
    assert audit["support_class_count"] == 2
    assert audit["k_shot"] == 2
    assert audit["support_selected_ids"] == [
        "c20-r1",
        "c10-r0",
        "c20-r0",
        "c10-r1",
    ]
    assert audit["support_ids_preserved"] is True
    assert audit["query_input_opened"] is False
    assert audit["query_output_rows"] == 0


def test_formal_export_copies_query_iq_and_ids_without_truth(
    tmp_path: Path,
) -> None:
    support_input = tmp_path / "support_pool.npz"
    query_input = tmp_path / "query_pool.npz"
    _write_support(support_input)
    query = _write_query(query_input)
    support_output = tmp_path / "support.npz"
    query_output = tmp_path / "query.npz"
    audit_output = tmp_path / "audit.json"

    audit = export_target_row(
        support_input=support_input,
        support_output=support_output,
        audit_output=audit_output,
        k_shot=2,
        query_input=query_input,
        query_output=query_output,
    )

    with np.load(query_output, allow_pickle=False) as exported:
        assert set(exported.files) == {"received_iq", "query_ids"}
        np.testing.assert_array_equal(
            exported["received_iq"], query["query_leo_weak_iq"]
        )
        np.testing.assert_array_equal(exported["query_ids"], query["query_tokens"])
        assert not {"truth", "query_truth", "query_labels", "query_role"} & set(
            exported.files
        )
    assert audit["query_input_opened"] is True
    assert audit["query_input_rows"] == 3
    assert audit["query_output_rows"] == 3
    assert audit["query_ids"] == ["q-001", "q-002", "q-003"]
    assert audit["query_ids_preserved"] is True
    assert audit["query_truth_opened"] is False
    assert audit["query_role_opened"] is False


def test_accepts_existing_validated_once_bundle_member_schema(tmp_path: Path) -> None:
    support_input = tmp_path / "support_bundle.npz"
    query_input = tmp_path / "query_bundle.npz"
    support = _write_support(support_input)
    support.update(
        {
            "support_pool_overlay_tokens": np.asarray([f"ov-{i}" for i in range(6)]),
            "support_pool_satellite_seeds": np.arange(6, dtype=np.int64),
            "support_pool_post_channel_iq_sha256": np.asarray([f"sha-{i}" for i in range(6)]),
            "manifest_json": np.asarray("{}"),
        }
    )
    np.savez(support_input, **support)
    query = _write_query(query_input)
    query.update(
        {
            "query_overlay_tokens": np.asarray([f"qov-{i}" for i in range(3)]),
            "query_satellite_seeds": np.arange(3, dtype=np.int64),
            "query_post_channel_iq_sha256": np.asarray([f"qsha-{i}" for i in range(3)]),
            "manifest_json": np.asarray("{}"),
        }
    )
    np.savez(query_input, **query)

    audit = export_target_row(
        support_input=support_input,
        support_output=tmp_path / "support.npz",
        audit_output=tmp_path / "audit.json",
        k_shot=2,
        query_input=query_input,
        query_output=tmp_path / "query.npz",
    )
    assert audit["support_output_rows"] == 4
    assert audit["query_output_rows"] == 3


@pytest.mark.parametrize("forbidden_key", ["query_truth", "query_role", "query_labels"])
def test_query_payload_rejects_forbidden_extra_fields_without_export(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    support_input = tmp_path / "support_pool.npz"
    query_input = tmp_path / "query_pool.npz"
    _write_support(support_input)
    query = _write_query(query_input)
    np.savez(
        query_input,
        **query,
        **{forbidden_key: np.asarray([99, 98, 97], dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="query.*allowlist"):
        export_target_row(
            support_input=support_input,
            support_output=tmp_path / "support.npz",
            audit_output=tmp_path / "audit.json",
            k_shot=2,
            query_input=query_input,
            query_output=tmp_path / "query.npz",
        )
    assert not (tmp_path / "support.npz").exists()
    assert not (tmp_path / "query.npz").exists()
    assert not (tmp_path / "audit.json").exists()


def test_support_payload_is_ground_only_and_requires_complete_rank_prefix(
    tmp_path: Path,
) -> None:
    support_input = tmp_path / "support_pool.npz"
    support = _write_support(support_input)
    np.savez(
        support_input,
        **support,
        source_features=np.zeros((6, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="support.*allowlist"):
        export_target_row(
            support_input=support_input,
            support_output=tmp_path / "support.npz",
            audit_output=tmp_path / "audit.json",
            k_shot=2,
        )

    incomplete_input = tmp_path / "incomplete_support_pool.npz"
    support = _write_support(incomplete_input)
    support["support_pool_rank_within_class"][0] = 2
    np.savez(incomplete_input, **support)
    with pytest.raises(ValueError, match="rank prefix"):
        export_target_row(
            support_input=incomplete_input,
            support_output=tmp_path / "incomplete_support.npz",
            audit_output=tmp_path / "incomplete_audit.json",
            k_shot=2,
        )


def test_export_preflights_all_outputs_and_never_overwrites(
    tmp_path: Path,
) -> None:
    support_input = tmp_path / "support_pool.npz"
    query_input = tmp_path / "query_pool.npz"
    _write_support(support_input)
    _write_query(query_input)
    support_output = tmp_path / "support.npz"
    support_output.write_bytes(b"keep-existing-output")

    with pytest.raises(ValueError, match="already exists"):
        export_target_row(
            support_input=support_input,
            support_output=support_output,
            audit_output=tmp_path / "audit.json",
            k_shot=2,
            query_input=query_input,
            query_output=tmp_path / "query.npz",
        )
    assert support_output.read_bytes() == b"keep-existing-output"
    assert not (tmp_path / "query.npz").exists()
    assert not (tmp_path / "audit.json").exists()


def test_cli_support_only_mode_does_not_require_or_open_query(tmp_path: Path) -> None:
    support_input = tmp_path / "support_pool.npz"
    _write_support(support_input)

    exit_code = main(
        [
            "--support-input",
            str(support_input),
            "--support-output",
            str(tmp_path / "support.npz"),
            "--audit-output",
            str(tmp_path / "audit.json"),
            "--k-shot",
            "2",
        ]
    )

    assert exit_code == 0
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["mode"] == "support_only_no_query_smoke"
    assert audit["query_input_opened"] is False
