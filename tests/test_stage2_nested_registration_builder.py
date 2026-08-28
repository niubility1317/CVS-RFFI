import json
from pathlib import Path

import numpy as np

from cvsrffi.stage2_nested_registration_builder import (
    build_nested_registration_slices,
)


def test_nested_builder_emits_full_query_and_truth_isolated_slices(tmp_path: Path) -> None:
    classes = 26
    support_labels = np.repeat(np.arange(classes, dtype=np.int64), 10)
    support_ranks = np.tile(np.arange(10, dtype=np.int64), classes)
    support_ids = np.asarray([f"s-{i}" for i in range(classes * 10)])
    support_iq = np.zeros((classes * 10, 2, 256), dtype=np.float32)
    support_iq[:, 0, 0] = np.arange(classes * 10)
    shuffle = np.random.default_rng(7).permutation(classes * 10)
    np.savez(
        tmp_path / "support.npz",
        support_pool_leo_weak_iq=support_iq[shuffle],
        support_pool_class_indices=support_labels[shuffle],
        support_pool_rank_within_class=support_ranks[shuffle],
        support_pool_tokens=support_ids[shuffle],
    )
    query_labels = np.repeat(np.arange(classes, dtype=np.int64), 20)
    query_ids = np.asarray([f"q-{i}" for i in range(classes * 20)])
    np.savez(
        tmp_path / "query.npz",
        query_leo_weak_iq=np.zeros((classes * 20, 2, 256), dtype=np.float32),
        query_tokens=query_ids,
    )
    sidecar = {
        "schema": "cvs.phase2.query_truth_sidecar.v2",
        "rows": [
            {
                "query_token": token,
                "scenario": "leo_clear_weak",
                "true_class_index": int(label),
            }
            for token, label in zip(query_ids.tolist(), query_labels.tolist())
        ],
    }
    (tmp_path / "truth.json").write_text(json.dumps(sidecar), encoding="utf-8")

    audit = build_nested_registration_slices(
        max_support_path=tmp_path / "support.npz",
        max_query_path=tmp_path / "query.npz",
        truth_sidecar_path=tmp_path / "truth.json",
        scenario="leo_clear_weak",
        output_root=tmp_path / "nested",
        capsule_id="capsule",
        da_split_id="split-clear",
        base_checkpoint_path="/base.pth",
    )

    assert audit["status"] == "VALIDATED_ONCE"
    assert [row["query_rows"] for row in audit["rows"]] == [140, 160, 180, 220, 320, 420, 520]
    with np.load(tmp_path / "nested/old_support.npz", allow_pickle=False) as old:
        assert old["received_iq"].shape == (60, 2, 256)
        assert old["support_labels"].tolist() == np.repeat(np.arange(6), 10).tolist()
    with np.load(
        tmp_path / "nested/new15/predictor/support_leo_clear_weak.npz",
        allow_pickle=False,
    ) as support:
        assert support["received_iq"].shape == (210, 2, 256)
        assert set(support.files) == {
            "received_iq",
            "support_labels",
            "support_physical_ids",
        }
        with np.load(tmp_path / "nested/old_support.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(
                old["received_iq"], support["received_iq"][:60]
            )
            np.testing.assert_array_equal(
                old["support_labels"], support["support_labels"][:60]
            )
            np.testing.assert_array_equal(
                old["support_physical_ids"], support["support_physical_ids"][:60]
            )
    with np.load(
        tmp_path / "nested/new15/predictor/query_leo_clear_weak.npz",
        allow_pickle=False,
    ) as query:
        assert query["received_iq"].shape == (420, 2, 256)
        assert "query_labels" not in query.files
    with np.load(
        tmp_path / "nested/new15/scorer/truth_leo_clear_weak.npz",
        allow_pickle=False,
    ) as truth:
        assert truth["query_labels"].shape == (420,)
