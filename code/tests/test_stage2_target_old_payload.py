import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays


def test_payload_uses_target_old_role_for_old_leo_queries():
    features = []
    tx_ids = []
    roles = []

    def add(tx, role, base, count):
        for idx in range(count):
            features.append([base, float(idx)])
            tx_ids.append(tx)
            roles.append(role)

    add("old0", "source", 0.0, 4)
    add("old1", "source", 1.0, 4)
    add("old0", "target_old", 10.0, 3)
    add("old1", "target_old", 11.0, 3)
    add("new0", "target_new", 20.0, 3)
    add("unk0", "target_new", 30.0, 3)

    payload = build_sfe_payload_from_feature_arrays(
        features=np.asarray(features, dtype=np.float32),
        tx_ids=np.asarray(tx_ids, dtype=str),
        dataset_roles=np.asarray(roles, dtype=str),
        source_tx_ids="old0,old1",
        target_old_tx_ids="old0,old1",
        new_tx_ids="new0",
        unknown_tx_ids="unk0",
        source_proto_per_tx=2,
        source_query_per_tx=2,
        target_old_query_per_tx=2,
        shots=1,
        query_per_tx=1,
        seed=7,
    )

    arrays = payload.arrays
    manifest = payload.manifest

    old_query_tx_ids = arrays["query_tx_ids"][:4].tolist()
    old_query_features = arrays["query_features"][:4, 0].tolist()

    assert old_query_tx_ids.count("old0") == 2
    assert old_query_tx_ids.count("old1") == 2
    assert all(value in {10.0, 11.0} for value in old_query_features)
    assert manifest["target_old_tx_ids"] == ["old0", "old1"]
    assert manifest["counts"]["target_old_query"] == 4


def test_payload_can_split_target_old_support_from_query():
    features = []
    tx_ids = []
    roles = []

    def add(tx, role, base, count):
        for idx in range(count):
            features.append([base, float(idx)])
            tx_ids.append(tx)
            roles.append(role)

    add("old0", "source", 0.0, 4)
    add("old1", "source", 1.0, 4)
    add("old0", "target_old", 10.0, 4)
    add("old1", "target_old", 11.0, 4)
    add("new0", "target_new", 20.0, 3)
    add("unk0", "target_new", 30.0, 3)

    payload = build_sfe_payload_from_feature_arrays(
        features=np.asarray(features, dtype=np.float32),
        tx_ids=np.asarray(tx_ids, dtype=str),
        dataset_roles=np.asarray(roles, dtype=str),
        source_tx_ids="old0,old1",
        target_old_tx_ids="old0,old1",
        new_tx_ids="new0",
        unknown_tx_ids="unk0",
        source_proto_per_tx=2,
        target_old_support_per_tx=1,
        target_old_query_per_tx=2,
        shots=0,
        query_per_tx=1,
        seed=7,
    )

    arrays = payload.arrays
    manifest = payload.manifest

    assert arrays["support_tx_ids"].tolist() == ["old0", "old1"]
    assert arrays["support_labels"].tolist() == [0, 1]
    assert manifest["counts"]["target_old_support"] == 2
    assert manifest["counts"]["target_old_query"] == 4
    assert manifest["counts"]["support_features"] == 2
    support_query_overlap = [
        overlap
        for name, overlap in manifest["split_overlap_audit"].items()
        if {"target_old_support", "target_old_query"} == set(name.split("__"))
    ]
    assert support_query_overlap == [[]]
