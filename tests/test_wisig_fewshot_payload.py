import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class WiSigFewShotPayloadTest(unittest.TestCase):
    def test_build_sfe_payload_remaps_nonoverlap_tx_and_keeps_unknown(self):
        from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays

        tx_ids = []
        features = []
        for tx in ["old_a", "old_b", "new_c", "unk_d"]:
            for i in range(6):
                tx_ids.append(tx)
                features.append([len(tx_ids), i])
        payload = build_sfe_payload_from_feature_arrays(
            features=np.asarray(features, dtype=np.float32),
            tx_ids=np.asarray(tx_ids),
            source_tx_ids=["old_a", "old_b"],
            new_tx_ids=["new_c"],
            unknown_tx_ids=["unk_d"],
            shots=2,
            source_proto_per_tx=2,
            source_query_per_tx=1,
            query_per_tx=2,
            seed=7,
        )

        arrays = payload.arrays
        self.assertEqual(arrays["source_features"].shape[0], 4)
        self.assertEqual(arrays["support_features"].shape[0], 2)
        self.assertEqual(arrays["query_features"].shape[0], 6)
        self.assertEqual(sorted(set(arrays["source_labels"].tolist())), [0, 1])
        self.assertEqual(sorted(set(arrays["support_labels"].tolist())), [2])
        self.assertIn(-1, arrays["query_labels"].tolist())
        self.assertEqual(payload.manifest["overlap_audit"]["source_new"], [])
        self.assertEqual(payload.manifest["counts"]["query_unknown"], 2)
        self.assertIn("source_sample_indices", arrays)
        self.assertIn("split_overlap_audit", payload.manifest)
        self.assertTrue(all(v == [] for v in payload.manifest["split_overlap_audit"].values()))

    def test_build_sfe_payload_uses_nonoverlapping_role_splits_per_tx(self):
        from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays

        tx_ids = []
        features = []
        for tx in ["old_a", "new_b", "unk_c"]:
            for i in range(12):
                tx_ids.append(tx)
                features.append([i, 0.0])

        payload = build_sfe_payload_from_feature_arrays(
            features=np.asarray(features, dtype=np.float32),
            tx_ids=np.asarray(tx_ids),
            source_tx_ids=["old_a"],
            new_tx_ids=["new_b"],
            unknown_tx_ids=["unk_c"],
            shots=3,
            source_proto_per_tx=3,
            source_query_per_tx=3,
            query_per_tx=3,
            seed=13,
        )

        audit = payload.manifest["split_overlap_audit"]
        self.assertTrue(all(overlap == [] for overlap in audit.values()))

    def test_build_sfe_payload_preserves_source_support_and_query_receiver_metadata(self):
        from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays

        tx_ids = []
        rx_ids = []
        dataset_roles = []
        features = []
        for role, tx, rx in [
            ("source", "old_a", "rx0"),
            ("target_old", "old_a", "rx7"),
            ("target_new", "new_b", "rx7"),
            ("target_new", "unk_c", "rx7"),
        ]:
            for i in range(8):
                tx_ids.append(tx)
                rx_ids.append(rx)
                dataset_roles.append(role)
                features.append([len(features), i])

        payload = build_sfe_payload_from_feature_arrays(
            features=np.asarray(features, dtype=np.float32),
            tx_ids=np.asarray(tx_ids),
            dataset_roles=np.asarray(dataset_roles),
            sample_metadata={"rx_ids": np.asarray(rx_ids)},
            source_tx_ids=["old_a"],
            target_old_tx_ids=["old_a"],
            new_tx_ids=["new_b"],
            unknown_tx_ids=["unk_c"],
            shots=2,
            source_proto_per_tx=2,
            source_query_per_tx=2,
            target_old_support_per_tx=2,
            query_per_tx=2,
            seed=11,
        )

        self.assertIn("source_rx_ids", payload.arrays)
        self.assertIn("support_rx_ids", payload.arrays)
        self.assertIn("query_rx_ids", payload.arrays)
        self.assertEqual(set(payload.arrays["source_rx_ids"].tolist()), {"rx0"})
        self.assertEqual(set(payload.arrays["support_rx_ids"].tolist()), {"rx7"})
        self.assertEqual(set(payload.arrays["query_rx_ids"].tolist()), {"rx7"})

    def test_build_sfe_payload_prefers_target_unknown_role_for_unknown_query(self):
        from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays

        tx_ids = []
        dataset_roles = []
        features = []
        for role, tx in [
            ("source", "old_a"),
            ("target_new", "new_b"),
            ("target_unknown", "unk_c"),
        ]:
            for i in range(8):
                tx_ids.append(tx)
                dataset_roles.append(role)
                features.append([len(features), i])

        payload = build_sfe_payload_from_feature_arrays(
            features=np.asarray(features, dtype=np.float32),
            tx_ids=np.asarray(tx_ids),
            dataset_roles=np.asarray(dataset_roles),
            source_tx_ids=["old_a"],
            new_tx_ids=["new_b"],
            unknown_tx_ids=["unk_c"],
            shots=2,
            source_proto_per_tx=2,
            source_query_per_tx=2,
            query_per_tx=2,
            seed=11,
        )

        arrays = payload.arrays
        self.assertEqual(set(arrays["support_tx_ids"].tolist()), {"new_b"})
        self.assertNotIn("unk_c", arrays["support_tx_ids"].tolist())
        self.assertEqual(
            set(zip(arrays["query_tx_ids"].tolist(), arrays["query_roles"].tolist(), arrays["query_dataset_roles"].tolist())),
            {("old_a", "source_query", "source"), ("new_b", "new_query", "target_new"), ("unk_c", "unknown_query", "target_unknown")},
        )

    def test_build_sfe_payload_rejects_tx_overlap(self):
        from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays

        features = np.ones((8, 2), dtype=np.float32)
        tx_ids = np.asarray(["a"] * 4 + ["b"] * 4)
        with self.assertRaisesRegex(ValueError, "source_new"):
            build_sfe_payload_from_feature_arrays(
                features=features,
                tx_ids=tx_ids,
                source_tx_ids=["a", "b"],
                new_tx_ids=["b"],
                shots=1,
                source_proto_per_tx=1,
                source_query_per_tx=1,
                query_per_tx=1,
            )

    def test_build_sfe_payload_rejects_insufficient_samples(self):
        from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays

        features = np.ones((4, 2), dtype=np.float32)
        tx_ids = np.asarray(["a", "a", "b", "b"])
        with self.assertRaisesRegex(ValueError, "not enough samples"):
            build_sfe_payload_from_feature_arrays(
                features=features,
                tx_ids=tx_ids,
                source_tx_ids=["a"],
                new_tx_ids=["b"],
                shots=2,
                source_proto_per_tx=1,
                source_query_per_tx=1,
                query_per_tx=1,
            )


if __name__ == "__main__":
    unittest.main()
