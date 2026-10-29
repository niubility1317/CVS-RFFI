"""Synthetic checks for U-only reciprocal labels and anonymous registration."""
import importlib
import json
import unittest

import numpy as np


class ReciprocalCoreTests(unittest.TestCase):
    def core(self):
        try:
            return importlib.import_module(
                "experiments.pamf_phase3_open_world.star_reciprocal_core"
            )
        except ModuleNotFoundError as exc:
            self.fail(f"reciprocal core is not implemented: {exc}")

    @staticmethod
    def ring(n=16, offset=0, radius=0.1):
        angles = np.arange(n) * (2 * np.pi / n)
        x = np.zeros((n, 160), np.float32)
        x[:, offset] = np.sqrt(1 - radius * radius)
        x[:, offset + 1] = radius * np.cos(angles)
        x[:, offset + 2] = radius * np.sin(angles)
        return x

    def test_graph_excludes_self_and_resolves_equal_similarity_by_opaque_id(self):
        m = self.core()
        x = np.zeros((12, 160), np.float32)
        x[:, 0] = 1
        ids = np.asarray([f"opaque-{i:02}" for i in range(11, -1, -1)])
        adjacency, neighbors = m.reciprocal_graph(x, ids)
        self.assertEqual(adjacency.dtype, np.bool_)
        self.assertEqual(neighbors.dtype, np.int64)
        self.assertEqual(neighbors.shape, (12, 10))
        self.assertFalse(np.diag(adjacency).any())
        self.assertEqual(ids[neighbors[0]].tolist(), [f"opaque-{i:02}" for i in range(10)])
        self.assertEqual(ids[neighbors[-1]].tolist(), [f"opaque-{i:02}" for i in range(1, 11)])
        self.assertFalse(adjacency[0].any())
        self.assertTrue(adjacency[1:, 1:][~np.eye(11, dtype=bool)].all())
        permutation = np.asarray([4, 0, 11, 7, 1, 10, 6, 2, 9, 3, 8, 5])
        graph2, neighbors2 = m.reciprocal_graph(x[permutation], ids[permutation])
        np.testing.assert_array_equal(graph2, adjacency[np.ix_(permutation, permutation)])
        for new_row, old_row in enumerate(permutation):
            np.testing.assert_array_equal(ids[permutation][neighbors2[new_row]], ids[neighbors[old_row]])

    def test_graph_uses_similarity_order_and_handles_small_pools(self):
        m = self.core()
        x = np.zeros((4, 160), np.float32)
        x[:, :2] = [[1, 0], [0.8, 0.6], [0.6, 0.8], [-1, 0]]
        graph, neighbors = m.reciprocal_graph(x, np.asarray(["a", "b", "c", "d"]))
        np.testing.assert_array_equal(neighbors[0], [1, 2, 3])
        self.assertEqual(int(graph.sum()), 12)
        for n in (0, 1):
            with self.subTest(n=n):
                graph, neighbors = m.reciprocal_graph(x[:n], np.asarray(["a"][:n], dtype="U1"))
                self.assertEqual(graph.shape, (n, n))
                self.assertEqual(neighbors.shape, (n, 0))

    def test_reciprocal_labels_can_correct_argmax_without_assigning_rejected_rows(self):
        m = self.core()
        x = np.zeros((5, 160), np.float32)
        x[:, 0] = 1
        logits = np.zeros((5, 6), np.float32)
        logits[:3, 2] = 10
        logits[3, 1] = 2
        logits[4, 1] = 10
        margins = np.asarray([0, 0.1, 0.2, 0.3, -0.1], np.float64)
        ids = np.asarray(["e", "d", "c", "b", "a"])
        control = m.pseudolabels(x, logits, margins, ids, arm="self_pl")
        main = m.pseudolabels(x, logits, margins, ids, arm="reciprocal")
        np.testing.assert_array_equal(control["labels"], [2, 2, 2, -1, -1])
        np.testing.assert_array_equal(main["labels"], [2, 2, 2, 2, -1])
        np.testing.assert_array_equal(main["trusted"], [True, True, True, False, False])
        np.testing.assert_array_equal(main["expanded"], [False, False, False, True, False])
        np.testing.assert_array_equal(main["selected"], [True, True, True, True, False])
        self.assertEqual(main["targets"].dtype, np.float32)
        self.assertEqual(main["labels"].dtype, np.int64)
        np.testing.assert_array_equal(main["targets"][3], [0, 0, 1, 0, 0, 0])
        self.assertEqual(float(main["targets"][4].sum()), 0)
        self.assertEqual(main["statistics"]["expanded_argmax_changed_count"], 1)
        json.dumps(main["statistics"], allow_nan=False)
        np.testing.assert_array_equal(logits[3], [0, 2, 0, 0, 0, 0])

    def test_expansion_requires_three_trusted_neighbors_and_eighty_percent_vote(self):
        m = self.core()
        for trusted_labels, expected in (([2, 2], -1), ([2, 2, 2, 3], -1), ([2, 2, 2, 2, 3], 2)):
            with self.subTest(trusted_labels=trusted_labels):
                n = len(trusted_labels) + 1
                x = np.zeros((n, 160), np.float32)
                x[:, 0] = 1
                logits = np.zeros((n, 6), np.float32)
                logits[np.arange(n - 1), trusted_labels] = 10
                result = m.pseudolabels(x, logits, np.ones(n, np.float64),
                                        np.asarray([f"x{i}" for i in range(n)]), "reciprocal")
                self.assertEqual(int(result["labels"][-1]), expected)

    def test_registry_uses_all_u_graph_and_orders_components_by_opaque_ids(self):
        m = self.core()
        x = np.concatenate([self.ring(), self.ring(offset=3)])
        ids = np.asarray([f"z{i:02}" for i in range(16)] + [f"a{i:02}" for i in range(16)])
        margin = -np.ones(32, np.float64)
        state, diag = m.fit_registry(x, margin, margin, np.zeros(32, bool), ids)
        self.assertEqual(set(state), {"schema", "centers", "radii"})
        self.assertEqual(state["schema"], m.REGISTRY_SCHEMA)
        self.assertEqual(state["centers"].dtype, np.float64)
        self.assertEqual(state["radii"].dtype, np.float64)
        self.assertEqual(state["centers"].shape, (2, 160))
        self.assertAlmostEqual(float(state["centers"][0, 3]), np.sqrt(0.99), places=6)
        self.assertAlmostEqual(float(state["radii"][0]), 0.1 * 16 / 15, places=7)
        np.testing.assert_array_equal(diag["assignments"], [1] * 16 + [0] * 16)
        np.testing.assert_array_equal(diag["component_sizes"], [16, 16])
        self.assertEqual(diag["registered_count"], 2)
        self.assertTrue(all(not c["reasons"] for c in diag["components"]))
        json.dumps(diag["components"], allow_nan=False)
        p = np.random.default_rng(9).permutation(32)
        state2, diag2 = m.fit_registry(x[p], margin[p], margin[p], np.zeros(32, bool), ids[p])
        np.testing.assert_array_equal(state2["centers"], state["centers"])
        np.testing.assert_array_equal(state2["radii"], state["radii"])
        np.testing.assert_array_equal(diag2["assignments"], diag["assignments"][p])

    def test_registry_rejects_small_boundary_and_unstable_components(self):
        m = self.core()
        fixtures = []
        small = self.ring(15)
        fixtures.append((small, -np.ones(15), -np.ones(15), np.zeros(15, bool),
                         np.asarray([f"s{i:02}" for i in range(15)]), "size_below_minimum"))
        ring = self.ring()
        boundary = np.concatenate([ring, ring[:1]])
        fixtures.append((boundary, np.r_[-np.ones(16), 1.], np.r_[-np.ones(16), 1.],
                         np.asarray([False] * 16 + [True]), np.asarray([f"b{i:02}" for i in range(17)]),
                         "trusted_boundary_exceeds_limit"))
        unstable = np.zeros((16, 160), np.float32)
        unstable[::2, 0] = 1
        unstable[1::2, 1] = 1
        fixtures.append((unstable, -np.ones(16), -np.ones(16), np.zeros(16, bool),
                         np.asarray([f"h{i:02}" for i in range(16)]), "unstable_halves"))
        for x, current, original, trusted, ids, reason in fixtures:
            with self.subTest(reason=reason):
                state, diag = m.fit_registry(x, current.astype(np.float64), original.astype(np.float64), trusted, ids)
                self.assertEqual(state["centers"].shape, (0, 160))
                self.assertEqual(state["radii"].shape, (0,))
                self.assertTrue((diag["assignments"] == -1).all())
                self.assertTrue(any(reason in c["reasons"] for c in diag["components"]))
        current = -np.ones(16, np.float64)
        original = np.ones(16, np.float64)
        state, diag = m.fit_registry(ring, current, original, np.zeros(16, bool),
                                     np.asarray([f"o{i}" for i in range(16)]))
        self.assertEqual(diag["registered_count"], 0)
        self.assertFalse(diag["candidate_mask"].any())

    def test_empty_registry_and_radius_floor_are_legal(self):
        m = self.core()
        empty = np.empty((0, 160), np.float32)
        state, diag = m.fit_registry(empty, np.empty(0, np.float64), np.empty(0, np.float64),
                                     np.empty(0, bool), np.empty(0, dtype="U1"))
        self.assertEqual(state["centers"].shape, (0, 160))
        self.assertEqual(diag["component_labels"].shape, (0,))
        x = self.ring(radius=1e-8)
        state, _ = m.fit_registry(x, -np.ones(16, np.float64), -np.ones(16, np.float64),
                                  np.zeros(16, bool), np.asarray([f"r{i:02}" for i in range(16)]))
        np.testing.assert_array_equal(state["radii"], [1e-6])

    def test_inference_preserves_old_gate_and_uses_anonymous_only_after_rejection(self):
        m = self.core()
        x = np.zeros((4, 160), np.float32)
        x[:3, 0] = 1
        x[3, 1] = 1
        logits = np.zeros((4, 6), np.float32)
        logits[:, 2] = 30
        margins = np.asarray([0, 0.2, -0.2, -0.2], np.float64)
        center = np.zeros((1, 160), np.float64)
        center[0, 0] = 1
        state = {"schema": m.REGISTRY_SCHEMA, "centers": center, "radii": np.asarray([0.5], np.float64)}
        pred, closed, knownness, joint = m.infer(x, logits, margins, state, registered=True)
        np.testing.assert_array_equal(pred, [2, 2, 6, -1])
        np.testing.assert_array_equal(closed, [2, 2, 2, 2])
        np.testing.assert_allclose(knownness, [0, 0.2, 1, -0.2], atol=1e-15)
        np.testing.assert_array_equal(joint[:2, 6], [-2, -2])
        np.testing.assert_array_equal(joint[0, :6], [-1, -1, 0, -1, -1, -1])
        plain = m.infer(x, logits, margins, state, registered=False)
        np.testing.assert_array_equal(plain[0], [2, 2, -1, -1])
        self.assertEqual(plain[3].shape, (4, 6))
        for i in range(4):
            one = m.infer(x[i:i + 1], logits[i:i + 1], margins[i:i + 1], state, True)
            for batch_value, single_value in zip((pred, closed, knownness, joint), one):
                np.testing.assert_array_equal(batch_value[i:i + 1], single_value)
        no_registry = {"schema": m.REGISTRY_SCHEMA, "centers": np.empty((0, 160), np.float64),
                       "radii": np.empty(0, np.float64)}
        np.testing.assert_array_equal(m.infer(x, logits, margins, no_registry, True)[0], plain[0])

    def test_invalid_shapes_nonfinite_ids_and_state_fail_at_public_boundary(self):
        m = self.core()
        x = np.zeros((2, 160), np.float32)
        x[:, 0] = 1
        ids = np.asarray(["a", "b"])
        bad_unit = x.copy()
        bad_unit[0, 0] = np.nan
        for bad_x, bad_ids in ((x.astype(np.float64), ids), (x[:, :3], ids), (bad_unit, ids),
                               (x * 2, ids), (x, np.asarray(["a", "a"])),
                               (x, np.asarray(["a", "b"], dtype=object))):
            with self.subTest(unit_shape=bad_x.shape, ids_dtype=str(bad_ids.dtype)):
                with self.assertRaises(ValueError):
                    m.reciprocal_graph(bad_x, bad_ids)
        logits = np.zeros((2, 6), np.float32)
        margin = np.zeros(2, np.float64)
        with self.assertRaises(ValueError):
            m.pseudolabels(x, logits, margin, ids, "unsupported")
        with self.assertRaises(ValueError):
            m.pseudolabels(x, logits[:, :5], margin, ids)
        with self.assertRaises(ValueError):
            m.fit_registry(x, margin, margin, np.zeros(2, np.int64), ids)
        invalid_state = {"schema": m.REGISTRY_SCHEMA, "centers": np.zeros((1, 160), np.float64),
                         "radii": np.zeros(1, np.float64)}
        with self.assertRaises(ValueError):
            m.infer(x, logits, margin, invalid_state, True)


if __name__ == "__main__":
    unittest.main()
