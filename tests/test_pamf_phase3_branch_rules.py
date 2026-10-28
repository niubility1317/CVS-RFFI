import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from experiments.pamf_phase3_open_world import branch_rules
except ImportError:
    branch_rules = None

from experiments.pamf_phase3_open_world import source_geometry


BRANCHES = (
    ('fused160', 160), ('time160', 160), ('frequency160', 160), ('pa160', 160),
    ('reference29', 29), ('signature336', 336), ('spectrum128', 128), ('pa_moments48', 48),
)


def fixture(root):
    rng = np.random.default_rng(1457)
    y = np.repeat(np.arange(4, dtype=np.int64), 4)
    vy = np.tile(np.arange(6, dtype=np.int64), 2)
    logits = np.eye(4, dtype=np.float32)[vy % 4] * 8
    # The full head deliberately misclassifies one known observation.  Branch
    # distances must never replace that class prediction.
    logits[0] = np.asarray([0, 8, 0, 0], np.float32)
    arrays = dict(train_y=y, train_ids=np.asarray([f'L{i}' for i in range(len(y))]),
        validation_logits=logits, validation_y=vy,
        validation_ids=np.asarray([f'V{i}' for i in range(len(vy))]),
        validation_receivers=np.repeat(np.asarray([1, 8], np.int64), 6))
    for name, d in BRANCHES:
        train = (np.eye(d)[y] + rng.normal(0, .01, (len(y), d))).astype(np.float32)
        validation = np.eye(d)[vy].astype(np.float32)
        if name == 'reference29':
            train[:, 27] = validation[:, 27] = 0
        arrays['train_features__' + name] = train
        arrays['validation_features__' + name] = validation
    np.savez(root / 'source_branches.npz', **arrays)
    marker = dict(status='SOURCE_BRANCHES_EXPORTED', source_only=True,
        target_reads=0, optimizer_updates=0, head_changed=False,
        parent_epochs=20, parent_updates=540, train_count=len(y),
        validation_count=len(vy), heldout_source_rx=8)
    (root / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    return arrays, marker


class BranchRuleTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(branch_rules, 'Source branch rules are not implemented')

    def test_training_only_location_scale_and_constant_reference_coordinate(self):
        rng = np.random.default_rng(4)
        y = np.repeat(np.arange(4, dtype=np.int64), 3)
        x = rng.normal(size=(12, 29)).astype(np.float32)
        x[:, 27] = 0
        bundle = branch_rules.fit_branch(x, y, standardize=True)
        np.testing.assert_allclose(bundle['location'], x.astype(float).mean(axis=0))
        expected = x.astype(float).std(axis=0, ddof=0)
        expected[expected <= 1e-8] = 1
        np.testing.assert_allclose(bundle['scale'], expected)
        self.assertEqual(bundle['location'][27], 0)
        self.assertEqual(bundle['scale'][27], 1)
        before = {k: v.copy() for k, v in bundle.items()}
        branch_rules.branch_scores(np.full((2, 29), 99., np.float32), bundle)
        for key in before:
            np.testing.assert_array_equal(bundle[key], before[key])

    def test_fused_cosine_and_mahalanobis_match_previous_s020_baseline(self):
        rng = np.random.default_rng(10)
        y = np.repeat(np.arange(4, dtype=np.int64), 3)
        x = (np.eye(160)[y] + rng.normal(0, .02, (len(y), 160))).astype(np.float32)
        z = (np.eye(160)[np.arange(6)] + .001).astype(np.float32)
        old = source_geometry.fit_geometry(x, y)
        old_scores = source_geometry.geometry_scores(z, np.zeros((6, 4)), old)
        new = branch_rules.fit_branch(x, y, standardize=False)
        self.assertTrue((new['location'] == 0).all())
        self.assertTrue((new['scale'] == 1).all())
        np.testing.assert_allclose(new['prototypes'], old['prototypes'], atol=1e-10)
        np.testing.assert_allclose(new['whitener'], old['whiteners'][1], atol=1e-8)
        np.testing.assert_allclose(branch_rules.branch_scores(z, new),
            old_scores[:, [source_geometry.NAMES.index('cosine'),
                           source_geometry.NAMES.index('mahalanobis_s020')]], atol=1e-7)

    def test_scores_are_row_local_and_class_permutation_invariant(self):
        rng = np.random.default_rng(51)
        y = np.repeat(np.arange(4, dtype=np.int64), 4)
        x = rng.normal(size=(16, 29)).astype(np.float32)
        z = rng.normal(size=(5, 29)).astype(np.float32)
        bundle = branch_rules.fit_branch(x, y, standardize=True)
        scores = branch_rules.branch_scores(z, bundle)
        np.testing.assert_allclose(scores[:1], branch_rules.branch_scores(z[:1], bundle), atol=1e-8)
        alternate = branch_rules.fit_branch(x, np.argsort([2, 0, 3, 1])[y], standardize=True)
        np.testing.assert_allclose(scores, branch_rules.branch_scores(z, alternate), atol=1e-8)

    def test_mahalanobis_cholesky_matches_direct_residual_distance(self):
        rng = np.random.default_rng(19)
        y = np.repeat(np.arange(4, dtype=np.int64), 3)
        x = rng.normal(size=(12, 8)).astype(np.float32)
        bundle = branch_rules.fit_branch(x, y, standardize=True)
        z = rng.normal(size=(3, 8)).astype(np.float32)
        unit = (z - bundle['location']) / bundle['scale']
        unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
        residual = unit[:, None, :] - bundle['prototypes'][None, :, :]
        expected = -np.square(residual @ bundle['whitener'].T).sum(axis=2).min(axis=1)
        np.testing.assert_allclose(branch_rules.branch_scores(z, bundle)[:, 1], expected, atol=1e-8)

    def test_proxy_unknown_scores_never_calibrate_thresholds(self):
        known = np.arange(100, dtype=float)[:, None] + np.arange(16)[None, :]
        first = np.vstack([known, np.full((2, 16), -1e6)])
        second = np.vstack([known, np.full((2, 16), 1e6)])
        labels = np.r_[np.arange(100) % 4, [4, 5]].astype(np.int64)
        a = branch_rules.calibrate_known(first, labels)
        np.testing.assert_array_equal(a, branch_rules.calibrate_known(second, labels))
        np.testing.assert_array_equal(a, np.quantile(known, .01, axis=0, method='lower'))

    def test_incomplete_or_non_source_marker_never_opens_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, marker = fixture(root)
            cases = [dict(marker, status='EXPORTING'), dict(marker, target_reads=1),
                     dict(marker, target_reads=False), dict(marker, source_only=False),
                     dict(marker, optimizer_updates=1), dict(marker, head_changed=True),
                     dict(marker, parent_epochs=19), dict(marker, parent_updates=539),
                     dict(marker, heldout_source_rx=6)]
            for changed in cases:
                (root / 'complete.json').write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(changed=changed), patch.object(branch_rules.np, 'load',
                    side_effect=AssertionError('NPZ opened before completed source export')):
                    with self.assertRaises(ValueError):
                        branch_rules.evaluate(root / 'source_branches.npz', root / 'out')

    def test_evaluate_exports_fixed16_rules_only_aggregate_bundle_and_full_head_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays, _ = fixture(root)
            result = branch_rules.evaluate(root / 'source_branches.npz', root / 'out')
            expected = [f'{name}::{kind}' for name, _ in BRANCHES
                        for kind in ('cosine', 'mahalanobis_s020')]
            self.assertEqual(list(result['methods']), expected)
            self.assertEqual(result['status'], 'SOURCE_BRANCH_RULES_EVALUATED')
            self.assertEqual(result['evidence_scope'], 'source_proxy_only_not_target')
            self.assertFalse(result['final_six_class_method_selected'])
            closed = arrays['validation_logits'].argmax(axis=1)
            known = arrays['validation_y'] < 4
            expected_accuracy = float((closed[known] == arrays['validation_y'][known]).mean())
            for row in result['methods'].values():
                self.assertEqual(row['overall']['known_closed_accuracy'], expected_accuracy)
                self.assertEqual(set(row['by_receiver']), {'1', '8'})
            with np.load(root / 'out' / 'bundle.npz', allow_pickle=False) as saved:
                expected_keys = {'thresholds', 'method_names'} | {
                    f'{stat}__{name}' for name, _ in BRANCHES
                    for stat in ('location', 'scale', 'prototypes', 'whitener')}
                self.assertEqual(set(saved.files), expected_keys)
                self.assertEqual(saved['thresholds'].shape, (16,))
                self.assertEqual(saved['method_names'].tolist(), expected)
                for name, d in BRANCHES:
                    self.assertEqual(saved['prototypes__' + name].shape, (4, d))
                    self.assertEqual(saved['whitener__' + name].shape, (d, d))
            marker = json.loads((root / 'out' / 'complete.json').read_text(encoding='utf-8'))
            self.assertEqual(marker['status'], result['status'])
            self.assertTrue(marker['source_only'])
            self.assertEqual(marker['target_reads'], 0)
            with self.assertRaises(FileExistsError):
                branch_rules.evaluate(root / 'source_branches.npz', root / 'out')

    def test_feature_schema_shape_float_labels_and_physical_ids_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays, _ = fixture(root)
            changes = []
            changed = copy.deepcopy(arrays); changed.pop('train_features__time160'); changes.append(changed)
            changed = copy.deepcopy(arrays); changed['train_features__pa160'] = np.ones((16, 159), np.float32); changes.append(changed)
            changed = copy.deepcopy(arrays); changed['train_features__reference29'][0, 0] = np.nan; changes.append(changed)
            changed = copy.deepcopy(arrays); changed['train_features__spectrum128'] = changed['train_features__spectrum128'].astype(int); changes.append(changed)
            changed = copy.deepcopy(arrays); changed['train_y'][0] = 4; changes.append(changed)
            changed = copy.deepcopy(arrays); changed['validation_y'] = changed['validation_y'].astype(bool); changes.append(changed)
            changed = copy.deepcopy(arrays); changed['validation_ids'][0] = changed['train_ids'][0]; changes.append(changed)
            changed = copy.deepcopy(arrays); changed['validation_ids'][1] = changed['validation_ids'][0]; changes.append(changed)
            changed = copy.deepcopy(arrays); changed['validation_receivers'][0] = 0; changes.append(changed)
            changed = copy.deepcopy(arrays); changed['validation_logits'][0, 0] = np.inf; changes.append(changed)
            for i, changed in enumerate(changes):
                np.savez(root / 'source_branches.npz', **changed)
                with self.subTest(case=i), self.assertRaises(ValueError):
                    branch_rules.evaluate(root / 'source_branches.npz', root / f'bad{i}')
                self.assertFalse((root / f'bad{i}').exists())

    def test_degenerate_covariance_or_bad_bundle_fails_without_nonfinite_scores(self):
        y = np.repeat(np.arange(4, dtype=np.int64), 2)
        with self.assertRaises(ValueError):
            branch_rules.fit_branch(np.zeros((8, 29), np.float32), y, standardize=True)
        x = np.eye(29)[y].astype(np.float32)
        bundle = branch_rules.fit_branch(x, y, standardize=False)
        self.assertTrue(np.isfinite(bundle['whitener']).all())
        bundle['scale'][0] = 0
        with self.assertRaises(ValueError):
            branch_rules.branch_scores(x, bundle)

    def test_all_reject_and_empty_groups_cannot_promote(self):
        rejected = dict(known_count=4, unknown_count=2, unknown_macro_rejection=1.,
            known_macro_frr=1., known_accuracy_drop=1.)
        empty = dict(known_count=0, unknown_count=0, unknown_macro_rejection=None,
            known_macro_frr=None, known_accuracy_drop=None)
        self.assertFalse(branch_rules.eligible(rejected, rejected))
        self.assertFalse(branch_rules.eligible(empty, empty))


if __name__ == '__main__':
    unittest.main()
