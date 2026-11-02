import contextlib
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.phase1_dura_posterior import test_score_source_integration as fixtures
from experiments.phase1_equivariant_dictionary import score_source


class DictionaryScoreTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SourceScoreIntegrationTest('test_valid_fixed_source_predictions_connect_only_legal_labels')
        self.fixture.setUp()
        self.root, self.source, self.output = self.fixture.root, self.fixture.source, self.fixture.output
        manifest_path = self.source/'prediction_complete.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        manifest.update(blind_frozen=True, DURA_loaded=False)
        manifest_path.write_text(json.dumps(manifest), encoding='utf8')
        torch.save(dict(schema='phase1_equivariant_dictionary_v1', commit='fixed', epochs=5,
                        fit_ids=np.arange(50000, 58192, dtype=np.int64)), self.source/'checkpoint.pt')
        with np.load(self.source/'source_predictions.npz', allow_pickle=False) as z:
            candidate = {key: z[key].copy() for key in ('indices', 'rx', 'scene', 'values')}
        ref = self.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01'
        ref.mkdir(parents=True)
        extra = np.arange(10000, 21089, dtype=np.int64)
        ids = np.r_[candidate['indices'], extra]
        values = np.zeros((11761, 15), dtype=np.float32)
        values[:672, 2] = candidate['values'][:, 5]
        values[:672, 3] = np.where(candidate['values'][:, 2] == 1, .9, .1)
        values[:672, 5] = candidate['values'][:, 3]
        values[:672, 7] = candidate['values'][:, 5]
        np.savez(ref/'source_predictions.npz', indices=ids, values=values,
                 rx=np.r_[candidate['rx'], np.zeros(len(extra), dtype=np.int64)],
                 scene=np.r_[candidate['scene'], np.full(len(extra), 'clean')])
        ref_manifest = dict(status='SOURCE_PREDICTIONS_FIXED', count=11761, source_only=True, DURA_comparator_only=True,
                            TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0)
        (ref/'prediction_complete.json').write_text(json.dumps(ref_manifest), encoding='utf8')

    def tearDown(self): self.fixture.tearDown()

    def run_score(self):
        argv = ['score_source', '--root', str(self.root), '--source', str(self.source),
                '--out', str(self.output), '--commit', 'fixed']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), contextlib.redirect_stdout(io.StringIO()):
            score_source.main()

    def test_fixed_predictions_bind_only_authorized_labeled_rows(self):
        self.run_score()
        result = json.loads(self.output.read_text(encoding='utf8'))
        self.assertEqual(result['L_s_scored'], 300)
        self.assertEqual(result['U_s_truth_reads'], 0)
        self.assertEqual(result['results']['RX4_6']['clean']['L_s']['damages'], 1)
        self.assertEqual(result['results']['RX8']['all_impaired']['L_s']['count'], 0)

    def test_truth_is_not_opened_when_predictions_are_incomplete(self):
        (self.source/'prediction_complete.json').unlink()
        (self.root/'source_metadata/L_s_metadata.npz').unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_score()
        self.assertIn('prediction_complete.json', str(caught.exception))
        self.assertFalse(self.output.exists())

    def test_mismatched_raw_baseline_stops_before_opening_identity_metadata(self):
        path = self.source/'source_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            saved = {key: z[key].copy() for key in z.files}
        saved['values'][0, 3] = (saved['values'][0, 3]+1) % 6
        np.savez(path, **saved)
        (self.root/'source_metadata/L_s_metadata.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'Common raw predictions'):
            self.run_score()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
