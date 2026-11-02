"""Exercise complete source scoring with only synthetic, legally bound labels."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.phase1_dura_posterior import score_source


class SourceScoreIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source_result'
        self.source.mkdir()
        cache = self.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
        train = cache/'train'
        (cache/'epochs/19').mkdir(parents=True)
        train.mkdir(parents=True)
        physical = np.arange(63000, dtype=np.int64)
        roles = np.full(63000, 'U_s', dtype='<U3')
        roles[:6300] = 'L_s'
        rx = np.zeros(63000, dtype=np.int64)
        day = np.zeros(63000, dtype=np.int64)
        scene = np.full(63000, 'clean', dtype='<U24')
        ids = np.arange(6000, 6672, dtype=np.int64)
        split = np.asarray(['RX4_6']*336+['RX8']*336)
        rx[ids[:336]] = 4
        rx[ids[336:]] = 8
        day[ids] = 3
        for group in range(2):
            for index, view in enumerate(score_source.VIEWS):
                selected = ids[group*336+index*48:group*336+(index+1)*48]
                scene[selected] = view
        for name, array in (('physical_ids', physical), ('roles', roles), ('rx', rx), ('day', day)):
            np.save(train/(name+'.npy'), array, allow_pickle=False)
        np.savez(cache/'epochs/19/groups.npz', scene=scene)
        label_root = self.root/'source_metadata'
        label_root.mkdir()
        label_ids = physical[:6300]
        label_values = label_ids % 6
        np.savez(label_root/'L_s_metadata.npz', ids=label_ids, y=label_values)
        contract = self.root/'source_contract.json'
        contract.write_text(json.dumps({'role_ids': {'L_s': label_ids.tolist()}}), encoding='utf8')
        f_source = self.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source'
        f_source.mkdir(parents=True)
        (f_source/'complete.json').write_text(json.dumps({'config': {'source_contract': str(contract),
                                                                      'source_cache': str(label_root)}}), encoding='utf8')
        values = np.zeros((672, 9), dtype=np.float32)
        values[:, 0] = .99
        values[:, 1] = ids % 6
        values[:, 2] = np.arange(len(ids)) % 2
        values[:, 3] = values[:, 5] = values[:, 7] = ids % 6
        values[:, 4] = values[:, 6] = values[:, 8] = .1
        values[0, 7] = (values[0, 5]+1) % 6
        np.savez(self.source/'source_predictions.npz', indices=ids, split=split,
                 rx=rx[ids], scene=scene[ids], values=values)
        manifest = dict(status='SOURCE_PREDICTIONS_FIXED', commit='fixed', count=672, fit_ids=8192,
                        F_frozen=True, DURA_frozen=True, gate_frozen=True, posterior_frozen=True,
                        TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0,
                        model_updates_after_prediction=0, posterior_kind='mean', upstream_L_s_fit_labels_used=0)
        (self.source/'prediction_complete.json').write_text(json.dumps(manifest), encoding='utf8')
        torch.save(dict(schema='phase1_dura_posterior_source_v1', commit='fixed', epochs=5,
                        posterior_kind='mean', upstream_L_s_fit_labels_used=0,
                        fit_ids=np.arange(50000, 58192, dtype=np.int64)), self.source/'checkpoint.pt')
        self.output = self.root/'result/source_score.json'

    def tearDown(self):
        self.tmp.cleanup()

    def run_score(self):
        argv = ['score_source', '--root', str(self.root), '--source', str(self.source),
                '--out', str(self.output), '--commit', 'fixed']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), contextlib.redirect_stdout(io.StringIO()):
            score_source.main()

    def test_valid_fixed_source_predictions_connect_only_legal_labels(self):
        self.run_score()
        result = json.loads(self.output.read_text(encoding='utf8'))
        self.assertEqual(result['L_s_scored'], 300)
        self.assertEqual(result['U_s_truth_reads'], 0)
        self.assertEqual(result['results']['RX4_6']['clean']['L_s']['damages'], 1)
        self.assertEqual(result['results']['RX8']['all_impaired']['L_s']['count'], 0)

    def test_missing_prediction_manifest_fails_before_identity_metadata(self):
        (self.source/'prediction_complete.json').unlink()
        (self.root/'source_metadata/L_s_metadata.npz').unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_score()
        self.assertIn('prediction_complete.json', str(caught.exception))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
