import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.phase1_hankel_posterior import audit_full, score_full


class FullSourceScoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source'; self.source.mkdir()
        self.score = self.root/'score.json'; self.audit = self.root/'audit.json'
        ids = np.arange(11761); keys = np.array([f'L{i:05}' for i in range(6300)])
        labels = np.arange(6300) % 6
        meta = self.root/'meta'; meta.mkdir()
        np.savez(meta/'L_s_metadata.npz', ids=keys, y=labels)
        contract = self.root/'contract.json'; contract.write_text(json.dumps({'role_ids': {'L_s': keys.tolist()}}))
        complete = self.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json'
        complete.parent.mkdir(parents=True); complete.write_text(json.dumps({'config': {'source_cache': str(meta), 'source_contract': str(contract)}}))
        cache = self.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train'; cache.mkdir(parents=True)
        physical = np.array([f'L{i:05}' if i < 1047 else f'U{i:05}' for i in ids])
        roles = np.where(ids < 1047, 'L_s', 'U_s')
        np.save(cache/'physical_ids.npy', physical); np.save(cache/'roles.npy', roles)
        views = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban', 'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
        scene = np.array([views[i % 7] for i in ids]); rx = np.where((ids//7) % 2 == 0, 4, 8)
        old = np.zeros((len(ids), 15), dtype=np.float32)
        old[:, 0] = ids % 6; old[:, 1] = .99; old[:, 2] = ids % 6; old[:, 3] = .9
        old[:, 5] = ids % 6; old[:, 7] = (ids+1) % 6
        old[:, 4] = .1; old[:, 6] = .2; old[:, 8] = .3
        new = np.column_stack((ids % 6, np.full(len(ids), .25))).astype(np.float32)
        np.savez(self.source/'source_predictions.npz', indices=ids, rx=rx, scene=scene, prior_values=old, hankel_values=new)
        m = dict(status='SOURCE_PREDICTIONS_FIXED', commit='fixed', count=len(ids), pilot_replayed=672,
                 DURA_loaded=False, TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates=0)
        (self.source/'prediction_complete.json').write_text(json.dumps(m))
        prior = self.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01'; prior.mkdir(parents=True)
        np.savez(prior/'source_predictions.npz', indices=ids, rx=rx, scene=scene, values=old)

    def tearDown(self): self.temp.cleanup()

    def test_truth_last_scores_and_independent_audit(self):
        with patch.object(sys, 'argv', ['score_full', '--root', str(self.root), '--source', str(self.source), '--out', str(self.score), '--commit', 'fixed']):
            score_full.main()
        with patch.object(sys, 'argv', ['audit_full', '--root', str(self.root), '--source', str(self.source), '--score', str(self.score), '--out', str(self.audit)]):
            audit_full.main()
        result = json.loads(self.audit.read_text())
        self.assertEqual((result['status'], result['L_s_scored'], result['fields_recomputed']), ('PASS', 1047, 736))
        scored = json.loads(self.score.read_text()); scored['results']['RX8']['all_impaired']['L_s']['hankel']['repairs'] += 1
        self.score.write_text(json.dumps(scored))
        with patch.object(sys, 'argv', ['audit_full', '--root', str(self.root), '--source', str(self.source), '--score', str(self.score), '--out', str(self.root/'tampered.json')]):
            with self.assertRaisesRegex(ValueError, 'counts differ'): audit_full.main()

    def test_unfixed_predictions_stop_before_identity_metadata(self):
        (self.source/'prediction_complete.json').write_text(json.dumps({'status': 'PARTIAL'}))
        (self.root/'meta/L_s_metadata.npz').unlink()
        with patch.object(sys, 'argv', ['score_full', '--root', str(self.root), '--source', str(self.source), '--out', str(self.score), '--commit', 'fixed']):
            with self.assertRaisesRegex(ValueError, 'before truth'): score_full.main()


if __name__ == '__main__': unittest.main()
