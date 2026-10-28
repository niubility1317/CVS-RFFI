import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments.pamf_phase3_open_world import source_geometry as geometry


def sample():
    centers=np.eye(160)[:4]
    y=np.repeat(np.arange(4),2)
    x=centers[y].copy();x[:,9]=np.tile([.01,-.01],4)
    return x,y


class SourceGeometryTests(unittest.TestCase):
    def test_geometry_is_finite_row_local_and_class_permutation_invariant(self):
        x,y=sample();bundle=geometry.fit_geometry(x,y)
        logits=np.eye(4)[y]*8
        scores=geometry.geometry_scores(x,logits,bundle)
        self.assertEqual(scores.shape,(8,8));self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores[:1],geometry.geometry_scores(x[:1],logits[:1],bundle))
        permutation=np.array([2,0,3,1]);remap=np.argsort(permutation)
        alternate=geometry.fit_geometry(x,remap[y])
        np.testing.assert_allclose(scores,geometry.geometry_scores(x,logits[:,permutation],alternate),atol=1e-7)

    def test_covariance_uses_only_training_residuals_and_handles_low_rank(self):
        x,y=sample();first=geometry.fit_geometry(x,y);again=geometry.fit_geometry(x,y)
        np.testing.assert_array_equal(first['whiteners'],again['whiteners'])
        self.assertEqual(first['whiteners'].shape,(3,160,160))
        self.assertTrue(np.isfinite(first['whiteners']).all())
        self.assertFalse(np.array_equal(first['whiteners'][0],first['whiteners'][2]))
        with self.assertRaises(ValueError):geometry.fit_geometry(x,np.full(len(x),-1))

    def test_calibration_thresholds_do_not_use_proxy_unknown_scores(self):
        known=np.repeat(np.arange(100,dtype=float)[:,None],8,axis=1)
        first=np.vstack([known,np.full((3,8),-999.)]);second=np.vstack([known,np.full((3,8),999.)])
        labels=np.r_[np.arange(100)%4,[4,5,5]]
        np.testing.assert_array_equal(geometry.calibrate_known(first,labels),geometry.calibrate_known(second,labels))

    def test_completion_is_required_before_opening_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'complete.json').write_text(json.dumps(dict(status='TRAINING')),encoding='utf-8')
            with patch.object(geometry.np,'load',side_effect=AssertionError('opened early')):
                with self.assertRaises(ValueError):geometry.evaluate(root/'source_features.npz',root/'rules',8)

    def test_frozen_source_evaluation_exports_all_rules_without_target_io(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);x,y=sample();z=np.vstack([np.eye(160)[:4],np.eye(160)[10:12]])
            v_y=np.arange(6,dtype=np.int64);logits=np.zeros((6,4));logits[:4]=np.eye(4)*10
            np.savez(root/'source_features.npz',train_features=x.astype(np.float32),train_y=y.astype(np.int64),
                train_ids=np.asarray(['L'+str(i) for i in range(len(x))]),validation_features=z.astype(np.float32),
                validation_logits=logits.astype(np.float32),validation_y=v_y,
                validation_ids=np.asarray(['V'+str(i) for i in range(len(z))]),validation_receivers=np.full(6,8,np.int64))
            marker=dict(status='SOURCE_DEVELOPMENT_FROZEN',source_only=True,target_reads=0,U_reads=0,
                checkpoint_initialization='scratch',epochs=20,updates=540,train_count=8,validation_count=6,heldout_source_rx=8)
            (root/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
            result=geometry.evaluate(root/'source_features.npz',root/'rules',8)
            self.assertEqual(len(result['methods']),8)
            self.assertTrue(all(row['eligible'] for row in result['methods'].values()))
            self.assertEqual(result['status'],'SOURCE_RULES_EVALUATED')
            self.assertEqual(result['evidence_scope'],'source_proxy_only_not_target')
            with self.assertRaises(FileExistsError):geometry.evaluate(root/'source_features.npz',root/'rules',8)

    def test_all_reject_cannot_promote(self):
        row=dict(known_count=4,unknown_count=2,unknown_macro_rejection=1.,known_macro_frr=1.,known_accuracy_drop=1.)
        self.assertFalse(geometry.eligible(row,row))


if __name__=='__main__':unittest.main()
