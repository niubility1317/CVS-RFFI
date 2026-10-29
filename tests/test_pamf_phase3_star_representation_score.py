"""Completed synthetic predictions only; scorer truth cannot reach fitting."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class RepresentationScoreTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_representation_score')

    def fixture(self,root,arm='oe_vic'):
        m=self.module();s=m.evaluation
        rows=[(y,r,scene) for y in range(6) for r in s.RECEIVERS for scene in s.SCENES]
        y=np.array([x[0] for x in rows],np.int64);n=len(y)
        known=dict(ids=np.array(['k'+str(i) for i in range(n)]),y=y,rx_i=np.array([x[1] for x in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.array([x[2] for x in rows]),scores=np.ones((n,2,7),np.float64),
            closed=np.column_stack((y,y)),baseline_closed=np.column_stack((y,y)))
        rows=[(tx,r,scene) for tx in m.UNKNOWN_TX for r in s.RECEIVERS for scene in s.SCENES];n=len(rows)
        features=np.zeros((n,2,160),np.float32)
        for i,(tx,_,_) in enumerate(rows):features[i,:,m.UNKNOWN_TX.index(tx)]=1
        novel=dict(ids=np.array(['u'+str(i) for i in range(n)]),rx_i=np.array([x[1] for x in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scores=np.full((n,2,7),-1.,np.float64),
            closed=np.zeros((n,2),np.int64),fused_unit=features)
        truth=dict(ids=novel['ids'][::-1],tx=np.array([x[0] for x in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.array([x[2] for x in rows])[::-1])
        prediction=root/'prediction';prediction.mkdir()
        s._npz(prediction/'known_predictions.npz',known);s._npz(prediction/'novel_predictions.npz',novel)
        for path in s.PATHS:
            row=prediction/'by_path'/path;(row/'calibration').mkdir(parents=True)
            np.save(row/'thresholds.npy',np.zeros(7))
            s._write(row/'calibration/complete.json',dict(status=s.CALIBRATION_STATUS,schema=s.SCHEMA,run_id=s.RUN_ID,code_commit='a'*40,
                arm=arm,path=path,method_names=list(s.METHODS),group_count=42,source_only=True,target_reads=0,
                source_V_metric_fit=False,unknown_inputs_used=False,methods={name:dict(threshold=0.,constraints_passed=True) for name in s.METHODS}))
        marker=dict(status=s.STATUS,schema=s.SCHEMA,run_id=s.RUN_ID,code_commit='a'*40,arm=arm,path_names=list(s.PATHS),
            method_names=list(s.METHODS),primary_method='energy',primary_path='P1',source_only=True,target_reads=0,
            source_novel_truth_reads=0,source_OE_records_consumed=0,eval_neural_updates=0,model_parameters_buffers_unchanged=True,
            L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,known_count=len(y),novel_count=n,
            baseline='cache original F0 same-input same-path complete head')
        s._write(prediction/'complete.json',marker);tp=root/'truth.npz';s._npz(tp,truth)
        return prediction,tp,known,novel,truth

    def test_truthlast_all_methods_groups_metrics_and_retrieval(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,truth,*_=self.fixture(root)
            done=m.score(prediction,truth,root/'score')
            self.assertTrue(done['arm_source_gate_passed']);self.assertTrue(done['eligible_primary'])
            self.assertEqual(done['next_step'],'SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION')
            self.assertIsNone(done['A_N']);self.assertIsNone(done['H_OW']);self.assertFalse(done['final_phase3_complete'])
            for path in ('P0','P1'):
                self.assertEqual(len(done['paths'][path]['methods']),7)
                self.assertEqual(len(done['paths'][path]['methods']['energy']['groups']),42)
                self.assertEqual(len(done['paths'][path]['methods']['energy']['by_known_TX']),6)
                self.assertAlmostEqual(done['novel_cross_RX_retrieval'][path]['macro_hit_rate'],1.)

    def test_good_diagnostic_arm_cannot_replace_primary_arm(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,truth,*_=self.fixture(root,arm='anchor')
            done=m.score(prediction,truth,root/'score')
            self.assertTrue(done['arm_source_gate_passed']);self.assertFalse(done['eligible_primary'])
            self.assertEqual(done['next_step'],'NO_PROMOTION')

    def test_incomplete_or_private_predictions_block_truth(self):
        m=self.module()
        for defect in ('missing_marker','private_novel','missing_path','wrong_baseline_marker'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);prediction,truth,k,u,t=self.fixture(root)
                if defect=='missing_marker':(prediction/'complete.json').unlink()
                elif defect=='private_novel':u['tx']=np.array(['hidden']*len(u['ids']));np.savez(prediction/'novel_predictions.npz',**u)
                elif defect=='missing_path':(prediction/'by_path/P1/calibration/complete.json').unlink()
                else:
                    p=prediction/'complete.json';marker=json.loads(p.read_text());marker['baseline']='self closed';p.write_text(json.dumps(marker))
                with patch.object(m,'_truth',side_effect=AssertionError('private truth touched')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(prediction,truth,root/'score')

    def test_original_F0_identity_damage_and_missing_roster_fail(self):
        m=self.module()
        for defect in ('new_closed_damage','missing_known','missing_novel'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);prediction,truth,k,u,t=self.fixture(root)
                if defect=='new_closed_damage':k['closed']=(k['closed']+1)%6
                elif defect=='missing_known':k['y'][k['y']==0]=1
                else:t['tx'][t['tx']=='19-15']='19-16';np.savez(truth,**t)
                np.savez(prediction/'known_predictions.npz',**k)
                done=m.score(prediction,truth,root/'score')
                self.assertFalse(done['arm_source_gate_passed'])
                if defect=='missing_novel':self.assertIsNone(done['novel_cross_RX_retrieval']['P1']['macro_hit_rate'])
                if defect=='new_closed_damage':self.assertAlmostEqual(done['paths']['P1']['methods']['energy']['overall']['original_closed_minus_new_open'],1.)

    def test_retrieval_excludes_own_RX_and_breaks_exact_ties_by_ID(self):
        m=self.module();features=np.array([[1,0],[1,0],[1,0],[0,1]],np.float32)
        ids=np.array(['query','z-other','a-other','own-rx']);rx=np.array([1,3,3,1]);tx=np.array(['T','T','wrong','T'])
        result=m.cross_rx_retrieval(features,ids,rx,tx)
        self.assertEqual(result['by_record'][0]['neighbor_id'],'a-other')
        self.assertEqual(result['by_record'][3]['neighbor_id'],'a-other')
        self.assertFalse(result['by_record'][0]['hit']);self.assertTrue(result['exclude_same_RX'])

    def test_scoring_does_not_change_saved_predictions_or_thresholds(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,truth,*_=self.fixture(root)
            paths=[prediction/'known_predictions.npz',prediction/'novel_predictions.npz',prediction/'by_path/P0/thresholds.npy']
            before=[p.read_bytes() for p in paths];m.score(prediction,truth,root/'score')
            self.assertEqual(before,[p.read_bytes() for p in paths])


if __name__=='__main__':unittest.main()
