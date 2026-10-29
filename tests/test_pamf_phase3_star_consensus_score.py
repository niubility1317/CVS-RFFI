"""Synthetic source-only consensus scoring, without any IQ or model IO."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarConsensusScoreTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_consensus_score')

    def fixture(self,root,main_pass=True):
        m=self.module();rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        y=np.asarray([r[0] for r in rows],np.int64);n=len(rows)
        known=dict(ids=np.array(['k'+str(i) for i in range(n)]),y=y,rx_i=np.array([r[1] for r in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.array([r[2] for r in rows]),
            scores=np.ones((n,2,7),np.float64),closed=np.column_stack((y,y)),baseline_closed=np.column_stack((y,y)))
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        scores=np.full((n,2,7),-1.,np.float64)
        if not main_pass:scores[:,1,6]=1
        novel=dict(ids=np.array(['u'+str(i) for i in range(n)]),rx_i=np.array([r[1] for r in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scores=scores,closed=np.zeros((n,2),np.int64))
        truth=dict(ids=novel['ids'][::-1],tx=np.array([r[0] for r in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.array([r[2] for r in rows])[::-1])
        prediction=root/'source';prediction.mkdir();(prediction/'predict').mkdir()
        for role,data in [('known',known),('novel',novel)]:np.savez(prediction/'predict'/(role+'_predictions.npz'),**data)
        for path in m.PATHS:
            folder=prediction/'by_path'/path;(folder/'calibration').mkdir(parents=True)
            np.save(folder/'thresholds.npy',np.zeros(7,np.float64))
            (folder/'calibration/complete.json').write_text(json.dumps(dict(status=m.CALIBRATION_STATUS,schema=m.PREDICTION_SCHEMA,
                run_id=m.RUN_ID,code_commit='a'*40,path=path,method_names=list(m.METHODS),group_count=42,
                known_only=True,unknown_inputs_used=False,methods={name:dict(threshold=0.,constraints_passed=True) for name in m.METHODS})),encoding='utf-8')
        marker=dict(status=m.PREDICTION_STATUS,schema=m.PREDICTION_SCHEMA,run_id=m.RUN_ID,code_commit='a'*40,
            path_names=list(m.PATHS),method_names=list(m.METHODS),primary_path='P1',primary_method='mlp16_oe',
            known_count=len(y),novel_count=n,source_only=True,target_reads=0,source_novel_truth_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,model_parameters_buffers_unchanged=True,
            L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz')
        (prediction/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        truth_path=root/'truth.npz';np.savez(truth_path,**truth)
        return prediction,truth_path,known,novel,truth

    def test_all_methods_42groups_macro_metrics_and_fixed_primary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,truth,*_=self.fixture(root);done=m.score(prediction,truth,root/'score')
            self.assertTrue(done['source_gate_passed']);self.assertEqual(done['next_step'],'SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION')
            self.assertIsNone(done['A_N']);self.assertIsNone(done['H_OW']);self.assertFalse(done['final_phase3_complete'])
            for path in m.PATHS:
                methods=done['paths'][path]['methods'];self.assertEqual(tuple(methods),m.METHODS)
                for row in methods.values():
                    self.assertEqual(len(row['groups']),42);self.assertEqual(len(row['by_known_TX']),6);self.assertEqual(len(row['by_unknown_TX']),3)
                    self.assertAlmostEqual(row['overall']['H_OS'],1.)
            self.assertEqual(done['paired_primary_vs_energy']['overall']['known_macro_open_accuracy'],0.)
            with self.assertRaises(FileExistsError):m.score(prediction,truth,root/'score')

    def test_good_P0_linear_static_cannot_replace_failed_P1_MLP(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,truth,*_=self.fixture(root,main_pass=False);done=m.score(prediction,truth,root/'score')
            self.assertFalse(done['source_gate_passed']);self.assertEqual(done['next_step'],'NO_PROMOTION')
            self.assertTrue(done['paths']['P0']['methods']['mlp16_oe']['goals']['passed'])
            self.assertTrue(done['paths']['P1']['methods']['linear_oe']['goals']['passed'])
            self.assertEqual(done['paired_primary_vs_energy']['overall']['unknown_macro_rejection'],-1.)

    def test_all_prediction_checks_happen_before_truth_access(self):
        m=self.module()
        for defect in ('no_complete','missing_P1','private_novel','nan_score','duplicate_ID','wrong_count','threshold_mismatch','wrong_run'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);prediction,truth,k,u,t=self.fixture(root)
                if defect=='no_complete':(prediction/'complete.json').unlink()
                elif defect=='missing_P1':(prediction/'by_path/P1/calibration/complete.json').unlink()
                elif defect=='private_novel':u['tx']=np.array(['hidden']*len(u['ids']));np.savez(prediction/'predict/novel_predictions.npz',**u)
                elif defect=='nan_score':u['scores'][0,0,0]=np.nan;np.savez(prediction/'predict/novel_predictions.npz',**u)
                elif defect=='duplicate_ID':u['ids'][0]=u['ids'][1];np.savez(prediction/'predict/novel_predictions.npz',**u)
                elif defect=='threshold_mismatch':np.save(prediction/'by_path/P0/thresholds.npy',np.ones(7))
                else:
                    p=prediction/'complete.json';marker=json.loads(p.read_text());marker['known_count' if defect=='wrong_count' else 'run_id']=1 if defect=='wrong_count' else 'old_run';p.write_text(json.dumps(marker))
                with patch.object(m,'_truth',side_effect=AssertionError('Truth opened before prediction validation')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(prediction,truth,root/'score')
                self.assertFalse((root/'score/complete.json').exists())

    def test_v3_CK90_and_original_F0_2pp_boundaries(self):
        m=self.module();row=dict(coverage=dict(known_complete=True,unknown_complete=True,complete=True),
            known_macro_acceptance=.9,known_macro_frr=.1,original_closed_minus_new_open=.02,unknown_macro_rejection=.9)
        groups={key:copy.deepcopy(row) for key in m.GROUP_NAMES}
        self.assertTrue(m._goals(groups,True)['passed'])
        for field,value in [('known_macro_acceptance',.899999),('original_closed_minus_new_open',.020001),('unknown_macro_rejection',.899999)]:
            with self.subTest(field=field):
                bad=copy.deepcopy(groups);bad['overall'][field]=value
                self.assertFalse(m._goals(bad,True)['passed'])
        self.assertFalse(m._goals(groups,False)['passed'])

    def test_missing_coverage_is_NA_and_unknown_RXscene_only_diagnostic(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root)
            mask=(t['tx']=='19-15')&(t['rx_i']==1)&(t['scenario']==m.SCENES[0]);t['tx'][mask]='19-16';np.savez(tp,**t)
            done=m.score(prediction,tp,root/'score');self.assertTrue(done['source_gate_passed'])
            row=done['paths']['P1']['methods']['mlp16_oe']['groups']['rxscene/1/'+m.SCENES[0]]
            self.assertIsNone(row['unknown_macro_rejection']);self.assertIsNone(row['H_OS'])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root)
            t['tx'][t['tx']=='19-15']='19-16';np.savez(tp,**t)
            done=m.score(prediction,tp,root/'score');self.assertFalse(done['source_gate_passed'])
            self.assertIsNone(done['paths']['P1']['methods']['mlp16_oe']['overall']['H_OS'])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root)
            mask=(k['y']==0)&(k['rx_i']==1)&(k['scenario']==m.SCENES[0]);k['y'][mask]=1
            np.savez(prediction/'predict/known_predictions.npz',**k)
            done=m.score(prediction,tp,root/'score');self.assertFalse(done['source_gate_passed'])
            row=done['paths']['P1']['methods']['mlp16_oe']['groups']['rxscene/1/'+m.SCENES[0]]
            self.assertIsNone(row['known_macro_open_accuracy']);self.assertFalse(row['coverage']['known_complete'])

    def test_known_group_identity_protection_and_frozen_original_baseline(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root)
            mask=(k['y']==0)&(k['rx_i']==1)&(k['scenario']==m.SCENES[0]);k['scores'][mask,1,6]=-1
            np.savez(prediction/'predict/known_predictions.npz',**k);done=m.score(prediction,tp,root/'score')
            self.assertFalse(done['source_gate_passed'])
            self.assertGreater(done['paths']['P1']['methods']['mlp16_oe']['groups']['rxscene/1/'+m.SCENES[0]]['original_closed_minus_new_open'],.02)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root);k['baseline_closed']=(k['baseline_closed']+1)%6
            np.savez(prediction/'predict/known_predictions.npz',**k)
            with patch.object(m,'_truth',side_effect=AssertionError('invalid F0 baseline before truth')),self.assertRaises(ValueError):m.score(prediction,tp,root/'score')

    def test_scoring_never_modifies_predictions_or_source_thresholds(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,*_=self.fixture(root)
            files=[prediction/'predict/known_predictions.npz',prediction/'predict/novel_predictions.npz',prediction/'by_path/P1/thresholds.npy']
            before=[p.read_bytes() for p in files];m.score(prediction,tp,root/'score')
            self.assertEqual(before,[p.read_bytes() for p in files])


if __name__=='__main__':unittest.main()
