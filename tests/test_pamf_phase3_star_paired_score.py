"""Synthetic five-method paired-evidence truth-last scoring contracts."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarPairedScoreTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_paired_score')

    def fixture(self,root,main_pass=True):
        m=self.module();rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        y=np.array([r[0] for r in rows],np.int64);n=len(rows)
        known=dict(ids=np.array(['k'+str(i) for i in range(n)]),y=y,rx_i=np.array([r[1] for r in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.array([r[2] for r in rows]),
            scores=np.ones((n,5),np.float64),closed=y.copy(),baseline_closed=y.copy())
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        scores=np.full((n,5),-1.,np.float64)
        if not main_pass:scores[:,-1]=1
        novel=dict(ids=np.array(['u'+str(i) for i in range(n)]),rx_i=np.array([r[1] for r in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scores=scores,closed=np.zeros(n,np.int64))
        truth=dict(ids=novel['ids'][::-1],tx=np.array([r[0] for r in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.array([r[2] for r in rows])[::-1])
        prediction=root/'source';(prediction/'predict').mkdir(parents=True);(prediction/'calibration').mkdir()
        np.savez(prediction/'predict/known_predictions.npz',**known);np.savez(prediction/'predict/novel_predictions.npz',**novel)
        np.save(prediction/'thresholds.npy',np.zeros(5,np.float64))
        marker=dict(schema=m.PREDICTION_SCHEMA,status=m.PREDICTION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),primary_method=m.PRIMARY,classification_path='P1',rejection_input_paths=['P1','P0'],
            control_p0_zeroed=True,known_count=len(y),novel_count=n,source_only=True,target_reads=0,source_novel_truth_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,model_parameters_buffers_unchanged=True,
            L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz')
        (prediction/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        calibration=dict(schema=m.PREDICTION_SCHEMA,status=m.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),group_count=42,known_only=True,unknown_inputs_used=False,
            methods={name:dict(threshold=0.,constraints_passed=True) for name in m.METHODS})
        (prediction/'calibration/complete.json').write_text(json.dumps(calibration),encoding='utf-8')
        tp=root/'truth.npz';np.savez(tp,**truth)
        return prediction,tp,known,novel,truth

    def test_complete_five_methods_42groups_and_source_only_primary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,*_=self.fixture(root);done=m.score(prediction,tp,root/'score')
            self.assertTrue(done['source_gate_passed']);self.assertEqual(done['next_step'],'SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION')
            self.assertIsNone(done['A_N']);self.assertIsNone(done['H_OW']);self.assertFalse(done['final_phase3_complete'])
            self.assertEqual(tuple(done['methods']),m.METHODS);self.assertEqual(len(done['paired_comparisons']),5)
            for row in done['methods'].values():
                self.assertEqual(len(row['groups']),42);self.assertEqual(len(row['by_known_TX']),6);self.assertEqual(len(row['by_unknown_TX']),3)
                self.assertAlmostEqual(row['overall']['H_OS'],1.)
            with self.assertRaises(FileExistsError):m.score(prediction,tp,root/'score')

    def test_good_200_and_p1_controls_never_replace_failed_paired1000(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,*_=self.fixture(root,main_pass=False);done=m.score(prediction,tp,root/'score')
            self.assertFalse(done['source_gate_passed']);self.assertEqual(done['next_step'],'NO_PROMOTION')
            self.assertTrue(done['methods']['paired_mlp16_step200']['goals']['passed'])
            self.assertTrue(done['methods']['p1_mlp16_step1000']['goals']['passed'])

    def test_all_marker_shape_and_threshold_checks_precede_truth(self):
        m=self.module()
        for defect in ('no_complete','no_calibration','wrong_primary','wrong_path','scores_3D','private_novel','bad_count','nan','bad_threshold','duplicate'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);prediction,tp,k,u,t=self.fixture(root)
                if defect=='no_complete':(prediction/'complete.json').unlink()
                elif defect=='no_calibration':(prediction/'calibration/complete.json').unlink()
                elif defect in ('wrong_primary','wrong_path','bad_count'):
                    p=prediction/'complete.json';marker=json.loads(p.read_text())
                    key,value={'wrong_primary':('primary_method',m.METHODS[1]),'wrong_path':('classification_path','P0'),'bad_count':('novel_count',1)}[defect]
                    marker[key]=value;p.write_text(json.dumps(marker))
                elif defect=='bad_threshold':np.save(prediction/'thresholds.npy',np.ones(5,np.float64))
                else:
                    if defect=='scores_3D':u['scores']=u['scores'][:,None,:]
                    elif defect=='private_novel':u['tx']=np.array(['hidden']*len(u['ids']))
                    elif defect=='nan':u['scores'][0,0]=np.nan
                    else:u['ids'][0]=u['ids'][1]
                    np.savez(prediction/'predict/novel_predictions.npz',**u)
                with patch.object(m,'_truth',side_effect=AssertionError('truth touched')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(prediction,tp,root/'score')

    def test_input_and_budget_pairing_directions_and_correct_identity_counts(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root)
            k['closed'][0]=1;k['baseline_closed'][0]=1  # Wrong accepted identity must not become a correct gain.
            k['scores'][0,1]=-1;k['scores'][1:3,2]=-1;k['scores'][:3,4]=-1
            u['scores'][:]=1
            for j,n in ((1,10),(2,20),(3,30),(4,40)):u['scores'][:n,j]=-1
            np.savez(prediction/'predict/known_predictions.npz',**k);np.savez(prediction/'predict/novel_predictions.npz',**u)
            done=m.score(prediction,tp,root/'score');p=done['paired_comparisons']
            expected={'input_step200':(-1,-2,10),'input_step1000':(-3,-2,10),
                'budget_p1_only':(1,0,20),'budget_paired':(-1,0,20),'primary_vs_energy':(-3,-2,40)}
            for key,(accept,correct,rejected) in expected.items():
                with self.subTest(comparison=key):
                    row=p[key];c=row['counts'];self.assertEqual(c['known_accept_net'],accept)
                    self.assertEqual(c['known_correct_accept_net'],correct);self.assertEqual(c['unknown_rejected_net'],rejected)
                    self.assertAlmostEqual(row['overall']['known_macro_open_accuracy'],correct/180)
                    self.assertAlmostEqual(row['overall']['unknown_macro_rejection'],rejected/90)
                    self.assertEqual(len(row['groups']),42)

    def test_v3_known_acceptance90_and_original_F0_identity_protection(self):
        m=self.module();row=dict(coverage=dict(known_complete=True,unknown_complete=True,complete=True),
            known_macro_acceptance=.9,original_closed_minus_new_open=.02,unknown_macro_rejection=.9)
        groups={key:copy.deepcopy(row) for key in m.GROUP_NAMES};self.assertTrue(m._goals(groups,True)['passed'])
        for field,value in [('known_macro_acceptance',.89999),('original_closed_minus_new_open',.02001),('unknown_macro_rejection',.89999)]:
            with self.subTest(field=field):
                bad=copy.deepcopy(groups);bad['overall'][field]=value;self.assertFalse(m._goals(bad,True)['passed'])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,k,u,t=self.fixture(root);k['baseline_closed'][0]=1
            np.savez(prediction/'predict/known_predictions.npz',**k)
            with patch.object(m,'_truth',side_effect=AssertionError('bad original F0 baseline')),self.assertRaises(ValueError):m.score(prediction,tp,root/'score')

    def test_missing_coverage_is_NA_with_unknown_RXscene_diagnostic_only(self):
        m=self.module()
        for kind in ('unknown_cell','unknown_full','known_cell'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);prediction,tp,k,u,t=self.fixture(root)
                if kind=='unknown_full':t['tx'][t['tx']=='19-15']='19-16'
                elif kind=='unknown_cell':t['tx'][(t['tx']=='19-15')&(t['rx_i']==1)&(t['scenario']==m.SCENES[0])]='19-16'
                else:k['y'][(k['y']==0)&(k['rx_i']==1)&(k['scenario']==m.SCENES[0])]=1
                np.savez(tp,**t);np.savez(prediction/'predict/known_predictions.npz',**k)
                done=m.score(prediction,tp,root/'score');self.assertEqual(done['source_gate_passed'],kind=='unknown_cell')
                row=done['methods'][m.PRIMARY]['overall' if kind=='unknown_full' else 'groups']
                if kind!='unknown_full':row=row['rxscene/1/'+m.SCENES[0]]
                self.assertIsNone(row['H_OS'])

    def test_scorer_never_changes_states_or_thresholds(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prediction,tp,*_=self.fixture(root)
            files=[prediction/'predict/known_predictions.npz',prediction/'predict/novel_predictions.npz',prediction/'thresholds.npy']
            before=[p.read_bytes() for p in files];m.score(prediction,tp,root/'score');self.assertEqual(before,[p.read_bytes() for p in files])


if __name__=='__main__':unittest.main()
