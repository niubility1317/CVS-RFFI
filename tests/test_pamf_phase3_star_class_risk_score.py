"""Synthetic truth-last contracts for predicted-class source calibration."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarClassRiskScoreTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_class_risk_score')

    def save_predictions(self, folder, known, novel, global_tau=0., conditional=None):
        conditional=np.zeros(6,np.float64) if conditional is None else np.asarray(conditional,np.float64)
        for name,data in (('known',known),('novel',novel)):
            raw=data.pop('_raw',None)
            if raw is not None:data['scores']=np.column_stack((raw-global_tau,raw-conditional[data['closed']])).astype(np.float64)
            np.savez(folder/'predict'/(name+'_predictions.npz'),**data)

    def fixture(self, root, main_pass=True):
        m=self.module();rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        y=np.asarray([row[0] for row in rows],np.int64);n=len(rows)
        known=dict(ids=np.asarray(['k'+str(i) for i in range(n)]),y=y,rx_i=np.asarray([row[1] for row in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.asarray([row[2] for row in rows]),
            _raw=np.full(n,3.,np.float64),closed=y.copy(),baseline_closed=y.copy())
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        novel=dict(ids=np.asarray(['u'+str(i) for i in range(n)]),rx_i=np.asarray([row[1] for row in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),_raw=np.full(n,-1.,np.float64),closed=np.zeros(n,np.int64))
        truth=dict(ids=novel['ids'][::-1],tx=np.asarray([row[0] for row in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.asarray([row[2] for row in rows])[::-1])
        folder=root/'prediction';(folder/'predict').mkdir(parents=True);(folder/'calibration').mkdir()
        conditional=np.zeros(6,np.float64) if main_pass else np.full(6,-2.,np.float64)
        self.save_predictions(folder,known,novel,conditional=conditional)
        # Exact on-disk name is 'global', which is a Python reserved word.
        np.savez(folder/'thresholds.npz',**{'global':np.zeros(1,np.float64),'conditional':conditional})
        marker=dict(schema=m.PREDICTION_SCHEMA,status=m.PREDICTION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),primary_method=m.PRIMARY,classification_path='P1',routing='predicted_class',
            use_predicted_class_only=True,score_threshold=0.,source_only=True,target_reads=0,source_novel_truth_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,new_readout_updates=0,all_thresholds_frozen_before_novel=True,
            model_parameters_buffers_unchanged=True,known_count=len(y),novel_count=n,
            parent_run_id='20261029-phase3-star-paired-evidence-s2026092701-r01',parent_code_commit='bcc873cd9f9273133d74b5bc591c947b37ce16e1',
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz')
        (folder/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        calibration=dict(schema=m.PREDICTION_SCHEMA,status=m.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),group_count=42,known_only=True,unknown_inputs_used=False,routing='predicted_class',
            score_threshold=0.,global_threshold=0.,conditional_thresholds=conditional.tolist(),
            methods={name:dict(constraints_passed=True) for name in m.METHODS})
        (folder/'calibration/complete.json').write_text(json.dumps(calibration),encoding='utf-8')
        tp=root/'truth.npz';np.savez(tp,**truth)
        return folder,tp,known,novel,truth

    def test_complete_margin_methods_trueTX_macro_and_source_scope(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,*_=self.fixture(root);done=m.score(folder,tp,root/'score')
            self.assertTrue(done['source_gate_passed']);self.assertEqual(done['next_step'],'SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION')
            self.assertEqual(tuple(done['methods']),m.METHODS);self.assertEqual(done['primary_method'],'predicted_class')
            for row in done['methods'].values():
                self.assertEqual(len(row['groups']),42);self.assertEqual(len(row['by_known_TX']),6);self.assertEqual(len(row['by_unknown_TX']),3)
                self.assertEqual(row['threshold'],0.);self.assertAlmostEqual(row['overall']['H_OS'],1.)
            self.assertIsNone(done['A_N']);self.assertIsNone(done['H_OW']);self.assertFalse(done['final_phase3_complete'])
            self.assertTrue(done['conditional_shift_can_change_ranking']);self.assertFalse(done['conformal_guarantee'])
            with self.assertRaises(FileExistsError):m.score(folder,tp,root/'score')

    def test_passing_global_control_cannot_replace_failed_primary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,*_=self.fixture(root,main_pass=False);done=m.score(folder,tp,root/'score')
            self.assertTrue(done['methods']['global_control']['goals']['passed']);self.assertFalse(done['source_gate_passed'])
            self.assertEqual(done['next_step'],'NO_PROMOTION')

    def test_all_predictions_markers_thresholds_and_routing_checked_before_truth(self):
        m=self.module()
        defects=('no_complete','no_calibration','primary','routing','one_column','nan','bad_conditional','threshold_mismatch',
            'trueTX_margin','baseline','private_novel','duplicate','updates')
        for defect in defects:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);folder,tp,k,u,t=self.fixture(root)
                if defect=='no_complete':(folder/'complete.json').unlink()
                elif defect=='no_calibration':(folder/'calibration/complete.json').unlink()
                elif defect in ('primary','routing','updates'):
                    p=folder/'complete.json';d=json.loads(p.read_text())
                    key,value={'primary':('primary_method','global_control'),'routing':('routing','true_TX'),'updates':('new_readout_updates',1)}[defect]
                    d[key]=value;p.write_text(json.dumps(d))
                elif defect in ('bad_conditional','threshold_mismatch'):
                    np.savez(folder/'thresholds.npz',**{'global':np.zeros(1,np.float64),'conditional':np.ones(5 if defect=='bad_conditional' else 6,np.float64)})
                elif defect=='baseline':
                    k['baseline_closed'][0]=1;np.savez(folder/'predict/known_predictions.npz',**k)
                else:
                    if defect=='one_column':u['scores']=u['scores'][:,:1]
                    elif defect=='nan':u['scores'][0,0]=np.nan
                    elif defect=='trueTX_margin':u['scores'][0,1]+=1
                    elif defect=='private_novel':u['tx']=np.asarray(['19-15']*len(u['ids']))
                    else:u['ids'][0]=u['ids'][1]
                    np.savez(folder/'predict/novel_predictions.npz',**u)
                with patch.object(m,'_truth',side_effect=AssertionError('truth touched')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(folder,tp,root/'score')

    def test_pair_counts_separate_acceptance_and_correct_identity(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,k,u,t=self.fixture(root);conditional=np.asarray([1.,-1.,0.,0.,0.,0.])
            kr=np.full(len(k['ids']),3.);ur=np.full(len(u['ids']),-3.)
            k['closed'][0]=k['baseline_closed'][0]=1;kr[0]=-.5;kr[30]=-.5;kr[1]=.5
            u['closed'][:3]=[0,1,0];ur[:3]=[.5,-.5,.2]
            k['_raw']=kr;u['_raw']=ur;self.save_predictions(folder,k,u,conditional=conditional)
            np.savez(folder/'thresholds.npz',**{'global':np.zeros(1,np.float64),'conditional':conditional})
            p=folder/'calibration/complete.json';d=json.loads(p.read_text());d['conditional_thresholds']=conditional.tolist();p.write_text(json.dumps(d))
            done=m.score(folder,tp,root/'score');pair=done['paired_primary_vs_global'];counts=pair['counts']
            self.assertEqual(counts['known_accept_only_primary'],2);self.assertEqual(counts['known_accept_only_global'],1)
            self.assertEqual(counts['known_accept_net'],1);self.assertEqual(counts['known_correct_accept_net'],0)
            self.assertEqual(counts['novel_reject_only_primary'],2);self.assertEqual(counts['novel_reject_only_global'],1)
            self.assertEqual(counts['unknown_rejected_net'],1);self.assertAlmostEqual(pair['overall']['known_macro_acceptance'],1/180)
            self.assertAlmostEqual(pair['overall']['known_macro_open_accuracy'],0.);self.assertAlmostEqual(pair['overall']['unknown_macro_rejection'],1/90)
            self.assertEqual(len(pair['groups']),42);self.assertEqual(len(pair['group_counts']),42)

    def test_zero_margin_tie_is_accepted_and_trueTX_not_predictedclass_macro(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,k,u,t=self.fixture(root);k['closed'][:]=0;k['baseline_closed'][:]=0
            k['scores'][:]=0.;u['scores'][:]=0.;self.save_predictions(folder,k,u)
            done=m.score(folder,tp,root/'score');row=done['primary_summary']
            self.assertAlmostEqual(row['known_macro_acceptance'],1.);self.assertAlmostEqual(row['known_macro_open_accuracy'],1/6)
            self.assertEqual(row['unknown_macro_rejection'],0.);self.assertFalse(done['source_gate_passed'])
            self.assertEqual(len(done['methods'][m.PRIMARY]['by_known_TX']),6)

    def test_missing_coverage_and_v3_boundary_remain_scientific_fail_or_NA(self):
        m=self.module();row=dict(coverage=dict(known_complete=True,unknown_complete=True,complete=True),
            known_macro_acceptance=.9,original_closed_minus_new_open=.02,unknown_macro_rejection=.9)
        groups={key:copy.deepcopy(row) for key in m.GROUP_NAMES};self.assertTrue(m._goals(groups,True)['passed'])
        for field,value in [('known_macro_acceptance',.89999),('original_closed_minus_new_open',.02001),('unknown_macro_rejection',.89999)]:
            with self.subTest(field=field):
                bad=copy.deepcopy(groups);bad['overall'][field]=value;self.assertFalse(m._goals(bad,True)['passed'])
        for kind in ('unknown_cell','unknown_full','known_cell'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);folder,tp,k,u,t=self.fixture(root)
                if kind=='unknown_full':t['tx'][t['tx']=='19-15']='19-16'
                elif kind=='unknown_cell':t['tx'][(t['tx']=='19-15')&(t['rx_i']==1)&(t['scenario']==m.SCENES[0])]='19-16'
                else:k['y'][(k['y']==0)&(k['rx_i']==1)&(k['scenario']==m.SCENES[0])]=1
                np.savez(tp,**t);self.save_predictions(folder,k,u)
                done=m.score(folder,tp,root/'score');self.assertEqual(done['source_gate_passed'],kind=='unknown_cell')
                group='overall' if kind=='unknown_full' else 'rxscene/1/'+m.SCENES[0]
                self.assertIsNone(done['methods'][m.PRIMARY]['groups'][group]['H_OS'])

    def test_scorer_does_not_change_frozen_margin_or_threshold_state(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,*_=self.fixture(root)
            files=[folder/'thresholds.npz',folder/'predict/known_predictions.npz',folder/'predict/novel_predictions.npz']
            before=[p.read_bytes() for p in files];m.score(folder,tp,root/'score');self.assertEqual(before,[p.read_bytes() for p in files])


if __name__=='__main__':unittest.main()
