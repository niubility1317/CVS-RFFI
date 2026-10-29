"""Synthetic reconstructed joint-OVA scores and truth-only structural guard."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class JointOvaScoreTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_ova_score')

    def save(self,m,folder,k,u):
        with np.load(folder/'thresholds.npz') as z:tau=z['conditional']
        for name,data in (('known',k),('novel',u)):
            raw,c=m.evaluation.scalars(data['default_logits'],data['aux_logits'])
            data['closed']=c;data['scores']=raw-tau[:,c].T;np.savez(folder/(name+'_predictions.npz'),**data)

    def fixture(self,root,arm='joint',primary_fail=False,degrade=False):
        m=self.module();rows=[(c,r,s) for c in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        y=np.asarray([r[0] for r in rows],np.int64);z=np.full((n,6),-2.,np.float32);g=np.full((n,6),-1.,np.float32)
        z[np.arange(n),y]=2.;g[np.arange(n),y]=1.
        k=dict(ids=np.asarray(['k'+str(i) for i in range(n)]),y=y,rx_i=np.asarray([r[1] for r in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.asarray([r[2] for r in rows]),default_logits=z,aux_logits=g,baseline_closed=y.copy())
        rows=[(t,r,s) for t in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        z=np.full((n,6),-2.,np.float32);z[:,0]=-1.;g=np.full((n,6),-1.,np.float32)
        if primary_fail:g[:,1]=1.
        unit=np.zeros((n,160),np.float32)
        for i,row in enumerate(rows):unit[i,m.UNKNOWN_TX.index(row[0])]=1.
        current=unit.copy()
        if degrade:current[:]=0.;current[:,0]=1.
        u=dict(ids=np.asarray(['u'+str(i).zfill(3) for i in range(n)]),rx_i=np.asarray([r[1] for r in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),default_logits=z,aux_logits=g,candidate_fused_unit=current,baseline_fused_unit=unit)
        truth=dict(ids=u['ids'][::-1],tx=np.asarray([r[0] for r in rows])[::-1],rx_i=u['rx_i'][::-1],day_i=u['day_i'][::-1],scenario=np.asarray([r[2] for r in rows])[::-1])
        folder=root/'prediction';folder.mkdir();(folder/'calibration').mkdir()
        np.savez(folder/'thresholds.npz',**{'global':np.zeros(3,np.float64),'conditional':np.zeros((3,6),np.float64)})
        self.save(m,folder,k,u)
        identity=dict(run_id=m.RUN_ID,code_commit='a'*40,arm=arm,updates=2220,cumulative_updates=13320,
            encoder_updates=2220,identity_head_updates=2220,auxiliary_updates=2220,model_parameters=232993,aux_parameters=966,
            identity_head_replaced=False,input_path='cached_P1_only')
        marker=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,run_id=m.RUN_ID,code_commit='a'*40,arm=arm,
            method_names=list(m.METHODS),primary_method='ova_max',primary_arm='joint',classification_path='P1',score_threshold=0.,
            model_identity=identity,source_only=True,target_reads=0,source_novel_truth_reads=0,source_OE_records_consumed=0,
            source_L_records_consumed=0,eval_neural_updates=0,model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True,
            all_thresholds_frozen_before_novel=True,original_F0_loaded_after_all_thresholds=True,known_count=len(k['ids']),novel_count=n,
            baseline='cache original F0 same-input P1 closed; never candidate closed')
        (folder/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,arm=arm,
            method_names=list(m.METHODS),conditional_thresholds=np.zeros((3,6)).tolist(),global_thresholds=[0.]*3,group_count=42,
            known_only=True,unknown_inputs_used=False,source_V_neural_fit=False,
            methods={name:dict(constraints_passed=True,conditional_thresholds=[0.]*6,groups={g:{} for g in m.GROUP_NAMES}) for name in m.METHODS})
        (folder/'calibration/complete.json').write_text(json.dumps(cal),encoding='utf-8')
        tp=root/'truth.npz';np.savez(tp,**truth);return folder,tp,k,u,truth

    def test_joint_primary_and_control_evaluate_independently_not_control_promotion(self):
        m=self.module()
        for arm in ('joint','detach_control'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
                root=Path(td);v=self.fixture(root,arm);done=m.score(v[0],v[1],root/'score')
                self.assertTrue(done['arm_source_gate_passed']);self.assertEqual(done['eligible_primary'],arm=='joint')
                self.assertEqual(done['next_step'],'SOURCE_JOINT_OVA_READY_FOR_TARGET_FALSIFICATION' if arm=='joint' else 'NO_PROMOTION')
                self.assertTrue(done['novel_structure_guard']['passed']);self.assertIsNone(done['A_N']);self.assertIsNone(done['H_OW'])
                self.assertEqual(len(done['methods']['ova_max']['groups']),42)

    def test_better_routed_method_does_not_replace_failed_max_primary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root,primary_fail=True);done=m.score(v[0],v[1],root/'score')
            self.assertTrue(done['methods']['ova_routed']['goals']['passed']);self.assertFalse(done['eligible_primary'])

    def test_original_baseline_is_not_replaced_when_candidate_head_is_wrong(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,k,u,t=self.fixture(root);c=(k['y']+1)%6
            k['default_logits'][:]=-2.;k['default_logits'][np.arange(len(c)),c]=2.
            k['aux_logits'][:]=-1.;k['aux_logits'][np.arange(len(c)),c]=1.;self.save(m,folder,k,u)
            done=m.score(folder,tp,root/'score');row=done['primary_summary']
            self.assertEqual(row['original_F0_macro_closed_accuracy'],1.);self.assertEqual(row['candidate_macro_closed_accuracy'],0.)
            self.assertEqual(row['known_macro_open_accuracy'],0.);self.assertFalse(done['arm_source_gate_passed'])

    def test_novel_crossRX_structure_drop_blocks_otherwise_good_primary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root,degrade=True);done=m.score(v[0],v[1],root/'score')
            self.assertTrue(done['methods']['ova_max']['goals']['passed']);self.assertFalse(done['novel_structure_guard']['passed'])
            self.assertAlmostEqual(done['novel_structure_guard']['macro_drop_from_original'],2/3)
            self.assertFalse(done['eligible_primary'])
            self.assertTrue(done['novel_cross_RX_retrieval']['candidate']['exclude_same_RX'])
            self.assertTrue(done['novel_cross_RX_retrieval']['original_F0']['not_future_registration_accuracy'])

    def test_complete_three_scalar_logits_thresholds_features_before_truth(self):
        m=self.module()
        for defect in ('marker','calibration','aux_identity','margin','tau','unit','private_novel','freeze','overlap'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);folder,tp,k,u,t=self.fixture(root)
                if defect=='marker':(folder/'complete.json').unlink()
                elif defect=='calibration':(folder/'calibration/complete.json').unlink()
                elif defect=='freeze':
                    p=folder/'complete.json';d=json.loads(p.read_text());d['all_thresholds_frozen_before_novel']=False;p.write_text(json.dumps(d))
                elif defect=='tau':np.savez(folder/'thresholds.npz',**{'global':np.zeros(3,np.float64),'conditional':np.ones((3,6),np.float64)})
                else:
                    if defect=='aux_identity':u['aux_logits'][:,1]=3.;u['closed'][:]=1
                    elif defect=='margin':u['scores'][0,2]+=1
                    elif defect=='unit':u['baseline_fused_unit'][0]*=2
                    elif defect=='private_novel':u['tx']=np.asarray(['19-15']*len(u['ids']))
                    else:u['ids'][0]=k['ids'][0]
                    np.savez(folder/'novel_predictions.npz',**u)
                with patch.object(m,'_truth',side_effect=AssertionError('truth touched')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(folder,tp,root/'score')

    def test_missing_coverage_is_NA_and_scoring_never_changes_predictions(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,tp,k,u,t=self.fixture(root);t['tx'][t['tx']=='19-15']='19-16';np.savez(tp,**t)
            files=[folder/'known_predictions.npz',folder/'novel_predictions.npz',folder/'thresholds.npz'];before=[p.read_bytes() for p in files]
            done=m.score(folder,tp,root/'score');self.assertIsNone(done['primary_summary']['unknown_macro_rejection'])
            self.assertFalse(done['eligible_primary']);self.assertEqual(before,[p.read_bytes() for p in files])
            with self.assertRaises(FileExistsError):m.score(folder,tp,root/'score')


if __name__=='__main__':unittest.main()
