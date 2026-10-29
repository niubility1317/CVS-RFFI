"""Three frozen policies, exact parent controls and monotone acceptance."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tests import test_pamf_phase3_star_oe_risk_score as old_fixtures


class CommonSeededRiskScoreTests(unittest.TestCase):
    def fixture(self,root,arm='joint',bad_primary=False):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_common_seeded_risk_score')
        _,oe_root,tp=old_fixtures.OERiskScoreTests().fixture(root,arm=arm)
        for path in (oe_root/'complete.json',oe_root/'calibration/complete.json'):
            d=json.loads(path.read_text());d['code_commit']=m.OE_PARENT_CODE;path.write_text(json.dumps(d))
        om=json.loads((oe_root/'complete.json').read_text());parent=Path(om['parent_prediction_root'])
        with np.load(oe_root/'thresholds.npz') as z:old_tau=z['conditional'].copy()
        tau=np.vstack((old_tau,np.full(6,2. if bad_primary else 0.,np.float64)))
        folder=root/'seeded';folder.mkdir();(folder/'calibration').mkdir()
        np.savez(folder/'thresholds.npz',conditional=tau)
        for role in ('known','novel'):
            with np.load(parent/(role+'_predictions.npz')) as z:d={key:z[key].copy() for key in z.files}
            d['scores']=d['aux_logits'].astype(np.float64).max(1)[:,None]-tau[:,d['closed']].T
            np.savez(folder/(role+'_predictions.npz'),**d)
        with np.load(oe_root/'OE_scores.npz') as z:oe={key:z[key].copy() for key in z.files}
        np.savez(folder/'OE_scores.npz',**oe)
        marker=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,run_id=m.RUN_ID,code_commit='c'*40,arm=arm,
            method_names=list(m.METHODS),primary_arm='joint',primary_method=m.PRIMARY,classification_path='P1',
            parent_run_id=m.PARENT_RUN,parent_code_commit=m.PARENT_CODE,parent_model_identity=om['parent_model_identity'],
            parent_prediction_root=str(parent),OE_parent_run_id=m.OE_PARENT_RUN,OE_parent_code_commit=m.OE_PARENT_CODE,
            OE_prediction_root=str(oe_root),known_count=om['known_count'],novel_count=om['novel_count'],OE_count=om['OE_count'],
            source_only=True,target_reads=0,source_L_reads=0,source_U_reads=0,source_novel_truth_reads=0,neural_updates=0,G_forward_rows=0,F_forward_rows=2,
            real_P1_forward_rows=0,synthetic_P0_forward_rows=2,monotone_thresholds_from_common=True,
            model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True)
        (folder/'complete.json').write_text(json.dumps(marker))
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='c'*40,arm=arm,
            method_names=list(m.METHODS),conditional_thresholds=tau.tolist(),group_count=42,source_only=True,target_reads=0,
            source_V_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,novel_inputs_used=False,
            methods={name:dict(constraints_passed=True,conditional_thresholds=tau[j].tolist(),groups={g:{} for g in m.GROUP_NAMES})
                for j,name in enumerate(m.METHODS)})
        cal['methods'][m.PRIMARY]['initial_thresholds']=tau[0].tolist()
        (folder/'calibration/complete.json').write_text(json.dumps(cal));return m,folder,tp

    def save_thresholds(self,m,folder,tau):
        np.savez(folder/'thresholds.npz',conditional=tau)
        p=folder/'calibration/complete.json';cal=json.loads(p.read_text());cal['conditional_thresholds']=tau.tolist()
        for j,name in enumerate(m.METHODS):cal['methods'][name]['conditional_thresholds']=tau[j].tolist()
        p.write_text(json.dumps(cal))
        for role in ('known','novel'):
            p=folder/(role+'_predictions.npz')
            with np.load(p) as z:d={key:z[key].copy() for key in z.files}
            d['scores']=d['aux_logits'].astype(np.float64).max(1)[:,None]-tau[:,d['closed']].T;np.savez(p,**d)

    def test_fixed_primary_pass_or_fail_not_replaced_by_better_zero_policy(self):
        for failed in (False,True):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as td:
                m,p,t=self.fixture(Path(td),bad_primary=failed);result=m.score(p,t,Path(td)/'score')
                self.assertEqual(result['eligible_primary'],not failed)
                self.assertTrue(result['methods']['zero_initialized_budget']['goals']['passed'])
                self.assertEqual(result['primary_summary']['unknown_macro_rejection'],1.)
                self.assertEqual(result['primary_summary']['known_macro_open_accuracy'],float(not failed))
                self.assertTrue(result['monotone_policy_verified']);self.assertIsNone(result['A_N']);self.assertIsNone(result['H_OW'])
                self.assertEqual(result['primary_summary']['original_F0_macro_closed_accuracy'],1.)
                with self.assertRaises(FileExistsError):m.score(p,t,Path(td)/'score')

    def test_control_cannot_promote_and_complete_groups_and_pairing_are_reported(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t=self.fixture(Path(td),arm='detach_control');r=m.score(p,t,Path(td)/'score')
            self.assertTrue(r['arm_source_gate_passed']);self.assertFalse(r['eligible_primary'])
            self.assertEqual(len(r['methods'][m.PRIMARY]['groups']),42)
            self.assertEqual(r['paired_primary_minus']['common_tail']['unknown_rejection_lost'],0)

    def test_corrupt_controls_or_nonmonotone_policy_fail_before_truth(self):
        for defect in ('freeze','parent_code','OE_parent_code','novel_truth_read','margin','zero_tau','lower_tau','private_novel','identity','OE_score'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                m,p,t=self.fixture(Path(td))
                if defect in ('freeze','parent_code','OE_parent_code','novel_truth_read'):
                    path=p/'complete.json';d=json.loads(path.read_text())
                    key={'freeze':'all_thresholds_frozen_before_novel','parent_code':'parent_code_commit','OE_parent_code':'OE_parent_code_commit','novel_truth_read':'source_novel_truth_reads'}[defect]
                    d[key]=False if defect=='freeze' else 1 if defect=='novel_truth_read' else 'd'*40;path.write_text(json.dumps(d))
                elif defect in ('zero_tau','lower_tau'):
                    with np.load(p/'thresholds.npz') as z:tau=z['conditional'].copy()
                    tau[1 if defect=='zero_tau' else 2]=1. if defect=='zero_tau' else -3.
                    self.save_thresholds(m,p,tau)
                elif defect=='OE_score':
                    with np.load(p/'OE_scores.npz') as z:d={key:z[key].copy() for key in z.files}
                    d['score'][0]+=.5;np.savez(p/'OE_scores.npz',**d)
                else:
                    with np.load(p/'novel_predictions.npz') as z:d={key:z[key].copy() for key in z.files}
                    if defect=='margin':d['scores'][0,2]+=1.
                    elif defect=='identity':d['closed'][0]=1
                    else:d['tx']=np.asarray(['19-15']*len(d['ids']))
                    np.savez(p/'novel_predictions.npz',**d)
                with patch.object(m,'_truth',side_effect=AssertionError('truth before complete integrity checks')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(p,t,Path(td)/'score')


if __name__=='__main__':unittest.main()
