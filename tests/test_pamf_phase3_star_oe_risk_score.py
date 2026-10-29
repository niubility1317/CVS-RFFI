"""Source-only score integrity and fixed-primary policy on complete fixtures."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tests import test_pamf_phase3_star_joint_ova_score as parent_fixtures


class OERiskScoreTests(unittest.TestCase):
    def fixture(self,root,failed_primary=False,arm='joint'):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_oe_risk_score')
        parent,tp,k,u,_=parent_fixtures.JointOvaScoreTests().fixture(root,arm)
        code=m.PARENT_CODE
        for path in (parent/'complete.json',parent/'calibration/complete.json'):
            d=json.loads(path.read_text());d['code_commit']=code
            if 'model_identity' in d:d['model_identity']['code_commit']=code
            path.write_text(json.dumps(d))
        with np.load(parent/'thresholds.npz') as z:tau={key:z[key].copy() for key in z.files}
        tau['conditional'][2]=-2. if not failed_primary else 0.
        np.savez(parent/'thresholds.npz',**tau)
        cp=parent/'calibration/complete.json';cal=json.loads(cp.read_text());cal['conditional_thresholds']=tau['conditional'].tolist()
        cal['methods']['ova_max']['conditional_thresholds']=tau['conditional'][2].tolist();cp.write_text(json.dumps(cal))
        parent_fixtures.JointOvaScoreTests().save(importlib.import_module('experiments.pamf_phase3_open_world.star_joint_ova_score'),parent,k,u)
        folder=root/'new';folder.mkdir();(folder/'calibration').mkdir()
        new_tau=np.stack([tau['conditional'][2],np.full(6,-2. if failed_primary else 0.)])
        np.savez(folder/'thresholds.npz',conditional=new_tau)
        for role,data in (('known',k),('novel',u)):
            data=dict(data);data['scores']=data['aux_logits'].max(1).astype(np.float64)[:,None]-new_tau[:,data['closed']].T
            np.savez(folder/(role+'_predictions.npz'),**data)
        n=15;z=np.full((n,6),-1.,np.float32)
        np.savez(folder/'OE_scores.npz',ids=np.asarray(['oe'+str(i) for i in range(n)]),
            rx_i=np.repeat(np.asarray(m.RECEIVERS,np.int64),3),day_i=np.tile(np.arange(1,4,dtype=np.int64),5),
            domain=np.arange(n,dtype=np.int64),default_logits=z,aux_logits=z,closed=np.zeros(n,np.int64),score=np.full(n,-1.,np.float64))
        parent_marker=json.loads((parent/'complete.json').read_text())
        marker=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,run_id=m.RUN_ID,code_commit='b'*40,arm=arm,
            parent_run_id=m.PARENT_RUN,parent_code_commit=m.PARENT_CODE,parent_model_identity=parent_marker['model_identity'],
            parent_prediction_root=str(parent),method_names=list(m.METHODS),primary_arm='joint',primary_method='oe_risk_budget',
            classification_path='P1',known_count=len(k['ids']),novel_count=len(u['ids']),OE_count=n,source_only=True,
            target_reads=0,source_L_reads=0,source_U_reads=0,neural_updates=0,G_forward_rows=0,
            all_thresholds_frozen_before_novel=True,model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True)
        (folder/'complete.json').write_text(json.dumps(marker))
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='b'*40,arm=arm,
            method_names=list(m.METHODS),conditional_thresholds=new_tau.tolist(),group_count=42,source_only=True,target_reads=0,
            source_V_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,novel_inputs_used=False,
            methods={name:dict(constraints_passed=True,conditional_thresholds=new_tau[i].tolist(),groups={g:{} for g in m.GROUP_NAMES}) for i,name in enumerate(m.METHODS)})
        (folder/'calibration/complete.json').write_text(json.dumps(cal));return m,folder,tp

    def test_matched_common_control_does_not_replace_failed_budget_primary(self):
        for failed in (False,True):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as td:
                m,p,t=self.fixture(Path(td),failed)
                result=m.score(p,t,Path(td)/'score')
                self.assertEqual(result['eligible_primary'],not failed)
                self.assertEqual(result['primary_summary']['unknown_macro_rejection'],float(not failed))
                self.assertEqual(result['primary_summary']['known_macro_open_accuracy'],1.)
                self.assertIsNone(result['A_N']);self.assertIsNone(result['H_OW'])
                with self.assertRaises(FileExistsError):m.score(p,t,Path(td)/'score')

    def test_control_arm_cannot_promote_and_all_groups_preserve_original_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t=self.fixture(Path(td),arm='detach_control');result=m.score(p,t,Path(td)/'score')
            self.assertTrue(result['arm_source_gate_passed']);self.assertFalse(result['eligible_primary'])
            self.assertEqual(len(result['methods']['oe_risk_budget']['groups']),42)
            self.assertEqual(result['primary_summary']['original_F0_macro_closed_accuracy'],1.)

    def test_broken_predictions_fail_before_truth(self):
        for defect in ('freeze','margin','common_tau','novel_tx','parent_code','default_identity'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                m,p,t=self.fixture(Path(td))
                if defect in ('freeze','parent_code'):
                    marker=json.loads((p/'complete.json').read_text())
                    marker['all_thresholds_frozen_before_novel' if defect=='freeze' else 'parent_code_commit']=False if defect=='freeze' else 'c'*40
                    (p/'complete.json').write_text(json.dumps(marker))
                elif defect=='common_tau':np.savez(p/'thresholds.npz',conditional=np.zeros((2,6),np.float64))
                else:
                    with np.load(p/'novel_predictions.npz') as z:d={key:z[key].copy() for key in z.files}
                    if defect=='margin':d['scores'][0,1]+=1.
                    elif defect=='novel_tx':d['tx']=np.asarray(['19-15']*len(d['ids']))
                    else:d['closed'][0]=1
                    np.savez(p/'novel_predictions.npz',**d)
                with patch.object(m,'_truth',side_effect=AssertionError('private truth read too early')),\
                     self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(p,t,Path(td)/'score')


if __name__=='__main__':unittest.main()
