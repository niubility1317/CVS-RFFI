from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')


class RegistrationScoreTests(unittest.TestCase):
    def module(self):
        name='experiments.pamf_phase3_open_world.unlabeled_registration_score'
        self.assertTrue((Path(__file__).resolve().parents[1]/(name.replace('.','/')+'.py')).is_file())
        return importlib.import_module(name)

    def fixture(self,root,condition='C0'):
        scenes=('clean',) if condition=='C0' else ('practical_low_suburban','practical_mid_urban','practical_low_urban')
        u_y=np.tile(np.repeat(np.arange(8,dtype=np.int64),16),3)
        q_y=np.tile(np.repeat(np.arange(9,dtype=np.int64),32),3)
        u_ids=np.asarray(['U-'+str(i).zfill(4) for i in range(384)])
        q_ids=np.asarray(['Q-'+str(i).zfill(4) for i in range(864)])
        u_scene=np.full(384,'clean') if condition=='C0' else np.repeat(scenes,128)
        q_scene=np.full(864,'clean') if condition=='C0' else np.repeat(scenes,288)
        closed=np.where(q_y<6,q_y,0).astype(np.int64)
        values=np.broadcast_to(np.where(q_y<6,1.,-1.)[None,:,None],(2,864,7)).copy()
        for i in (0,288,576):values[0,i,:]=-1.
        diagnostic=np.where(values>=0.,closed[None,:,None],-1).astype(np.int64)
        raw=np.stack((diagnostic[0,:,6],diagnostic[1,:,6],diagnostic[0,:,6],diagnostic[1,:,6]))
        raw[2:,q_y==6]=6;raw[2:,q_y==7]=7
        assignments=np.where(u_y>=6,u_y-6,-1).astype(np.int64)
        p=dict(ids=q_ids,U_ids=u_ids,state_names=np.asarray(STATES),prediction=raw,
            closed_prediction=closed,baseline_closed_prediction=closed.copy(),knownness=values,
            diagnostic_reg0_prediction=diagnostic,thresholds=np.zeros(7,np.float64),
            joint_margins_DA0=np.zeros((864,8)),joint_margins_DA1=np.zeros((864,8)))
        diag=dict(ids=u_ids,assignments_DA0=assignments.copy(),assignments_DA1=assignments.copy(),
            candidate_DA0=u_y>=6,candidate_DA1=u_y>=6)
        done=dict(run_id='20261029-phase3-unlabeled-registration-s2026092701-r01',condition=condition,
            view_family='clean_cross_receiver' if condition=='C0' else 'practical_cross_receiver',count=864,U_count=384,
            bias_applied=True,states=list(STATES),prototype_counts={'DA0':2,'DA1':2})
        ut=root/'U_truth.npz';qt=root/'Q_truth.npz'
        np.savez(ut,ids=u_ids[::-1],y=u_y[::-1],receiver=np.full(384,26,np.int64),scenario=u_scene[::-1])
        np.savez(qt,ids=q_ids[::-1],y=q_y[::-1],receiver=np.full(864,26,np.int64),scenario=q_scene[::-1])
        return done,p,diag,ut,qt,u_y,q_y,q_scene

    def targets(self):
        return Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'

    def test_real_truth_dictionary_four_states_all_scenes_and_full_pipeline(self):
        module=self.module()
        for condition in ('C0','P1'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,scenes=self.fixture(root,condition)
                before=deepcopy((p,diag))
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                    result=module.score(root/'predict',ut,qt,self.targets(),root/'score')
                self.assertTrue(result['truth_last']);self.assertTrue(result['primary_pass'])
                self.assertEqual(set(result['states']),set(STATES));self.assertEqual(result['U_true_arrival_labels'],[6,7])
                self.assertEqual(result['planned_later_labels'],[8])
                self.assertEqual(result['cluster_diagnostics']['DA1']['mapping'],{'0':6,'1':7})
                for state in STATES:
                    row=result['states'][state]
                    self.assertEqual(set(row['by_scenario']),set(scenes.tolist()))
                    self.assertEqual(len(row['by_transmitter']),9)
                    if state.endswith('REG0'):
                        self.assertIsNone(row['overall']['new_macro_open_accuracy']);self.assertIsNone(row['overall']['H'])
                    else:
                        self.assertEqual(row['overall']['new_macro_open_accuracy'],1.)
                self.assertTrue(result['adaptation_goals']['passed'])
                self.assertTrue(result['registration_goals']['DA1']['passed'])
                self.assertEqual(result['states']['DA1_REG1']['overall']['old_macro_open_accuracy'],1.)
                self.assertEqual(len(result['diagnostic_methods_reg0']['DA0']),7)
                self.assertIsNone(result['deltas']['DA1_REG1_minus_DA1_REG0']['new_macro_open_accuracy'])
                self.assertFalse(result['business_registration_authorized']);self.assertFalse(result['final_phase3_complete'])
                for a,b in zip((p,diag),before):
                    for k in a:self.assertTrue(np.array_equal(a[k],b[k]))

    def test_Hungarian_mapping_is_U_only_when_Q_new_labels_are_swapped(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);done,p,diag,ut,qt,uy,qy,scenes=self.fixture(root)
            with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                first=module.score(root/'predict',ut,qt,self.targets(),root/'first')
                swapped=qy.copy();swapped[qy==6]=7;swapped[qy==7]=6
                np.savez(qt,ids=p['ids'],y=swapped,receiver=np.full(864,26,np.int64),scenario=scenes)
                second=module.score(root/'predict',ut,qt,self.targets(),root/'second')
            self.assertEqual(first['cluster_diagnostics']['DA1']['mapping'],second['cluster_diagnostics']['DA1']['mapping'])
            self.assertEqual(first['states']['DA1_REG1']['overall']['new_macro_open_accuracy'],1.)
            self.assertEqual(second['states']['DA1_REG1']['overall']['new_macro_open_accuracy'],0.)
            self.assertFalse(second['primary_pass']);self.assertFalse(second['Q_truth_used_for_cluster_mapping'])

    def test_zero_clusters_and_pure_old_clusters_remain_unmatched_accepted_errors(self):
        module=self.module()
        for pure_old in (False,True):
            with self.subTest(pure_old=pure_old),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,scenes=self.fixture(root)
                for da in ('DA0','DA1'):
                    done['prototype_counts'][da]=int(pure_old)
                    diag['assignments_'+da]=np.where(uy==0,0,-1).astype(np.int64) if pure_old else np.full(384,-1,np.int64)
                p['prediction'][2:,qy>=6]=6 if pure_old else -1
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                    result=module.score(root/'predict',ut,qt,self.targets(),root/'score')
                row=result['states']['DA1_REG1']['overall']
                self.assertEqual(result['cluster_diagnostics']['DA1']['mapping'],{})
                self.assertEqual(row['new_macro_open_accuracy'],0.)
                self.assertEqual(row['planned_later_macro_rejection'],0. if pure_old else 1.)
                self.assertEqual(row['actual_unregistered_macro_rejection'],0. if pure_old else 1.)
                self.assertEqual(row['unmatched_anonymous_accepted'],288 if pure_old else 0)
                self.assertFalse(result['primary_pass'])
                if pure_old:self.assertEqual(result['cluster_diagnostics']['DA1']['pure_old_cluster_count'],1)

    def test_old_contamination_does_not_remove_missed_true_arrival_denominator(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);done,p,diag,ut,qt,uy,qy,scenes=self.fixture(root)
            for da in ('DA0','DA1'):
                done['prototype_counts'][da]=1
                diag['assignments_'+da]=np.where(np.isin(uy,[0,6]),0,-1).astype(np.int64)
            p['prediction'][2:,qy==6]=6;p['prediction'][2:,qy==7]=-1
            with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                result=module.score(root/'predict',ut,qt,self.targets(),root/'score')
            row=result['states']['DA1_REG1']['overall'];cluster=result['cluster_diagnostics']['DA1']
            self.assertEqual(cluster['mapping'],{'0':6});self.assertEqual(cluster['old_contaminated_cluster_count'],1)
            self.assertEqual(row['new_macro_denominator'],2);self.assertEqual(row['new_macro_open_accuracy'],.5)
            self.assertAlmostEqual(row['intended_registered_macro_frr'],1/8)
            self.assertIn(7,result['states']['DA1_REG1']['actual_unregistered_labels'])
            self.assertFalse(result['registration_goals']['DA1']['passed'])

    def test_later_false_anonymous_accept_clean95_star90_main_scene_rule(self):
        module=self.module()
        for condition in ('C0','P1'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,scenes=self.fixture(root,condition)
                if condition=='C0':bad=np.flatnonzero(qy==8)[-6:]
                else:bad=np.concatenate([np.flatnonzero((qy==8)&(scenes==s))[-2:] for s in np.unique(scenes)])
                p['prediction'][3,bad]=6
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                    result=module.score(root/'predict',ut,qt,self.targets(),root/'score')
                self.assertAlmostEqual(result['states']['DA1_REG1']['overall']['planned_later_macro_rejection'],.9375)
                self.assertEqual(result['primary_pass'],condition=='P1')
                self.assertEqual(result['condition_unknown_rejection_min'],.95 if condition=='C0' else .90)

    def test_incomplete_prediction_and_overwrite_never_connect_truth(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(module,'_prediction_gate',side_effect=ValueError('incomplete')), \
                 patch.object(module,'_truth',side_effect=AssertionError('Truth remains closed')) as truth:
                with self.assertRaisesRegex(ValueError,'incomplete'):module.score(root,root/'U',root/'Q',self.targets(),root/'out')
                self.assertEqual(truth.call_count,0)
            (root/'out').mkdir()
            with patch.object(module,'_prediction_gate',side_effect=AssertionError('No work after output collision')):
                with self.assertRaises(FileExistsError):module.score(root,root/'U',root/'Q',self.targets(),root/'out')

    def test_truth_RX_and_U_Q_physical_disjointness_and_bias_claim(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);done,p,diag,ut,qt,uy,qy,scenes=self.fixture(root)
            with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                done['bias_applied']=False
                result=module.score(root,ut,qt,self.targets(),root/'no_adaptation')
                self.assertFalse(result['adaptation_goals']['passed']);self.assertFalse(result['primary_pass'])
                np.savez(ut,ids=p['U_ids'],y=uy,receiver=np.full(384,27,np.int64),scenario=np.full(384,'clean'))
                with self.assertRaises(ValueError):module.score(root,ut,qt,self.targets(),root/'bad_rx')
                p['U_ids'][0]=p['ids'][0]
                with patch.object(module,'_truth',side_effect=AssertionError('Overlap rejected before truth')):
                    with self.assertRaises(ValueError):module.score(root,ut,qt,self.targets(),root/'overlap')


if __name__=='__main__':unittest.main()
