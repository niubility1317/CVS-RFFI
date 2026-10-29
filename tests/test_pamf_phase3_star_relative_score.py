from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')


class StarRelativeScoreTests(unittest.TestCase):
    def module(self):
        name='experiments.pamf_phase3_open_world.star_relative_score'
        self.assertTrue((Path(__file__).resolve().parents[1]/(name.replace('.','/')+'.py')).is_file())
        return importlib.import_module(name)

    def fixture(self,root,condition='P0'):
        uy=np.tile(np.repeat(np.arange(8,dtype=np.int64),8),6)
        qy=np.tile(np.repeat(np.arange(9,dtype=np.int64),16),6)
        ui=np.asarray(['U-star-'+str(i) for i in range(384)]);qi=np.asarray(['Q-star-'+str(i) for i in range(864)])
        us=np.repeat(SCENES,64);qs=np.repeat(SCENES,144)
        closed=np.where(qy<6,qy,0).astype(np.int64)
        score=np.broadcast_to(np.where(qy<6,1.,-1.)[None,:,None],(2,864,7)).copy()
        score[0,np.arange(0,864,144),:]=-1.
        diagnostic=np.where(score>=0.,closed[None,:,None],-1).astype(np.int64)
        raw=np.stack((diagnostic[0,:,6],diagnostic[1,:,6],diagnostic[0,:,6],diagnostic[1,:,6]))
        raw[2:,qy==6]=6;raw[2:,qy==7]=7
        assignment=np.where(uy>=6,uy-6,-1).astype(np.int64)
        p=dict(ids=qi,U_ids=ui,state_names=np.asarray(STATES),prediction=raw,closed_prediction=closed,
            baseline_closed_prediction=closed.copy(),knownness=score,diagnostic_reg0_prediction=diagnostic,
            thresholds=np.zeros(7),joint_margins_DA0=np.zeros((864,8)),joint_margins_DA1=np.zeros((864,8)))
        diag=dict(ids=ui,assignments_DA0=assignment.copy(),assignments_DA1=assignment.copy(),candidate_DA0=uy>=6,candidate_DA1=uy>=6)
        done=dict(status='STAR_RELATIVE_PREDICTIONS_COMPLETE',schema='pamf_phase3_star_relative_target_v1',
            run_id='20261029-phase3-star-relative-s2026092701-r01',condition=condition,view_family='practical_cross_receiver',
            count=864,U_count=384,bias_applied=True,states=list(STATES),prototype_counts={'DA0':2,'DA1':2},
            all_new_IQ_satellite=True,encoder_models=1,auxiliary_E51_read=False)
        ut,qt=root/'U.npz',root/'Q.npz'
        np.savez(ut,ids=ui[::-1],y=uy[::-1],receiver=np.full(384,26,np.int64),scenario=us[::-1])
        np.savez(qt,ids=qi[::-1],y=qy[::-1],receiver=np.full(864,26,np.int64),scenario=qs[::-1])
        return done,p,diag,ut,qt,uy,qy,qs

    def targets(self):
        return Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'

    def test_two_star_paths_four_states_six_scene_success_and_counter_freeze(self):
        module=self.module()
        for condition in ('P0','P1'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,qs=self.fixture(root,condition);before=deepcopy((p,diag))
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})), \
                     patch.object(module.transfer,'_goals',side_effect=AssertionError('Three-scene goal is forbidden')):
                    result=module.score(root,ut,qt,self.targets(),root/'score')
                self.assertTrue(result['primary_pass']);self.assertEqual(result['condition_unknown_rejection_min'],.90)
                self.assertEqual(result['primary_method'],'relative_mahalanobis');self.assertEqual(result['future_unknown_labels'],[6,7,8])
                self.assertEqual(result['U_true_arrival_labels'],[6,7]);self.assertEqual(result['planned_later_labels'],[8])
                self.assertEqual(result['cluster_diagnostics']['DA1']['mapping'],{'0':6,'1':7})
                for state,row in result['states'].items():
                    self.assertEqual(set(row['by_scenario']),set(SCENES));self.assertEqual(len(row['by_transmitter']),9)
                    self.assertTrue(row['overall']['coverage']['complete'])
                    self.assertFalse(row['by_transmitter']['0']['coverage']['complete'])
                    if state.endswith('REG0'):
                        self.assertIsNone(row['overall']['new_macro_open_accuracy']);self.assertIsNone(row['overall']['H'])
                self.assertEqual(result['states']['DA1_REG1']['overall']['old_correct_accepted'],576)
                self.assertEqual(result['states']['DA1_REG1']['overall']['new_macro_open_accuracy'],1.)
                self.assertIsNone(result['deltas']['DA1_REG1_minus_DA1_REG0']['new_macro_open_accuracy'])
                self.assertEqual(set(result['diagnostic_methods_reg0']['DA1']),set(module.METHODS))
                self.assertEqual(result['encoder_models'],1);self.assertFalse(result['auxiliary_E51_read'])
                for current,original in zip((p,diag),before):
                    for key in current:self.assertTrue(np.array_equal(current[key],original[key]))

    def test_missing_scene_or_class_is_NA_coverage_false_without_zero_imputation(self):
        module=self.module()
        for mode in ('empty_scene','missing_unknown','missing_old'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,qs=self.fixture(root)
                mask=qs==SCENES[0]
                if mode=='missing_unknown':mask&=qy==7
                elif mode=='missing_old':mask&=qy==5
                qs=qs.copy();qs[mask]=SCENES[1]
                np.savez(qt,ids=p['ids'],y=qy,receiver=np.full(864,26,np.int64),scenario=qs)
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                    result=module.score(root,ut,qt,self.targets(),root/'score')
                self.assertFalse(result['primary_pass'])
                reg0=result['states']['DA1_REG0']['by_scenario'][SCENES[0]]
                reg1=result['states']['DA1_REG1']['by_scenario'][SCENES[0]]
                self.assertFalse(reg0['coverage']['complete']);self.assertFalse(reg1['coverage']['complete'])
                if mode in ('empty_scene','missing_unknown'):self.assertIsNone(reg0['unknown_macro_rejection'])
                if mode in ('empty_scene','missing_old'):self.assertIsNone(reg0['old_macro_acceptance'])
                self.assertIsNone(reg1['intended_registered_macro_frr'])
                self.assertTrue(result['states']['DA1_REG1']['overall']['coverage']['complete'])
                self.assertEqual(set(result['states']['DA1_REG1']['by_scenario']),set(SCENES))

    def test_U_only_mapping_and_missing_future_arrival_observations_are_not_certified(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);done,p,diag,ut,qt,uy,qy,qs=self.fixture(root)
            with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                first=module.score(root,ut,qt,self.targets(),root/'first')
                changed=qy.copy();changed[qy==6]=7;changed[qy==7]=6
                np.savez(qt,ids=p['ids'],y=changed,receiver=np.full(864,26,np.int64),scenario=qs)
                swapped=module.score(root,ut,qt,self.targets(),root/'swapped')
                missing=qy.copy();missing[qy==7]=8
                np.savez(qt,ids=p['ids'],y=missing,receiver=np.full(864,26,np.int64),scenario=qs)
                absent=module.score(root,ut,qt,self.targets(),root/'absent')
            self.assertEqual(first['cluster_diagnostics']['DA1']['mapping'],swapped['cluster_diagnostics']['DA1']['mapping'])
            self.assertEqual(swapped['states']['DA1_REG1']['overall']['new_macro_open_accuracy'],0.)
            row=absent['states']['DA1_REG1']['overall']
            self.assertEqual(row['new_macro_denominator'],2);self.assertIsNone(row['new_macro_open_accuracy'])
            self.assertFalse(row['coverage']['new_arrivals']);self.assertFalse(absent['primary_pass'])
            self.assertEqual(absent['future_unknown_labels'],[6,8])

    def test_zero_cluster_pure_old_unmatched_and_undiscovered_arrival_denominator(self):
        module=self.module()
        for mode in ('zero','pure_old','missed_arrival'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,qs=self.fixture(root)
                for da in ('DA0','DA1'):
                    done['prototype_counts'][da]=0 if mode=='zero' else 1
                    if mode=='zero':a=np.full(384,-1,np.int64)
                    elif mode=='pure_old':a=np.where(uy==0,0,-1).astype(np.int64)
                    else:a=np.where(np.isin(uy,[0,6]),0,-1).astype(np.int64)
                    diag['assignments_'+da]=a
                p['prediction'][2:,qy>=6]=-1
                if mode=='pure_old':p['prediction'][2:,qy>=6]=6
                elif mode=='missed_arrival':p['prediction'][2:,qy==6]=6
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                    result=module.score(root,ut,qt,self.targets(),root/'score')
                row=result['states']['DA1_REG1']['overall'];cl=result['cluster_diagnostics']['DA1']
                self.assertEqual(row['new_macro_denominator'],2);self.assertTrue(row['coverage']['new_arrivals'])
                self.assertEqual(row['new_macro_open_accuracy'],.5 if mode=='missed_arrival' else 0.)
                if mode=='pure_old':
                    self.assertEqual(cl['mapping'],{});self.assertEqual(row['unmatched_anonymous_accepted'],288)
                    self.assertEqual(row['planned_later_macro_rejection'],0.)
                if mode=='missed_arrival':self.assertAlmostEqual(row['intended_registered_macro_frr'],1/8)
                self.assertFalse(result['primary_pass'])

    def test_high_scene_later_protection_is_required_beyond_overall90(self):
        module=self.module()
        for concentrated in (False,True):
            with self.subTest(concentrated=concentrated),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);done,p,diag,ut,qt,uy,qy,qs=self.fixture(root,'P1')
                bad=np.flatnonzero((qy==8)&(qs==SCENES[0]))[:6] if concentrated else np.asarray([np.flatnonzero((qy==8)&(qs==s))[0] for s in SCENES])
                p['prediction'][3,bad]=6
                with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                    result=module.score(root,ut,qt,self.targets(),root/'score')
                self.assertAlmostEqual(result['states']['DA1_REG1']['overall']['planned_later_macro_rejection'],.9375)
                self.assertEqual(result['primary_pass'],not concentrated)
                self.assertEqual(result['registration_goals']['DA1']['by_scenario'][SCENES[0]]['planned_later_unknown_protection'],not concentrated)

    def test_incomplete_or_wrong_condition_never_opens_truth_and_output_is_immutable(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(module,'_prediction_gate',side_effect=ValueError('incomplete')),patch.object(module,'_truth',side_effect=AssertionError('No truth')) as truth:
                with self.assertRaisesRegex(ValueError,'incomplete'):module.score(root,root/'U',root/'Q',self.targets(),root/'out')
                self.assertEqual(truth.call_count,0)
            done,p,diag,ut,qt,*_=self.fixture(root);done['condition']='C0'
            with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})),patch.object(module,'_truth',side_effect=AssertionError('No clean new Phase3')):
                with self.assertRaises(ValueError):module.score(root,ut,qt,self.targets(),root/'clean')
            (root/'out').mkdir()
            with patch.object(module,'_prediction_gate',side_effect=AssertionError('No work')):
                with self.assertRaises(FileExistsError):module.score(root,ut,qt,self.targets(),root/'out')

    def test_receiver_disjointness_and_nonidentity_adaptation_claim(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);done,p,diag,ut,qt,uy,qy,qs=self.fixture(root)
            with patch.object(module,'_prediction_gate',return_value=(done,p,diag,{})):
                done['bias_applied']=False
                result=module.score(root,ut,qt,self.targets(),root/'identity')
                self.assertFalse(result['adaptation_goals']['passed']);self.assertFalse(result['primary_pass'])
                np.savez(qt,ids=p['ids'],y=qy,receiver=np.full(864,30,np.int64),scenario=qs)
                with self.assertRaises(ValueError):module.score(root,ut,qt,self.targets(),root/'formal')
                p['U_ids'][0]=p['ids'][0]
                with patch.object(module,'_truth',side_effect=AssertionError('No overlap truth')):
                    with self.assertRaises(ValueError):module.score(root,ut,qt,self.targets(),root/'overlap')


if __name__=='__main__':unittest.main()
