"""Source-only synthetic calibration and unchanged consensus primitives."""
import copy
import importlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


class StarConsensusReadoutTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_consensus_readout.py').is_file(),
                        'Star consensus readout is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_consensus_readout')

    def inputs(self,m,count=10,counts=None):
        rows=[(y,rx,scene,k) for y in range(6) for rx in m.RECEIVERS for scene in m.SCENES
              for k in range(count if counts is None else counts[y])]
        n=len(rows);y=np.asarray([r[0] for r in rows],np.int64)
        return rows,np.asarray(['source-V-'+str(i) for i in range(n)]),y,np.asarray([r[1] for r in rows],np.int64),np.asarray([r[2] for r in rows])

    def test_90_acceptance_rejects_ten_percent_errors_and_keeps_complete_ties(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m)
        closed=y.copy();low=np.asarray([r[3]==0 for r in rows]);closed[low]=(closed[low]+1)%6
        scores=np.tile(np.where(low,0.,1.)[:,None],(1,7));before=scores.copy()
        thresholds,report=m.calibrate_known(ids,y,closed,closed,scores,rx,scene)
        np.testing.assert_array_equal(thresholds,np.ones(7));self.assertEqual(thresholds.dtype,np.float64)
        self.assertTrue(report['constraints_passed']);self.assertEqual(report['group_count'],42)
        self.assertEqual(report['status'],'STAR_CONSENSUS_THRESHOLDS_FROZEN')
        self.assertEqual(report['schema'],'pamf_phase3_star_consensus_prediction_v1')
        for row in report['methods'].values():
            self.assertEqual(len(row['groups']),42);self.assertFalse(row['next_candidate_passed'])
            g=row['groups']['overall'];self.assertEqual(g['macro_frr'],.1);self.assertEqual(g['known_macro_acceptance'],.9)
            self.assertEqual(g['original_closed_minus_new_open'],0.)
        self.assertEqual(report['limits']['known_macro_frr'],.1)
        self.assertEqual(report['limits']['known_macro_acceptance_min'],.9)
        np.testing.assert_array_equal(scores,before);json.dumps(report,allow_nan=False)

    def test_exact_two_percent_original_closed_loss_controls_more_than_acceptance(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m,100)
        values=np.asarray([0. if r[3]<2 else 1. if r[3]==2 else 2. for r in rows])
        thresholds,report=m.calibrate_known(ids,y,y,y,np.tile(values[:,None],(1,7)),rx,scene)
        np.testing.assert_array_equal(thresholds,np.ones(7));self.assertTrue(report['constraints_passed'])
        for row in report['methods'].values():
            self.assertEqual(row['groups']['overall']['original_closed_minus_new_open'],.02)
            self.assertEqual(row['next_candidate_threshold'],2.);self.assertFalse(row['next_candidate_passed'])

    def test_class_macro_not_micro_and_one_RX_scene_can_limit_global_threshold(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m,counts=[1,100,100,100,100,100])
        closed=y.copy();closed[y==0]=1
        thresholds,report=m.calibrate_known(ids,y,closed,closed,np.tile((y!=0).astype(float)[:,None],(1,7)),rx,scene)
        np.testing.assert_array_equal(thresholds,np.zeros(7));self.assertTrue(report['constraints_passed'])
        self.assertTrue(report['methods']['msp']['no_rejection_feasible'])
        rows,ids,y,rx,scene=self.inputs(m)
        low=np.asarray([r[3] < (2 if r[1]==1 and r[2]==m.SCENES[0] else 1) for r in rows])
        closed=y.copy();closed[low]=(closed[low]+1)%6
        thresholds,report=m.calibrate_known(ids,y,closed,closed,np.tile(np.where(low,0.,1.)[:,None],(1,7)),rx,scene)
        np.testing.assert_array_equal(thresholds,np.zeros(7))
        self.assertEqual(report['methods']['msp']['local_maximum_thresholds']['rxscene/1/'+m.SCENES[0]],0.)

    def test_large_tie_cannot_be_split_and_row_or_class_permutation_preserves_result(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m)
        low=np.asarray([r[3]<3 for r in rows]);closed=y.copy();closed[low]=(closed[low]+1)%6
        scores=np.tile(np.where(low,-2.,1.)[:,None],(1,7))
        expected,report=m.calibrate_known(ids,y,closed,closed,scores,rx,scene)
        np.testing.assert_array_equal(expected,np.full(7,-2.))
        permutation=np.array([3,5,0,4,1,2]);order=np.random.default_rng(8).permutation(len(ids))
        got,again=m.calibrate_known(ids[order],permutation[y[order]],permutation[closed[order]],
            permutation[closed[order]],scores[order],rx[order],scene[order])
        np.testing.assert_array_equal(got,expected);self.assertEqual(again['constraints_passed'],report['constraints_passed'])

    def test_missing_coverage_or_preexisting_closed_damage_accepts_all_and_stays_false(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m)
        scores=np.tile(np.asarray([r[3] for r in rows],float)[:,None],(1,7))
        keep=~((y==0)&(rx==1)&(scene==m.SCENES[0]))
        for label in ('coverage','damage'):
            with self.subTest(label=label):
                ix=keep if label=='coverage' else np.ones(len(y),bool)
                closed=y if label=='coverage' else (y+1)%6
                thresholds,report=m.calibrate_known(ids[ix],y[ix],closed[ix],y[ix],scores[ix],rx[ix],scene[ix])
                self.assertFalse(report['constraints_passed']);np.testing.assert_array_equal(thresholds,np.zeros(7))
                self.assertTrue(np.all(scores[ix]>=thresholds))
                self.assertTrue(all(row['zero_rejection_fallback'] for row in report['methods'].values()))
                if label=='coverage':
                    group=report['methods']['msp']['groups']['rxscene/1/'+m.SCENES[0]]
                    self.assertFalse(group['certified']);self.assertFalse(group['constraints_passed'])
                else:self.assertEqual(report['methods']['msp']['groups']['overall']['original_closed_minus_new_open'],1.)

    def test_invalid_unknown_labels_nonfinite_shapes_and_duplicate_ids_rejected(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m)
        good=[ids,y,y.copy(),y.copy(),np.ones((len(y),7)),rx,scene]
        for defect in ('unknown','head','baseline','nan','columns','duplicate','RX','scene','integer_scores'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                args=[a.copy() for a in good]
                if defect=='unknown':args[1][0]=6
                elif defect=='head':args[2][0]=-1
                elif defect=='baseline':args[3][0]=6
                elif defect=='nan':args[4][0,0]=np.nan
                elif defect=='columns':args[4]=args[4][:,:6]
                elif defect=='duplicate':args[0][0]=args[0][1]
                elif defect=='RX':args[5][0]=26
                elif defect=='scene':args[6][0]='invalid'
                else:args[4]=args[4].astype(np.int64)
                m.calibrate_known(*args)
        with self.assertRaises(TypeError):m.calibrate_known(*good,unknown_scores=np.ones((10,7)))

    def test_reused_algorithms_and_actual_8branch_descriptor_seven_scores_without_mutation(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import source_consensus as consensus, source_open_readout as core, source_domain_oe_readout as domain, scores
        from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
        self.assertIs(m.fit_readouts,domain.fit_readouts);self.assertIs(m.fit_geometry,core.fit_geometry)
        self.assertIs(m.descriptor,consensus.descriptor);self.assertIs(m.score_readouts,core.score_readouts)
        self.assertEqual(inspect.signature(m.fit_readouts),inspect.signature(domain.fit_readouts))
        rng=np.random.default_rng(7);y=np.repeat(np.arange(6),3)
        branches={name:rng.normal(size=(18,d)).astype(np.float32) for name,d in DIMENSIONS.items()}
        branches['reference29'][:,27]=0.;before=copy.deepcopy(branches)
        geometry=m.fit_geometry(branches,y);logits=rng.normal(size=(18,6)).astype(np.float32)
        state=dict(schema=core.READOUT_SCHEMA,descriptor_dim=100,method_names=list(core.METHODS),training={},
            mean=np.zeros(100),scale=np.ones(100),models=dict(
                linear_oe=dict(weight=np.ones((1,100),np.float32)*.01,bias=np.zeros(1,np.float32)),
                mlp16_oe=dict(weight1=np.zeros((16,100),np.float32),bias1=np.zeros(16,np.float32),
                    weight2=np.zeros((1,16),np.float32),bias2=np.ones(1,np.float32))))
        value=m.all_scores(branches,logits,geometry,state)
        self.assertEqual((value.shape,value.dtype),((18,7),np.dtype('float64')))
        static=scores.knownness_scores(branches['fused160'],logits,geometry['branches']['fused160']['prototypes'])
        np.testing.assert_array_equal(value,m.score_compact(m.descriptor(branches,logits,geometry),static,state))
        for name in branches:np.testing.assert_array_equal(branches[name],before[name])
        self.assertEqual(m.METHODS,('msp','maxlogit','margin','energy','cosine','linear_oe','mlp16_oe'))

    def test_explicit_nonseven_method_names_and_default_backward_compatibility(self):
        m=self.module();rows,ids,y,rx,scene=self.inputs(m)
        values=np.ones((len(y),7),np.float64)
        old,original=m.calibrate_known(ids,y,y,y,values,rx,scene)
        same,explicit=m.calibrate_known(ids,y,y,y,values,rx,scene,method_names=m.METHODS)
        np.testing.assert_array_equal(same,old);self.assertEqual(explicit,original)
        names=('first','second','third')
        got,report=m.calibrate_known(ids,y,y,y,values[:,:3],rx,scene,method_names=names)
        self.assertEqual(got.shape,(3,));self.assertEqual(report['method_names'],list(names))
        self.assertEqual(list(report['methods']),list(names));self.assertTrue(report['constraints_passed'])
        for bad in ((),('same','same'),('good',''),('good',2),'one'):
            with self.subTest(method_names=bad),self.assertRaises(ValueError):
                m.calibrate_known(ids,y,y,y,values[:,:3],rx,scene,method_names=bad)


if __name__=='__main__':unittest.main()
