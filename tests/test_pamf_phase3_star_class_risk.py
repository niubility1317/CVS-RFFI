"""Synthetic known-V calibration; no unknown data, models, IQ or IO."""
from fractions import Fraction
import importlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


class StarClassRiskTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_class_risk.py').is_file(),
                        'Predicted-class risk calibration is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_class_risk')

    def data(self,m,counts=None,wrong=False):
        counts=[50]*6 if counts is None else counts
        rows=[(y,r,s,k) for y in range(6) for r in m.RECEIVERS for s in m.SCENES for k in range(counts[y])]
        y=np.asarray([r[0] for r in rows],np.int64)
        closed=(y+1)%6 if wrong else y.copy()
        score=np.asarray([100*c+(c+1)*row[3] for c,row in zip(closed,rows)],np.float64)
        return dict(ids=np.asarray(['known-V-'+str(i) for i in range(len(y))]),y=y,closed=closed,
            baseline=closed.copy(),score=score,rx=np.asarray([r[1] for r in rows],np.int64),
            scenario=np.asarray([r[2] for r in rows])),rows

    def calibrate(self,m,data):
        return m.calibrate_known(data['ids'],data['y'],data['closed'],data['baseline'],data['score'],data['rx'],data['scenario'])

    def test_class_offset_heteroscedastic_scores_use_q_two_percent_and_42group_protection(self):
        m=self.module();data,_=self.data(m);state,report=self.calibrate(m,data)
        self.assertEqual(report['common_tail_rate'],{'numerator':1,'denominator':50})
        np.testing.assert_array_equal(state['conditional_thresholds'],[100*c+c+1 for c in range(6)])
        self.assertTrue(report['constraints_passed']);self.assertEqual(report['group_count'],42)
        self.assertEqual(len(report['methods']['predicted_class']['groups']),42)
        for group in report['methods']['predicted_class']['groups'].values():
            self.assertEqual(group['known_macro_acceptance'],.98)
            self.assertEqual(group['original_closed_minus_new_open'],.02)
        from experiments.pamf_phase3_open_world import star_consensus_readout as old
        expected,_=old.calibrate_known(data['ids'],data['y'],data['closed'],data['baseline'],data['score'][:,None],
            data['rx'],data['scenario'],method_names=('global_control',))
        self.assertEqual(state['global_threshold'],expected[0])
        margins=m.apply(data['score'],data['closed'],state)
        self.assertEqual((margins.shape,margins.dtype),((len(data['y']),2),np.dtype('float64')))
        np.testing.assert_array_equal(margins[:,1]>=0,data['score']>=state['conditional_thresholds'][data['closed']])
        self.assertFalse(report['source_V_neural_fit']);self.assertFalse(report['source_V_normalizer_fit'])
        self.assertTrue(report['source_V_threshold_fit']);json.dumps(report,allow_nan=False)
        self.assertEqual(set(state),{'schema','global_threshold','conditional_thresholds'})

    def test_fraction_union_floor_budget_is_maximal_for_unequal_predicted_class_sizes(self):
        m=self.module();counts=[50,60,70,80,90,100];data,_=self.data(m,counts)
        state,report=self.calibrate(m,data)
        candidates=sorted({Fraction(j,n) for n in counts for j in range(n+1)})
        def loss(q):return sum((Fraction((q.numerator*n)//q.denominator,n) for n in counts),Fraction())/6
        expected=max(q for q in candidates if loss(q)<=Fraction(1,50))
        q=Fraction(**dict(numerator=report['common_tail_rate']['numerator'],denominator=report['common_tail_rate']['denominator']))
        self.assertEqual(q,expected)
        np.testing.assert_array_equal(state['conditional_thresholds'],
            [100*c+(c+1)*((q.numerator*n)//q.denominator) for c,n in enumerate(counts)])
        following=candidates[candidates.index(expected)+1]
        self.assertEqual(report['next_common_tail_rate'],{'numerator':following.numerator,'denominator':following.denominator})
        self.assertFalse(report['next_constraints_passed']);self.assertTrue(report['next_failure_groups'])

    def test_complete_ties_cannot_be_split_and_q_one_next_candidate_rejects_all(self):
        m=self.module();data,rows=self.data(m)
        data['score']=np.asarray([100*c+(0 if r[3]<2 else r[3]) for c,r in zip(data['closed'],rows)],np.float64)
        state,report=self.calibrate(m,data)
        self.assertEqual(report['common_tail_rate'],{'numerator':0,'denominator':1})
        np.testing.assert_array_equal(state['conditional_thresholds'],np.arange(6)*100)
        self.assertEqual(report['next_common_tail_rate'],{'numerator':1,'denominator':25})
        data['score']=data['closed'].astype(np.float64)
        state,report=self.calibrate(m,data)
        self.assertEqual(report['next_common_tail_rate'],{'numerator':1,'denominator':1})
        self.assertFalse(report['next_constraints_passed'])
        self.assertTrue(np.isfinite(report['next_conditional_thresholds']).all())
        self.assertTrue(np.all(np.asarray(report['next_conditional_thresholds'])>np.arange(6)))
        self.assertEqual(report['next_failure_groups']['overall']['macro_frr'],1.)

    def test_uses_predicted_not_true_class_and_wrong_accepted_labels_are_not_known_accuracy(self):
        m=self.module();data,_=self.data(m,wrong=True);state,report=self.calibrate(m,data)
        self.assertEqual(report['common_tail_rate'],{'numerator':1,'denominator':10})
        np.testing.assert_array_equal(state['conditional_thresholds'],[100*c+(c+1)*5 for c in range(6)])
        metrics=report['methods']['predicted_class']['groups']['overall']
        self.assertEqual(metrics['known_macro_acceptance'],.9);self.assertEqual(metrics['new_open_macro_accuracy'],0.)
        self.assertEqual(report['methods']['global_control']['groups']['overall']['new_open_macro_accuracy'],0.)
        score=np.full(6,250.,np.float64);closed=np.arange(6,dtype=np.int64)
        result=m.apply(score,closed,state)
        np.testing.assert_array_equal(result[:,1],score-state['conditional_thresholds'][closed])
        self.assertEqual(list(inspect.signature(m.apply).parameters),['score','closed','calibration'])
        with self.assertRaises(TypeError):m.apply(score,closed,state,y=np.arange(6))

    def test_missing_known_or_predicted_class_and_zero_rejection_infeasibility_fallback(self):
        m=self.module()
        for defect in ('known_group','predicted','closed_damage'):
            with self.subTest(defect=defect):
                data,_=self.data(m)
                if defect=='known_group':
                    mask=~((data['y']==0)&(data['rx']==1)&(data['scenario']==m.SCENES[0]))
                    data={k:v[mask] for k,v in data.items()}
                elif defect=='predicted':
                    data['closed'][data['closed']==5]=4;data['baseline']=data['closed'].copy()
                else:data['baseline']=data['y'].copy();data['closed']=(data['y']+1)%6
                state,report=self.calibrate(m,data)
                row=report['methods']['predicted_class']
                self.assertFalse(report['constraints_passed']);self.assertFalse(row['constraints_passed'])
                self.assertTrue(row['zero_rejection_fallback']);self.assertTrue(np.all(m.apply(data['score'],data['closed'],state)[:,1]>=0))
                if defect=='predicted':self.assertEqual(row['coverage']['missing_predicted_classes'],[5])
                if defect=='known_group':self.assertFalse(row['coverage']['known_groups_complete'])
                if defect=='closed_damage':self.assertFalse(row['no_rejection_feasible'])

    def test_bad_score_shape_labels_ids_and_deployment_inputs_are_rejected(self):
        m=self.module();data,_=self.data(m)
        for defect in ('score2d','nan','unknown','closed','baseline','ids','receiver','scene'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                changed={k:v.copy() for k,v in data.items()}
                if defect=='score2d':changed['score']=changed['score'][:,None]
                elif defect=='nan':changed['score'][0]=np.nan
                elif defect=='unknown':changed['y'][0]=6
                elif defect=='closed':changed['closed'][0]=-1
                elif defect=='baseline':changed['baseline'][0]=6
                elif defect=='ids':changed['ids'][0]=changed['ids'][1]
                elif defect=='receiver':changed['rx'][0]=26
                else:changed['scenario'][0]='invalid'
                self.calibrate(m,changed)
        state,_=self.calibrate(m,data)
        for score,closed in ((np.array([np.inf]),np.array([0])),(np.ones(2),np.array([0])),(np.ones(1),np.array([6]))):
            with self.subTest(score=score,closed=closed),self.assertRaises(ValueError):m.apply(score,closed,state)
        changed=dict(state,conditional_thresholds=np.zeros((6,1)))
        with self.assertRaises(ValueError):m.apply(np.ones(1),np.zeros(1,np.int64),changed)


if __name__=='__main__':unittest.main()
