"""Source-only OE budget allocation and exact original-F0 risk regressions."""
import importlib
import inspect
import json
import unittest

import numpy as np


class StarOeRiskBudgetTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_oe_risk_budget')
    def data(self,n=50):
        from experiments.pamf_phase3_open_world import star_consensus_readout as risk
        rows=[(y,r,s,k) for y in range(6) for r in risk.RECEIVERS for s in risk.SCENES for k in range(n)]
        y=np.asarray([t[0] for t in rows],np.int64)
        known=dict(ids=np.asarray(['V-'+str(i) for i in range(len(rows))]),y=y,closed=y.copy(),baseline=y.copy(),
            score=np.asarray([t[3] for t in rows],np.float64),rx=np.asarray([t[1] for t in rows],np.int64),
            scenario=np.asarray([t[2] for t in rows]))
        oe=dict(oe_ids=np.asarray(['OE-'+str(i) for i in range(15)]),oe_closed=np.zeros(15,np.int64),
            oe_score=np.full(15,5.5,np.float64),oe_domain=np.arange(15,dtype=np.int64))
        return known|oe,rows
    def calibrate(self,m,d):return m.calibrate(**d)

    def test_independent_class_budget_rejects_more_OE_than_common_tail_under_same_42_risks(self):
        m=self.module();d,_=self.data()
        from experiments.pamf_phase3_open_world import star_class_risk as common
        old,old_report=common.calibrate_known(*(d[k] for k in ('ids','y','closed','baseline','score','rx','scenario')))
        state,report=self.calibrate(m,d)
        self.assertTrue(old_report['methods']['predicted_class']['constraints_passed']);self.assertTrue(report['constraints_passed'])
        self.assertEqual(np.mean(common.apply(d['oe_score'],d['oe_closed'],old)[:,1]<0),0.)
        self.assertEqual(np.mean(m.apply(d['oe_score'],d['oe_closed'],state)<0),1.)
        self.assertEqual(report['objective_before'],0.);self.assertEqual(report['objective_after'],1.)
        self.assertEqual(len(report['groups']),42);self.assertEqual(set(state),{'conditional_thresholds'})
        self.assertEqual(state['conditional_thresholds'][0],np.nextafter(5.5,np.inf))
        for group in report['groups'].values():
            self.assertTrue(group['constraints_passed']);self.assertEqual(group['original_closed_minus_new_open'],.02)
        self.assertTrue(report['maximal']);self.assertFalse(report['global_optimum_claimed']);json.dumps(report,allow_nan=False)

    def test_original_F0_budget_and_zero_remaining_loss_allow_only_wrong_rejection(self):
        m=self.module();d,rows=self.data();wrong=(d['y']==0)&(np.asarray([r[3] for r in rows])<6)
        d['closed'][wrong]=1;d['score'][:]=10;d['score'][wrong]=0
        d.update(oe_ids=np.asarray(['OE-'+str(i) for i in range(30)]),oe_closed=np.tile(np.array([0,1],np.int64),15),
            oe_score=np.tile(np.array([10.,0.]),15),oe_domain=np.repeat(np.arange(15,dtype=np.int64),2))
        state,report=self.calibrate(m,d)
        self.assertTrue(report['constraints_passed']);self.assertAlmostEqual(report['objective_after'],.5)
        self.assertEqual([row['class_index'] for row in report['trace']],[1])
        for group in report['groups'].values():
            self.assertEqual(group['original_closed_minus_new_open'],.02);self.assertEqual(group['correct_rejection_loss'],0.)
        self.assertGreaterEqual(state['conditional_thresholds'][1],np.nextafter(0.,np.inf))
        self.assertTrue(report['next_candidates'][0]['failure_groups'])

    def test_domain_macro_gain_outweighs_large_single_domain_and_ties_are_whole(self):
        m=self.module();d,rows=self.data();k=np.asarray([r[3] for r in rows]);d['score']=np.where(k<6,0.,10.)
        d.update(oe_ids=np.asarray(['OE-'+str(i) for i in range(114)]),oe_closed=np.r_[np.zeros(100,np.int64),np.ones(14,np.int64)],
            oe_score=np.zeros(114,np.float64),oe_domain=np.r_[np.zeros(100,np.int64),np.arange(1,15,dtype=np.int64)])
        state,report=self.calibrate(m,d)
        self.assertTrue(report['constraints_passed']);self.assertEqual(report['trace'][0]['class_index'],1)
        self.assertAlmostEqual(report['objective_after'],14/15);self.assertEqual(len(report['trace']),1)
        self.assertTrue(np.all(m.apply(d['score'][d['closed']==1],d['closed'][d['closed']==1],state)[k[d['closed']==1]<6]<0))
        self.assertTrue(np.all(m.apply(d['oe_score'][:100],d['oe_closed'][:100],state)>=0))

    def test_zero_risk_absent_known_predicted_class_is_legal_and_order_is_deterministic(self):
        m=self.module();d,_=self.data();d['closed'][d['closed']==5]=4;d['baseline']=d['closed'].copy();d['score'][:]=10
        d['oe_closed'][:]=5;d['oe_score'][:]=0
        state,report=self.calibrate(m,d)
        self.assertTrue(report['constraints_passed']);self.assertEqual(report['objective_after'],1.)
        self.assertEqual(report['groups']['overall']['macro_frr'],0.)
        self.assertEqual(report['trace'][0]['cost'],0.)
        rng=np.random.default_rng(34);a=rng.permutation(len(d['ids']));b=rng.permutation(len(d['oe_ids']))
        changed={k:v[b if k.startswith('oe_') else a] for k,v in d.items()}
        state2,report2=self.calibrate(m,changed)
        np.testing.assert_array_equal(state['conditional_thresholds'],state2['conditional_thresholds']);self.assertEqual(report['trace'],report2['trace'])

    def test_equal_candidate_ratio_uses_class_index_and_never_splits_known_tie(self):
        m=self.module();d,rows=self.data();k=np.asarray([r[3] for r in rows]);d['score']=np.where(k<6,0.,10.)
        d.update(oe_ids=np.asarray(['OE-'+str(i) for i in range(30)]),oe_closed=np.tile(np.array([0,1],np.int64),15),
            oe_score=np.zeros(30,np.float64),oe_domain=np.repeat(np.arange(15,dtype=np.int64),2))
        state,report=self.calibrate(m,d);self.assertEqual(report['trace'][0]['class_index'],0);self.assertEqual(len(report['trace']),1)
        d['score']=np.where(k<7,0.,10.);state,report=self.calibrate(m,d)
        self.assertEqual(report['trace'],[]);self.assertEqual(report['objective_after'],0.)
        self.assertTrue(np.all(m.apply(d['score'],d['closed'],state)>=0))

    def test_current_remaining_group_budget_changes_next_choice_and_trace_stays_small(self):
        m=self.module();d,rows=self.data();k=np.asarray([r[3] for r in rows]);d['score'][:]=100.
        scene=m.SCENES[0]
        for c,receiver,count in ((0,1,3),(1,1,2),(2,3,3)):
            d['score'][(d['y']==c)&(d['rx']==receiver)&(d['scenario']==scene)&(k<count)]=0.
        d.update(oe_ids=np.asarray(['OE-'+str(i) for i in range(1500)]),
            oe_closed=np.tile(np.repeat(np.arange(3,dtype=np.int64),[50,21,29]),15),
            oe_score=np.zeros(1500,np.float64),oe_domain=np.repeat(np.arange(15,dtype=np.int64),100))
        state,report=self.calibrate(m,d)
        self.assertTrue(report['constraints_passed']);self.assertEqual([s['class_index'] for s in report['trace']],[0,2,1])
        self.assertTrue(all(not ({'groups','proposal','risks'}&set(s)) for s in report['trace']))
        rng=np.random.default_rng(57);a=rng.permutation(len(d['ids']));b=rng.permutation(len(d['oe_ids']))
        other={key:values[b if key.startswith('oe_') else a] for key,values in d.items()}
        state2,report2=self.calibrate(m,other)
        np.testing.assert_array_equal(state['conditional_thresholds'],state2['conditional_thresholds']);self.assertEqual(report['trace'],report2['trace'])

    def test_missing_true_class_or_initial_original_damage_uses_finite_accept_all_false(self):
        m=self.module()
        for defect in ('coverage','damage'):
            with self.subTest(defect=defect):
                d,rows=self.data()
                if defect=='coverage':
                    keep=~((d['y']==0)&(d['rx']==1)&(d['scenario']==d['scenario'][0]))
                    d={k:v if k.startswith('oe_') else v[keep] for k,v in d.items()}
                else:
                    wrong=(d['y']==0)&(np.asarray([r[3] for r in rows])<7);d['closed'][wrong]=1
                state,report=self.calibrate(m,d)
                self.assertFalse(report['constraints_passed']);self.assertTrue(report['zero_rejection_fallback']);self.assertEqual(report['trace'],[])
                self.assertTrue(np.isfinite(state['conditional_thresholds']).all())
                self.assertTrue(np.all(m.apply(d['score'],d['closed'],state)>=0));self.assertTrue(np.all(m.apply(d['oe_score'],d['oe_closed'],state)>=0))

    def test_invalid_inputs_and_no_TX_or_unknown_count_fit_or_apply_argument(self):
        m=self.module();d,_=self.data()
        for defect in ('nan','closed','y','baseline','oe_class','domain','missing_domain','ids','cross_ids','shape'):
            bad={k:v.copy() for k,v in d.items()}
            if defect=='nan':bad['oe_score'][0]=np.nan
            elif defect=='closed':bad['closed'][0]=6
            elif defect=='y':bad['y'][0]=-1
            elif defect=='baseline':bad['baseline'][0]=7
            elif defect=='oe_class':bad['oe_closed'][0]=6
            elif defect=='domain':bad['oe_domain'][0]=15
            elif defect=='missing_domain':bad['oe_domain'][0]=1
            elif defect=='ids':bad['ids'][0]=bad['ids'][1]
            elif defect=='cross_ids':bad['oe_ids'][0]=bad['ids'][0]
            else:bad['score']=bad['score'][:,None]
            with self.subTest(defect=defect),self.assertRaises(ValueError):self.calibrate(m,bad)
        self.assertEqual(list(inspect.signature(m.calibrate).parameters),list(d))
        self.assertEqual(list(inspect.signature(m.apply).parameters),['score','closed','state'])
        with self.assertRaises(TypeError):m.calibrate(**d,oe_tx=np.zeros(15))
        state,_=self.calibrate(m,d)
        with self.assertRaises(TypeError):m.apply(np.ones(1),np.zeros(1,np.int64),state,y=np.zeros(1))
        for bad in ({'conditional_thresholds':np.zeros(5)},dict(state,OE_rows=np.zeros(2))):
            with self.assertRaises(ValueError):m.apply(np.ones(1),np.zeros(1,np.int64),bad)


if __name__=='__main__':unittest.main()
