"""P71 pure-CPU risk/readout toys; zero model/IQ/checkpoint/network work."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_bounded_risk71_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p71_under_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def toy_records(a):
    counts=np.asarray([5,6,6,7]); p=counts/counts.sum(); pi=a.risk_weights(p)['pi_g']
    offsets=np.cumsum([0,*counts.tolist()]); remaining=[list(range(int(offsets[g]),int(offsets[g+1]))) for g in range(4)]
    records=[]; nominal=[]
    for step,quota in enumerate(([2,2,2,2],[2,2,2,2],[1,2,2,3]),1):
        idx=[]
        for g,n in enumerate(quota):idx.extend(remaining[g][:n]); remaining[g]=remaining[g][n:]
        groups=[g for g,n in enumerate(quota) for _ in range(n)]
        means=[(g+1)/3+step*.01 for g in range(4)]
        plain=sum(means[g]*n for g,n in enumerate(quota))/8; nm=float(np.dot(p,means)); risk=float(np.dot(pi,means))
        r=dict(step=step,cumulative_step=4440+step,epoch=1,cumulative_epoch=21,method='BOUNDED_RISK',allocation_method='QUALITY_STRAT',
            used_pool_indices=idx,groups=groups,counts=list(quota),p_g=p.tolist(),v_g=list(a.V_G),pi_g=pi,gamma=a.GAMMA,group_CE=means,
            origin_steps=[i//8+1 for i in idx],origin_slots=[2*(i%8)+1-((i//8+1)%2) for i in idx],
            As_CE_unweighted=plain,As_CE_weighted=risk,nominal_M2_As_CE=nm,used_risk_As_CE=risk,
            original_total_L=1.2,used_total_L=1.2+.25*(risk-plain),CE_correction_to_total=.25*(risk-plain),
            risk_vs_nominal_total_correction=.25*(risk-nm),nominal_M2_same_logits_total_L=1.2+.25*(nm-plain),
            As_KL_unweighted=.01,ordinary64_CE=1.,ordinary64_KL=.01,same_IQ_each_RD_pair=True,
            all_labeled_CE_retained=True,As_KL_quality_weighted=False,U_truth_read=False,extra_NN=0)
        records.append(r); n=copy.deepcopy(r); n.update(method='QUALITY_STRAT',As_CE_weighted=nm,
            CE_correction_to_total=.25*(nm-plain),used_total_L=n['nominal_M2_same_logits_total_L']); nominal.append(n)
    return counts,records,nominal


def checks(names=None):
    a=load_analysis(); c=a.c
    class Checks(unittest.TestCase):
        def test_fixed_gamma_v_pi_mass_support_and_true_pool_weights(self):
            p=np.asarray([21825,21572,12517,86166])/142080
            value=a.risk_weights(p)
            self.assertAlmostEqual(sum(value['pi_g']),1.)
            self.assertTrue(np.allclose(value['pi_g'],.75*p+.25*np.asarray([.15,.40,.25,.20])))
            self.assertAlmostEqual(sum(value['As_CE_group_coefficients']),.25)
            self.assertGreater(value['relative_to_nominal'][1],1.)
            with self.assertRaises(ValueError):a.risk_weights([0.,.3,.3,.4])
            with self.assertRaises(ValueError):a.risk_weights([.2]*4)

        def test_one_shared_as20_parent_exact_reference_rows(self):
            rows=a.row_specs('/toy')
            self.assertEqual([r['key'] for r in rows],list(a.ARMS))
            self.assertEqual([r['row_id'] for r in rows],['B00-s2026092701','B01-s2026092701','M2-s2026092701','M3-s2026092701'])
            self.assertTrue(all(r['reused'] for r in rows[:3])); self.assertFalse(rows[3]['reused'])
            self.assertEqual(rows[1]['parent'],rows[2]['parent']); self.assertEqual(rows[2]['parent'],rows[3]['parent'])
            self.assertEqual(rows[2]['run'],a.POOL_RUN); self.assertEqual(rows[3]['run'],a.RUN)

        def test_risk_group_ce_nominal_counterfactual_and_one_use(self):
            counts,records,nominal=toy_records(a)
            value,flat=a.validate_risk_steps(records,counts,nominal,3,8)
            self.assertEqual(value['packets'],24); self.assertTrue(value['same_M2_pool_order'])
            self.assertFalse(value['nominal_risk_unbiased_claimed']); self.assertEqual(len(flat),3)
            self.assertNotEqual(value['risk_CE_mean'],value['nominal_M2_same_logits_CE_mean'])
            self.assertTrue(all(abs(r['risk_vs_nominal_total_correction']-.25*(r['risk_CE']-r['nominal_M2_CE']))<1e-12 for r in flat))

        def test_wrong_gamma_pi_counterfactual_or_order_is_rejected(self):
            counts,records,nominal=toy_records(a)
            for key,value in [('gamma',.5),('pi_g',[.25]*4),('risk_vs_nominal_total_correction',9.),('nominal_M2_As_CE',9.),('nominal_M2_same_logits_total_L',9.)]:
                bad=copy.deepcopy(records); bad[0][key]=value
                with self.assertRaises(ValueError):a.validate_risk_steps(bad,counts,nominal,3,8)
            bad=copy.deepcopy(records); bad[0]['used_pool_indices']=bad[0]['used_pool_indices'][::-1]
            with self.assertRaises(ValueError):a.validate_risk_steps(bad,counts,nominal,3,8)

        def test_original_plain_counterfactual_is_not_reused_m2_loss(self):
            _,records,_=toy_records(a); r=records[0]
            self.assertNotEqual(r['original_total_L'],r['nominal_M2_same_logits_total_L'])
            normalized=a.nominal_record(r)
            self.assertAlmostEqual(normalized['used_total_L'],r['nominal_M2_same_logits_total_L'])
            self.assertAlmostEqual(normalized['As_CE_weighted'],r['nominal_M2_As_CE'])
            self.assertNotEqual(normalized['used_total_L'],r['used_total_L'])

        def test_source_two_thresholds_absolute_pp_and_no_extra_gate(self):
            self.assertTrue(a.source_criteria(.799,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.79899,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.50499,.5)['source_qualified'])
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json'; p.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,NN=0,
                    model_seed_count=1,source_qualified=False,checks=dict(clean=False,midlow=True))
                p.write_text(json.dumps(value),encoding='utf8'); self.assertEqual(a.read_source_decision(temp),value)
                p.write_text(json.dumps(dict(value,formal_scores_read=True)),encoding='utf8')
                with self.assertRaises(ValueError):a.read_source_decision(temp)

        def test_single_new_row_cost_and_formal_comparison_not_factorial(self):
            self.assertEqual(a.expected_new_NN(4),4351568)
            rows={arm:{v:dict(accuracy=.8+i*.001,macro_f1=.79+i*.001,worst_RX=.7+i*.001,satellite_retention=.6+i*.001)
                for v in c.VIEWS} for i,arm in enumerate(a.ARMS)}
            value=dict(rows=rows); a.enrich_formal(value)
            self.assertEqual(value['primary']['control'],'QUALITY_STRAT')
            self.assertAlmostEqual(value['primary']['effects']['clean']['accuracy_delta_pp'],.1)
            self.assertFalse(value['stage']['BOUNDED_RISK']['reproducible_dual_gain'])
            self.assertTrue(value['four_rows_are_not_a_two_by_two_factorial'])

        def test_truth_last_all_four_rows_before_only_truth_read(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'truth.json'; ids=np.asarray(list('abcdef'))
                p.write_text(json.dumps({v:dict(label=i,receiver='toy',day='toy') for i,v in enumerate(ids)}),encoding='utf8')
                seen=[]; original=c.read
                def read(path):seen.append('truth'); return original(path)
                def frozen(s,done,index,count):seen.append(s['key']); return {}
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=frozen):
                    c.truth_last_arrays(a.row_specs(temp),ids,p,temp,6,lambda s,p:{})
                self.assertEqual(seen,[*a.ARMS,'truth'])
                seen.clear()
                def incomplete(s,done,index,count):
                    if s['key']=='BOUNDED_RISK':raise ValueError('incomplete')
                    seen.append(s['key']); return {}
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=incomplete),self.assertRaises(ValueError):
                    c.truth_last_arrays(a.row_specs(temp),ids,p,temp,6,lambda s,p:{})
                self.assertNotIn('truth',seen)
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':checks()
