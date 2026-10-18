"""Focused P70 analysis toys: no model, IQ, checkpoint, network or GPU."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_quality_risk70_analysis_20261018.py')
    spec=importlib.util.spec_from_file_location('_p70_analysis_under_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def toy_records(method='QUALITY_STRAT'):
    counts=np.asarray([5,6,6,7]); weights=counts/counts.sum(); offsets=np.cumsum([0,*counts.tolist()])
    remaining=[list(range(int(offsets[g]),int(offsets[g+1]))) for g in range(4)]
    quotas=([2,2,2,2],[2,2,2,2],[1,2,2,3]) if method=='QUALITY_STRAT' else ([5,3,0,0],[0,3,5,0],[0,0,1,7])
    records=[]
    for step,quota in enumerate(quotas,1):
        groups=[g for g,n in enumerate(quota) for _ in range(n)]; idx=[]
        for g,n in enumerate(quota):idx.extend(remaining[g][:n]); remaining[g]=remaining[g][n:]
        means=[(g+1)/3+step*.01 if n else None for g,n in enumerate(quota)]
        ordinary=sum((means[g]*n if n else 0.) for g,n in enumerate(quota))/8
        used=float(np.dot(weights,means)) if method=='QUALITY_STRAT' else ordinary
        correction=.25*(used-ordinary)
        records.append(dict(step=step,cumulative_step=4440+step,method=method,used_pool_indices=idx,
            groups=groups,counts=list(quota),p_g=weights.tolist(),group_CE=means,
            origin_steps=[i//8+1 for i in idx],origin_slots=[2*(i%8)+1-((i//8+1)%2) for i in idx],
            As_CE_unweighted=ordinary,As_CE_weighted=used,CE_correction_to_total=correction,
            original_total_L=1.2,used_total_L=1.2+correction,As_KL_unweighted=.01,
            ordinary64_CE=1.,ordinary64_KL=.01,same_IQ_each_RD_pair=True,all_labeled_CE_retained=True,
            As_KL_quality_weighted=False,U_truth_read=False,extra_NN=0))
    return counts,records


def checks(names=None):
    a=load_analysis(); c=a.c
    class Checks(unittest.TestCase):
        def test_actual_row_mapping_shared_as20_and_reference_roles(self):
            rows=a.row_specs('/toy')
            self.assertEqual([s['key'] for s in rows],list(a.ARMS))
            self.assertEqual([s['row_id'] for s in rows],['B00-s2026092701','B01-s2026092701','M1-s2026092701','M2-s2026092701'])
            self.assertEqual(rows[1]['parent'],rows[2]['parent']); self.assertEqual(rows[2]['parent'],rows[3]['parent'])
            self.assertNotEqual(rows[0]['parent'],rows[2]['parent']); self.assertEqual(a.PAIRS[2],('QUALITY_STRAT','POOL_RANDOM'))

        def test_same_pool_random_and_stratified_one_use_count_variance(self):
            counts,records=toy_records(); value,flat=a.validate_pool_steps(records,counts,'QUALITY_STRAT',3,8)
            self.assertEqual(value['unique_pool_observations'],24); self.assertTrue(value['one_use'])
            self.assertEqual(value['group_counts'],[5,6,6,7]); self.assertEqual(len(flat),3)
            counts,records=toy_records('POOL_RANDOM'); random,_=a.validate_pool_steps(records,counts,'POOL_RANDOM',3,8)
            self.assertGreater(sum(random['batch_count_variance']),sum(value['batch_count_variance']))
            self.assertEqual(random['CE_correction_mean'],0.)
            counts,records=toy_records(); counts=np.asarray([0,11,6,7]); weights=counts/24
            for r in records:
                old=r['counts']; means=r['group_CE']; merged=(old[0]*means[0]+old[1]*means[1])/(old[0]+old[1])
                r.update(counts=[0,old[0]+old[1],old[2],old[3]],groups=[1 if g==0 else g for g in r['groups']],
                    p_g=weights.tolist(),group_CE=[None,merged,means[2],means[3]])
                r['As_CE_weighted']=sum(weights[g]*r['group_CE'][g] for g in range(1,4))
                r['CE_correction_to_total']=.25*(r['As_CE_weighted']-r['As_CE_unweighted'])
                r['used_total_L']=r['original_total_L']+r['CE_correction_to_total']
            empty,_=a.validate_pool_steps(records,counts,'QUALITY_STRAT',3,8)
            self.assertEqual(empty['empty_groups'],[0]); self.assertEqual(empty['p_g'][0],0.)

        def test_repeat_missing_sparse_and_wrong_quality_all_rejected(self):
            counts,records=toy_records()
            bad=copy.deepcopy(records); bad[1]['used_pool_indices'][0]=bad[0]['used_pool_indices'][0]
            with self.assertRaises(ValueError):a.validate_pool_steps(bad,counts,'QUALITY_STRAT',3,8)
            with self.assertRaises(ValueError):a.validate_pool_steps(records[:-1],counts,'QUALITY_STRAT',3,8)
            with self.assertRaises(ValueError):a.validate_pool_steps(records,[2,7,7,8],'QUALITY_STRAT',3,8)
            bad=copy.deepcopy(records); bad[0]['counts']=[1,3,2,2]
            with self.assertRaises(ValueError):a.validate_pool_steps(bad,counts,'QUALITY_STRAT',3,8)

        def test_fixed_p_group_means_and_only_quarter_CE_correction(self):
            counts,records=toy_records()
            for key,value in [('As_CE_weighted',9.),('CE_correction_to_total',9.),('used_total_L',9.)]:
                bad=copy.deepcopy(records); bad[0][key]=value
                with self.assertRaises(ValueError):a.validate_pool_steps(bad,counts,'QUALITY_STRAT',3,8)
            bad=copy.deepcopy(records); bad[1]['p_g']=[.25]*4
            with self.assertRaises(ValueError):a.validate_pool_steps(bad,counts,'QUALITY_STRAT',3,8)
            bad=copy.deepcopy(records); bad[0]['same_IQ_each_RD_pair']=False
            with self.assertRaises(ValueError):a.validate_pool_steps(bad,counts,'QUALITY_STRAT',3,8)
            bad=copy.deepcopy(records); bad[0]['origin_slots'][0]+=1
            with self.assertRaises(ValueError):a.validate_pool_steps(bad,counts,'QUALITY_STRAT',3,8)

        def test_actual_reference_decomposition_removes_only_same_logits_ce_correction(self):
            _,records=toy_records(); r=records[0]; d=r['CE_correction_to_total']
            reference=dict(total_L=r['used_total_L'],base_L_before_shape=r['used_total_L']-.2,
                extra_CE=.5*(r['ordinary64_CE']+r['As_CE_weighted']),
                extra_KL=.5*(r['ordinary64_KL']+r['As_KL_unweighted']))
            result=a.undo_quality_CE(reference,r)
            self.assertAlmostEqual(result['total_L'],r['original_total_L'])
            self.assertAlmostEqual(result['extra_CE'],.5*(r['ordinary64_CE']+r['As_CE_unweighted']))
            self.assertEqual(result['extra_KL'],reference['extra_KL']); self.assertEqual(reference['total_L'],r['used_total_L'])
            with self.assertRaises(ValueError):a.undo_quality_CE(dict(reference,extra_KL=9.),r)

        def test_source_thresholds_compare_m2_to_m1_absolute_percentage_points(self):
            good=a.source_criteria(.799,.8,.505,.5)
            self.assertTrue(good['source_qualified']); self.assertAlmostEqual(good['clean_delta_pp'],-.1)
            self.assertAlmostEqual(good['midlow_delta_pp'],.5)
            self.assertFalse(a.source_criteria(.79899,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.50499,.5)['source_qualified'])

        def test_stored_decision_must_precede_formal_and_only_two_criteria(self):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json'; path.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,NN=0,
                    model_seed_count=1,source_qualified=False,checks=dict(clean=False,midlow=True))
                path.write_text(json.dumps(value),encoding='utf8'); self.assertEqual(a.read_source_decision(temp),value)
                for bad in (dict(value,formal_scores_read=True),dict(value,checks=dict(clean=True,midlow=True,extra=True))):
                    path.write_text(json.dumps(bad),encoding='utf8')
                    with self.assertRaises(ValueError):a.read_source_decision(temp)

        def test_formal_primary_not_false_factorial_or_reproducibility(self):
            rows={arm:{v:dict(accuracy=.8+i*.001,macro_f1=.79+i*.001,worst_RX=.7+i*.001,satellite_retention=.6+i*.001)
                for v in c.VIEWS} for i,arm in enumerate(a.ARMS)}
            obj=dict(rows=rows); a.enrich_formal(obj)
            self.assertEqual(obj['primary']['control'],'POOL_RANDOM')
            self.assertAlmostEqual(obj['primary']['effects']['clean']['accuracy_delta_pp'],.1)
            self.assertTrue(obj['four_rows_are_not_a_two_by_two_factorial'])
            self.assertFalse(obj['stage']['QUALITY_STRAT']['reproducible_dual_gain'])
            self.assertIsNone(c.metric(np.asarray([],int),np.asarray([],int))['accuracy'])

        def test_truth_last_waits_for_all_four_frozen_rows(self):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'truth.json'; ids=np.asarray(['a','b','c','d','e','f'])
                path.write_text(json.dumps({str(v):dict(label=i,receiver='toy',day='toy') for i,v in enumerate(ids)}),encoding='utf8')
                seen=[]; original=c.read
                def read(p):seen.append('truth'); return original(p)
                def frozen(s,done,index,count):seen.append(s['key']); return dict(ids=index)
                specs=a.row_specs(temp)
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=frozen):
                    c.truth_last_arrays(specs,ids,path,temp,6,lambda s,p:{})
                self.assertEqual(seen,[*a.ARMS,'truth'])
                seen.clear()
                def incomplete(s,done,index,count):
                    if s['key']=='QUALITY_STRAT':raise ValueError('incomplete frozen prediction')
                    seen.append(s['key']); return {}
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=incomplete),self.assertRaises(ValueError):
                    c.truth_last_arrays(specs,ids,path,temp,6,lambda s,p:{})
                self.assertNotIn('truth',seen)
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':checks()
