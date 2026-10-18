"""P72 focused CPU-only saved-output checks; zero model/IQ/CP/network work."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_raw_as_mask72_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p72_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def toy_mask_records(a,mode):
    records=[];ordered=[]
    for step in (1,2):
        seeds={key:step*10+i for i,key in enumerate(('As0','As1','raw0','raw1'))};shared=mode=='SHARED_RAW_AS_FUSE'
        calls=[dict(index=0,role='ordinary',overridden_rows=0,native_dropout_executed=True)]
        for i in range(1,5):
            role,pair=('As',i-1) if i<=2 else ('raw',i-3);used='As' if shared and role=='raw' else role
            calls.append(dict(index=i,role=role,pair=pair,overridden_rows=64,native_dropout_executed=True,
                used_mask_role=used,used_mask_seed=seeds[used+str(pair)],independent_role_seed=seeds[role+str(pair)]))
        ids=list(range(64*(step-1),64*step));slots=[i for i in range(128) if (i+step)%2!=0]
        m=dict(method=mode,logical_method='MASK_SHARED' if shared else 'MASK_INDEPENDENT',step=step,cumulative_step=4440+step,
            epoch=1,cumulative_epoch=21,target=a.TARGET,dropout_p=.45,calls=5,native_dropout_calls=5,native_dropout_slots=102400,
            private_draws=4,overridden_calls=4,private_draw_slots=40960,overridden_slots=40960,overridden_rows_per_call=64,
            mask_seeds=seeds,private_kept_counts={k:5600+i for i,k in enumerate(seeds)},As_raw_pair_hamming=[0,0] if shared else [5000,4999],
            As_RD_hamming=5000,raw_RD_hamming=5000 if shared else 5001,unused_raw_RD_hamming=5001,
            paired_same=shared,RD_different=True,call_records=calls,As_origin_verified=True,As_origin_count=64,
            same_IQ_each_RD_pair=True,As_redraws=0,new_channel_calls=0,extra_NN=0,persistent_mask_state=0,new_parameters=0,CE_KL_unchanged=True,
            selected_raw_indices=slots,used_pool_indices=ids,physical_ids=[[1,1,1,0,i] for i in ids],source_labels=[i%6 for i in ids],source_domains=[i%15 for i in ids])
        records.append(m)
        ordered.append(dict(step=step,used_pool_indices=ids,origin_verified=True,no_quality_weighting=True,order='ORIGINAL_GENERATION_ORDER',
            selected_raw_indices=slots,physical_ids=m['physical_ids'],source_labels=m['source_labels'],source_domains=m['source_domains'],
            As_CE_weighted=1.,As_CE_unweighted=1.,CE_correction_to_total=0.,used_total_L=1.,original_total_L=1.,As_KL_quality_weighted=False))
    return records,ordered


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_exact_physical_parent_and_logical_rows(self):
            rows=a.row_specs('/toy')
            self.assertEqual([r['key'] for r in rows],list(a.ARMS))
            self.assertEqual([r['row_id'] for r in rows],['B00-s2026092701','B01-s2026092701','I1-s2026092701','S1-s2026092701'])
            self.assertEqual(rows[1]['parent'],rows[2]['parent']);self.assertEqual(rows[2]['parent'],rows[3]['parent'])
            self.assertNotEqual(rows[0]['parent'],rows[2]['parent'])
            self.assertEqual(a.MODES['MASK_SHARED'],'SHARED_RAW_AS_FUSE')

        def test_two_tail_cost_not_extra_model_forward(self):
            self.assertEqual(a.expected_new_NN(4),8703132)
            with self.assertRaises(ValueError):a.expected_new_NN(-1)

        def test_existing_fusion_raw_as_shared_but_two_rdrop_draws_independent(self):
            for mode in a.MODES.values():
                records,ordered=toy_mask_records(a,mode);value,flat=a.validate_mask_records(records,mode,2)
                self.assertEqual(value['target'],a.TARGET);self.assertEqual(value['private_draw_slots'],81920)
                self.assertEqual(value['native_dropout_calls'],10);self.assertEqual(len(flat),2)
                self.assertTrue(a.validate_ordered_records(ordered,records,2)['one_use'])
                if mode=='SHARED_RAW_AS_FUSE':self.assertEqual(value['epoch_summaries'][0]['raw_As_equal_fraction'],1.)
                self.assertLess(value['epoch_summaries'][0]['RDrop_pair_equal_fraction'],1.)

        def test_wrong_target_pair_seed_origin_or_ce_change_rejected(self):
            records,ordered=toy_mask_records(a,'SHARED_RAW_AS_FUSE')
            for key,value in [('target','wrong.joint.layer'),('dropout_p',.2),('As_raw_pair_hamming',[1,0]),('extra_NN',1)]:
                bad=copy.deepcopy(records);bad[0][key]=value
                with self.assertRaises(ValueError):a.validate_mask_records(bad,'SHARED_RAW_AS_FUSE',2)
            bad=copy.deepcopy(records);bad[0]['mask_seeds']['As1']=bad[0]['mask_seeds']['As0']
            with self.assertRaises(ValueError):a.validate_mask_records(bad,'SHARED_RAW_AS_FUSE',2)
            bad=copy.deepcopy(ordered);bad[0]['selected_raw_indices']=bad[0]['selected_raw_indices'][::-1]
            with self.assertRaises(ValueError):a.validate_ordered_records(bad,records,2)
            bad=copy.deepcopy(ordered);bad[0]['used_pool_indices']=bad[1]['used_pool_indices']
            with self.assertRaises(ValueError):a.validate_ordered_records(bad,records,2)
            bad=copy.deepcopy(ordered);bad[0]['As_CE_weighted']=2.
            with self.assertRaises(ValueError):a.validate_ordered_records(bad,records,2)

        def test_absolute_pp_source_thresholds_only(self):
            self.assertTrue(a.source_criteria(.799,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.79899,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.50499,.5)['source_qualified'])
            self.assertEqual(len(a.source_criteria(.8,.8,.505,.5)['checks']),2)

        def test_saved_source_decision_must_precede_formal(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';p.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,model_seed_count=1,
                    NN=0,source_qualified=False,checks=dict(clean=False,midlow=True))
                p.write_text(json.dumps(value),encoding='utf8');self.assertEqual(a.read_source_decision(temp),value)
                for bad in (dict(value,formal_scores_read=True),dict(value,checks=dict(clean=True,midlow=True,extra=True))):
                    p.write_text(json.dumps(bad),encoding='utf8')
                    with self.assertRaises(ValueError):a.read_source_decision(temp)

        def test_paired_primary_and_single_seed_claims(self):
            rows={arm:{v:dict(accuracy=.8+i*.001,macro_f1=.79+i*.001,worst_RX=.7+i*.001,satellite_retention=.6+i*.001)
                for v in c.VIEWS} for i,arm in enumerate(a.ARMS)}
            value=dict(rows=rows);a.enrich_formal(value)
            self.assertEqual(value['primary']['control'],'MASK_INDEPENDENT')
            self.assertAlmostEqual(value['primary']['effects']['clean']['accuracy_delta_pp'],.1)
            self.assertFalse(value['stage']['MASK_SHARED']['reproducible_dual_gain'])
            self.assertTrue(value['four_rows_are_not_a_two_by_two_factorial'])
            self.assertIsNone(c.metric(np.asarray([],int),np.asarray([],int))['accuracy'])

        def test_all_four_frozen_prediction_rows_before_truth(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'truth.json';ids=np.asarray(list('abcdef'))
                p.write_text(json.dumps({v:dict(label=i,receiver='toy',day='toy') for i,v in enumerate(ids)}),encoding='utf8')
                seen=[];old=c.read
                def read(path):seen.append('truth');return old(path)
                def frozen(s,done,index,count):seen.append(s['key']);return {}
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=frozen):
                    c.truth_last_arrays(a.row_specs(temp),ids,p,temp,6,lambda s,p:{})
                self.assertEqual(seen,[*a.ARMS,'truth']);seen.clear()
                def incomplete(s,done,index,count):
                    if s['key']=='MASK_SHARED':raise ValueError('not frozen')
                    seen.append(s['key']);return {}
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=incomplete),self.assertRaises(ValueError):
                    c.truth_last_arrays(a.row_specs(temp),ids,p,temp,6,lambda s,p:{})
                self.assertNotIn('truth',seen)
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':checks()
