"""P73 focused CPU-only formula and saved-output checks; zero model/IQ/CP work."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_as_marginal73_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p73_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


# The record fixture below is kept in the exact producer schema.
def toy_records(a,mode):
    records=[];ordered=[];probmean=mode=='AS_PROBMEAN_CE'
    for step in (1,2):
        ids=list(range(64*(step-1),64*step));labels=np.asarray([i%6 for i in ids]);groups=[0]*32+[1]*16+[2]*16
        first=np.zeros((64,6));first[np.arange(64),labels]=3.
        second=first.copy();second[:20]=0.;second[np.arange(20),(labels[:20]+1)%6]=4.
        pp=[np.exp(z-z.max(1,keepdims=True)) for z in (first,second)];pp=[p/p.sum(1,keepdims=True) for p in pp]
        pooled=(pp[0]+pp[1])/2;pred=[p.argmax(1) for p in (*pp,pooled)]
        ce=a.ce_aggregates(first,second,labels)
        def stats(mask):
            n=int(mask.sum());value=dict(count=n)
            value.update({k:float(v[mask].mean()) if n else None for k,v in ce.items()})
            value['used_CE']=value['probmean_CE' if probmean else 'mean_CE']
            value['mask_disagreement_count']=int(((pred[0]!=pred[1])&mask).sum())
            for i,p in enumerate((*pp,pooled)):
                key=('true_probability_mask0','true_probability_mask1','true_probability_pooled')[i]
                value[key]=float(p[np.arange(64),labels][mask].mean()) if n else None
                value[('mask0_correct','mask1_correct','pooled_correct')[i]]=int(((pred[i]==labels)&mask).sum())
            pool_correct=pred[2]==labels
            for i in (0,1):
                correct=pred[i]==labels
                value['pooled_vs_mask'+str(i)]={k:int((v&mask).sum()) for k,v in dict(repaired=~correct&pool_correct,
                    new_errors=correct&~pool_correct,both_correct=correct&pool_correct,both_wrong=~correct&~pool_correct).items()}
            return value
        value=stats(np.ones(64,bool));value['group_stats']=[dict(group=g,**stats(np.asarray(groups)==g)) for g in range(4)]
        delta=value['used_CE']-value['mean_CE'];correction=.25*delta
        slots=[i for i in range(128) if (i+step)%2!=0];physical=[[1,1,1,0,i] for i in ids]
        m=dict(value,method=mode,mode='PROBABILITY_MEAN_CE' if probmean else 'MEAN_CE',step=step,cumulative_step=4440+step,
            epoch=1,cumulative_epoch=21,label_smoothing=.01,coefficient=.25,
            satellite_CE_delta_unscaled=delta,total_CE_correction=correction,base_total_L=2.,used_total_L=2.+correction,
            base_mixed_CE=1.2,used_mixed_CE=1.2+.5*delta,base_mixed_RD=1.4,used_mixed_RD=1.4+.5*delta,
            base_L_before_shape=1.5,used_L_before_shape=1.5+correction,As_KL_unweighted=.1,mixed_KL=.2,KL_unchanged=True,
            forward_count=5,total_L_forwards=6,L_NN=768,all_As_CE_supervised=64,same_IQ_each_RD_pair=True,
            ordinary_raw_IQ_unchanged=True,As_origin_verified=True,selected_raw_indices=slots,used_pool_indices=ids,
            groups=groups,physical_ids=physical,extra_NN=0,new_channel_calls=0,
            prediction_changes_are_training_view_diagnostics=True,diagnostics_do_not_gate_or_weight=True,not_inference_ensemble=True)
        records.append(m)
        ordered.append(dict(step=step,used_pool_indices=ids,origin_verified=True,no_quality_weighting=True,order='ORIGINAL_GENERATION_ORDER',
            selected_raw_indices=slots,physical_ids=physical,source_labels=labels.tolist(),source_domains=[i%15 for i in ids],groups=groups,
            As_CE_weighted=value['used_CE'],As_CE_unweighted=value['mean_CE'],CE_correction_to_total=correction,
            used_total_L=2.+correction,original_total_L=2.,As_KL_quality_weighted=False))
    return records,ordered


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_exact_parent_and_four_logical_rows(self):
            rows=a.row_specs('/toy')
            self.assertEqual([r['key'] for r in rows],list(a.ARMS))
            self.assertEqual([r['row_id'] for r in rows],['B00-s2026092701','B01-s2026092701','C1-s2026092701','P1-s2026092701'])
            self.assertEqual(rows[1]['parent'],rows[2]['parent']);self.assertEqual(rows[2]['parent'],rows[3]['parent'])
            self.assertNotEqual(rows[0]['parent'],rows[2]['parent'])
            self.assertEqual(a.expected_new_NN(4),8703132)
            with self.assertRaises(ValueError):a.expected_new_NN(-1)

        def test_probability_mean_is_not_mean_logits_or_mean_ce(self):
            first=np.asarray([[5.,0,0,0,0,0],[0.,5,0,0,0,0]])
            second=np.asarray([[0.,5,0,0,0,0],[5.,0,0,0,0,0]])
            labels=np.asarray([0,1]);v=a.ce_aggregates(first,second,labels)
            self.assertTrue((v['Jensen_gap']>0).all())
            mean_logits=a.ce_aggregates((first+second)/2,(first+second)/2,labels)
            self.assertFalse(np.allclose(v['probmean_CE'],mean_logits['mean_CE']))
            same=a.ce_aggregates(first,first,labels)
            np.testing.assert_allclose(same['Jensen_gap'],0,atol=1e-14)
            extreme=a.ce_aggregates(first*1000,second*1000,labels)
            self.assertTrue(np.isfinite(extreme['probmean_CE']).all())

        def test_jensen_full_records_and_empty_groups(self):
            for mode in a.MODES.values():
                records,ordered=toy_records(a,mode)
                summary,flat=a.validate_marginal_records(records,mode,2)
                self.assertEqual(summary['steps'],2);self.assertEqual(summary['As_packets'],128)
                self.assertEqual(len(flat),2);self.assertEqual(sum(g['count'] for g in summary['quality_groups']),128)
                self.assertIsNone(summary['quality_groups'][3]['Jensen_gap'])
                self.assertTrue(a.validate_ordered_records(ordered,records,2)['one_use'])
                self.assertTrue(summary['Jensen_gap_is_not_identity_gain'])

        def test_missing_or_inconsistent_marginal_records_rejected(self):
            records,ordered=toy_records(a,'AS_PROBMEAN_CE')
            for key,value in [('Jensen_gap',-.1),('used_CE',8.),('total_CE_correction',1.),('extra_NN',1)]:
                bad=copy.deepcopy(records);bad[0][key]=value
                with self.assertRaises(ValueError):a.validate_marginal_records(bad,'AS_PROBMEAN_CE',2)
            bad=copy.deepcopy(records);bad[0].pop('group_stats')
            with self.assertRaises((ValueError,KeyError)):a.validate_marginal_records(bad,'AS_PROBMEAN_CE',2)
            bad=copy.deepcopy(ordered);bad[0]['used_pool_indices']=bad[1]['used_pool_indices']
            with self.assertRaises(ValueError):a.validate_ordered_records(bad,records,2)
            bad=copy.deepcopy(ordered);bad[0]['CE_correction_to_total']=3.
            with self.assertRaises(ValueError):a.validate_ordered_records(bad,records,2)
            with self.assertRaises(ValueError):a.ce_aggregates([[1,2]],[[1,2]],np.asarray([0]))

        def test_absolute_pp_source_thresholds_only(self):
            self.assertTrue(a.source_criteria(.799,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.79899,.8,.505,.5)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.50499,.5)['source_qualified'])
            self.assertEqual(len(a.source_criteria(.8,.8,.505,.5)['checks']),2)

        def test_source_decision_is_already_fixed_before_formal(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';p.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,model_seed_count=1,
                    NN=0,source_qualified=False,checks=dict(clean=False,midlow=True))
                p.write_text(json.dumps(value),encoding='utf8');self.assertEqual(a.read_source_decision(temp),value)
                for bad in (dict(value,formal_scores_read=True),dict(value,checks=dict(clean=True,midlow=True,extra=True))):
                    p.write_text(json.dumps(bad),encoding='utf8')
                    with self.assertRaises(ValueError):a.read_source_decision(temp)

        def test_primary_pair_and_single_seed_claims(self):
            rows={arm:{v:dict(accuracy=.8+i*.001,macro_f1=.79+i*.001,worst_RX=.7+i*.001,satellite_retention=.6+i*.001)
                for v in c.VIEWS} for i,arm in enumerate(a.ARMS)}
            value=dict(rows=rows);a.enrich_formal(value)
            self.assertEqual(value['primary']['control'],'AS_MEAN_CE')
            self.assertAlmostEqual(value['primary']['effects']['clean']['accuracy_delta_pp'],.1)
            self.assertFalse(value['stage']['AS_PROBMEAN_CE']['reproducible_dual_gain'])
            self.assertTrue(value['four_rows_are_not_a_two_by_two_factorial'])
            self.assertIsNone(c.metric(np.asarray([],int),np.asarray([],int))['accuracy'])

        def test_all_four_frozen_rows_before_any_truth(self):
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
                    if s['key']=='AS_PROBMEAN_CE':raise ValueError('not frozen')
                    seen.append(s['key']);return {}
                with patch.object(c,'read',side_effect=read),patch.object(c,'frozen_predictions',side_effect=incomplete),self.assertRaises(ValueError):
                    c.truth_last_arrays(a.row_specs(temp),ids,p,temp,6,lambda s,p:{})
                self.assertNotIn('truth',seen)
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':checks()
