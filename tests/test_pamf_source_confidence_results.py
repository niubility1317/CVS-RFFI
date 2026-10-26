import copy
import itertools
import unittest
import numpy as np
from experiments.pamf_u_history.source_confidence_metrics import grouped
from experiments.pamf_u_history.source_confidence_results import RUN,METHODS,SEEDS,summarize,collection_script


def fixture():
    metadata=np.array(list(itertools.product(range(6),(1,3,4,6,8),(1,2,3))))
    y,rx,day=np.repeat(metadata,300,axis=0).T
    logits=np.full((27000,6),-5.);logits[np.arange(27000),y]=5.
    groups=grouped(logits,y,rx,day)
    rows=[];checks=[]
    for m,s in itertools.product(METHODS,SEEDS):
        rows.append(dict(method=m,model_seed=s,source_role='V',V_seen=27000,source_L_smoke_seen=2,device='cpu',FP32=True,
            parameters_and_buffers_unchanged=True,optimizer_updates=0,U_TX_truth_reads=0,target_reads=0,G_weight_reads=0,G_forwards=0,rows=copy.deepcopy(groups)))
        checks.append(dict(method=m,model_seed=s,count=27000,exact_ID_label_RX_day_order=True))
    return dict(status='ALL_TWELVE_CPU_SOURCE_ROWS_COLLECTED',run_id=RUN,rows=rows,metadata_checks=checks)


class ConfidenceResultsTest(unittest.TestCase):
    def test_complete_three_seed_population_and_undefined_error_fraction(self):
        r,t=summarize(fixture())
        self.assertEqual(r['models'],12);self.assertEqual(len(t['per_seed_strata']),360)
        self.assertTrue(all(v['seeds']==3 for v in r['overall']))
        x=next(v for v in r['overall'] if v['method']=='SIG1-PL' and v['metric']=='rejected_share_of_all_errors_pct')
        self.assertEqual(x['defined_seeds'],0);self.assertIsNone(x['mean']);self.assertFalse(r['causal_generalization_claim'])

    def test_missing_seed_and_unpaired_population_rejected(self):
        b=fixture();b['rows'].pop()
        with self.assertRaisesRegex(ValueError,'all three seeds'):summarize(b)
        b=fixture();b['metadata_checks'][0]['exact_ID_label_RX_day_order']=False
        with self.assertRaisesRegex(ValueError,'pairing'):summarize(b)

    def test_never_drop_seed_with_undefined_accepted_precision(self):
        b=fixture();row=b['rows'][0]
        for g in row['rows']:
            g.update(accepted=0,accepted_correct=0,rejected=g['count'],coverage_pct=0,accepted_precision_pct=None,
                accepted_pseudo_CE_logit_gradient_norm_mean=None,accepted_LS001_opposes_hard_pseudo_CE_pct=None)
        r,_=summarize(b)
        x=next(v for v in r['overall'] if v['method']=='SIG1-PL' and v['metric']=='accepted_precision_pct')
        self.assertEqual(x['defined_seeds'],2);self.assertIsNone(x['mean']);self.assertIsNone(x['sample_SD'])

    def test_source_boundaries_counts_and_strata_rejected(self):
        for key,value in [('target_reads',1),('parameters_and_buffers_unchanged',False),('source_role','U_s')]:
            b=fixture();b['rows'][0][key]=value
            with self.assertRaises(ValueError):summarize(b)
        b=fixture();b['rows'][0]['rows'][0]['accepted_correct']-=1
        with self.assertRaisesRegex(ValueError,'counts'):summarize(b)
        b=fixture();b['rows'][0]['rows'].pop()
        with self.assertRaisesRegex(ValueError,'strata'):summarize(b)

    def test_all_twelve_barrier_script_and_runtime(self):
        script=collection_script(dict(run_id=RUN,cpu_only=True,run_root='/unused'))
        compile(script,'read_source_results','exec')
        self.assertIn("any(r['status']!='COMPLETE'",script)
        with self.assertRaises(ValueError):collection_script(dict(run_id=RUN,cpu_only=False))


if __name__=='__main__':unittest.main()
