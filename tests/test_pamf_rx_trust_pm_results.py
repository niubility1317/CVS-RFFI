import json
from pathlib import Path
import tempfile
import unittest
from experiments.pamf_u_history.rx_trust_pm_results import RUN,METHODS,SEEDS,PREDECESSOR,expected_run,summarize,collection_script


def bundle():
    from test_pamf_rx_trust_p_results import bundle as original
    b=original();names={'RXTrust-P':'RXTrust-PM','RXMean-MaxU':'RXTrust-P'}
    for row in b['scores']:
        row['method']=names.get(row['method'],row['method']);row['score'].update(run_id=expected_run(row['method'],row['model_seed']),row_id='F0-'+row['method']+'-s'+str(row['model_seed']))
    for row in b['pairing_checks']:row['method']=names.get(row['method'],row['method']);row['reference']=names.get(row['reference'],row['reference'])
    return b


class ModeResultsTest(unittest.TestCase):
    def test_paired_three_seed_references_and_weighted_accuracy(self):
        r,_=summarize(bundle());self.assertEqual(r['logical_models'],3);self.assertEqual(r['physical_pairings'],18)
        self.assertEqual(expected_run('RXTrust-P',SEEDS[0]),PREDECESSOR)
        s=next(v for v in r['summary'] if v['method']==METHODS[0] and v['population']=='overall_168000');self.assertAlmostEqual(s['raw_clean_mean'],79);self.assertAlmostEqual(s['raw_clean_sample_sd'],2)
        d=next(v for v in r['comparisons'] if v['reference']=='RXTrust-P' and v['population']=='overall_168000');self.assertAlmostEqual(d['raw_clean_delta_mean_pp'],1.5);self.assertFalse(r['full_goal_complete'])

    def test_missing_seed_and_wrong_reused_run_are_rejected(self):
        b=bundle();b['scores'].pop()
        with self.assertRaises(ValueError):summarize(b)
        b=bundle();b['scores'][-1]['score']['run_id']=RUN
        with self.assertRaises(ValueError):summarize(b)

    def test_waiting_jobs_never_become_formal_results(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);(p/'complete.json').write_text(json.dumps(dict(status='RX_TRUST_PM_THREE_SEED_COMPLETE',models=3,methods=list(METHODS),model_seeds=list(SEEDS),raw_only=True,truth_last=True)))
            (p/'pipeline_state.json').write_text(json.dumps(dict(status='WAITING_FOR_PREDECESSOR',jobs={})))
            with self.assertRaisesRegex(ValueError,'must finish first'):exec(collection_script(dict(run_id=RUN,project='/project',run_root=str(p))),{})


if __name__=='__main__':unittest.main()
