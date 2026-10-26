from copy import deepcopy
import unittest
import numpy as np
from experiments.pamf_u_history.fonly_results import metrics
from experiments.pamf_u_history.covariance_three_results import summarize, METHODS, SEEDS, TASKS, VIEWS, expected_run, pairing_keys,locations


def bundle():
    rows=[]
    for method in (*METHODS,'SIG1-PL','SIG1-Sup'):
        for i,seed in enumerate(SEEDS):
            result=[]
            for task,n in [('cross_rx',126000),('cross_both',42000)]:
                # 3:1 populations must yield 75%, not the unweighted 70%.
                rate=(.80 if task=='cross_rx' else .60)+i*.02
                if method in METHODS:rate+=.02
                each=n//6;correct=round(each*rate);cm=np.zeros((6,6),np.int64)
                for c in range(6):cm[c,c]=correct;cm[c,(c+1)%6]=each-correct
                m=metrics(cm)
                for view in VIEWS:
                    result.append(dict(test=task,view=view,readout='raw',dimension='overall',stratum='ALL',count=n,
                        accuracy=m['accuracy']/100,macro_f1=m['macro_f1']/100,confusion=cm.tolist()))
            rows.append(dict(method=method,model_seed=seed,score=dict(status='SCORED',truth_last=True,target_feedback_forbidden=True,
                model_seed=seed,run_id=expected_run(method,seed),row_id='F0-'+method+'-s'+str(seed),results=result)))
    checks=[dict(method=m,model_seed=s,reference=r,task=t,count={'cross_rx':126000,'cross_both':42000}[t],exact_ID_order=True) for m,s,r,t in pairing_keys()]
    return dict(status='THREE_SEED_FROZEN_RESULTS_COLLECTED',pairing_checks=checks,logical_candidate_raw_decisions=14112000,scores=rows,
        truth_reads=0,IQ_reads=0,weight_reads=0)


class ThreeSeedResultsTests(unittest.TestCase):
    def test_remote_locations_keep_posix_paths_on_windows(self):
        rows=locations(dict(project='/home/research/project'))
        self.assertEqual(len(rows),18)
        self.assertTrue(all(r['root'].startswith('/home/research/project/runs/') and '\\' not in r['root'] for r in rows))

    def test_population_weighting_and_paired_seed_standard_deviation(self):
        result,tables=summarize(bundle())
        row=next(v for v in result['summary'] if v['method']=='CovSel-U' and v['population']=='overall_168000')
        self.assertAlmostEqual(row['raw_clean_mean'],79.)
        self.assertAlmostEqual(row['raw_clean_sample_sd'],2.)
        paired=next(v for v in result['comparisons'] if v['method']=='CovSel-U' and v['reference']=='SIG1-PL' and v['population']=='overall_168000')
        self.assertAlmostEqual(paired['raw_clean_delta_mean_pp'],2.)
        self.assertAlmostEqual(paired['raw_clean_delta_sample_sd_pp'],0.)
        self.assertEqual(len(tables['per_seed']),54)
        self.assertFalse(result['full_goal_complete'])

    def test_missing_or_wrong_seed_and_physical_pairing_fail_closed(self):
        data=bundle();data['scores'].pop()
        with self.assertRaises(ValueError):summarize(data)
        data=bundle();data['scores'][1]['score']['model_seed']=SEEDS[0]
        with self.assertRaises(ValueError):summarize(data)
        data=bundle();data['pairing_checks'][0]['exact_ID_order']=False
        with self.assertRaises(ValueError):summarize(data)
        data=bundle();data['scores'][0]['score']['results'][0]['count']-=1
        with self.assertRaises(ValueError):summarize(data)


if __name__=='__main__':unittest.main()
