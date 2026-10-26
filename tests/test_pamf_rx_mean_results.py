from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.pamf_u_history.fonly_results import metrics,VIEWS
from experiments.pamf_u_history.rx_mean_results import METHODS,REFERENCES,SEEDS,TASKS,COUNTS,RUN,BASELINE,locations,collection_script,summarize


def bundle():
    rows=[]
    for method in (*METHODS,*REFERENCES):
        for i,seed in enumerate(SEEDS):
            cells=[]
            for task,n in COUNTS.items():
                rate=(.8 if task=='cross_rx' else .6)+i*.02+(0 if method in REFERENCES else .02)
                each=n//6;correct=round(each*rate);cm=np.zeros((6,6),np.int64)
                for k in range(6):cm[k,k]=correct;cm[k,(k+1)%6]=each-correct
                m=metrics(cm)
                for view in VIEWS:cells.append(dict(test=task,view=view,readout='raw',dimension='overall',stratum='ALL',count=n,accuracy=m['accuracy']/100,macro_f1=m['macro_f1']/100,confusion=cm.tolist()))
            rows.append(dict(method=method,model_seed=seed,score=dict(status='SCORED',truth_last=True,target_feedback_forbidden=True,model_seed=seed,
                run_id=BASELINE if method in REFERENCES else RUN,row_id='F0-'+method+'-s'+str(seed),results=cells)))
    checks=[dict(method=m,model_seed=s,reference=r,task=t,count=COUNTS[t],exact_ID_order=True) for m,s,r,t in itertools.product(METHODS,SEEDS,REFERENCES,TASKS)]
    return dict(status='THREE_SEED_FROZEN_RESULTS_COLLECTED',scores=rows,pairing_checks=checks,logical_candidate_raw_decisions=7056000,truth_reads=0,IQ_reads=0,weight_reads=0)


class RXMeanResultsTests(unittest.TestCase):
    def test_weighted_populations_and_same_seed_delta(self):
        result,tables=summarize(bundle())
        self.assertEqual(result['logical_models'],6);self.assertEqual(result['physical_pairings'],24);self.assertFalse(result['full_goal_complete'])
        row=next(v for v in result['summary'] if v['method']=='RXMean-U' and v['population']=='overall_168000')
        self.assertAlmostEqual(row['raw_clean_mean'],79.);self.assertAlmostEqual(row['raw_clean_sample_sd'],2.)
        paired=next(v for v in result['comparisons'] if v['method']=='RXMean-U' and v['reference']=='SIG1-PL' and v['population']=='overall_168000')
        self.assertAlmostEqual(paired['raw_clean_delta_mean_pp'],2.);self.assertAlmostEqual(paired['raw_clean_delta_sample_sd_pp'],0.)
        self.assertEqual(len(tables['per_seed']),36)

    def test_incomplete_seed_wrong_run_and_unpaired_ids_rejected(self):
        for mutation in ('missing_seed','wrong_run','IDs','count'):
            a=bundle()
            if mutation=='missing_seed':a['scores'].pop()
            elif mutation=='wrong_run':a['scores'][0]['score']['run_id']=BASELINE
            elif mutation=='IDs':a['pairing_checks'][0]['exact_ID_order']=False
            else:a['logical_candidate_raw_decisions']-=168000
            with self.assertRaises(ValueError):summarize(a)

    def test_collector_refuses_incomplete_run_before_opening_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'complete.json').write_text(json.dumps(dict(status='RX_MEAN_THREE_SEED_COMPLETE',models=6,methods=list(METHODS),model_seeds=list(SEEDS),raw_only=True,truth_last=True)))
            (root/'pipeline_state.json').write_text(json.dumps(dict(status='RUNNING',jobs={})))
            script=collection_script(dict(run_id=RUN,project='/project',run_root=str(root)))
            with self.assertRaisesRegex(ValueError,'must finish first'):exec(script,{})

    def test_locations_preserve_remote_posix_paths(self):
        rows=locations(dict(project='/home/research/project'));self.assertEqual(len(rows),12)
        self.assertTrue(all('\\' not in r['root'] for r in rows))
        with self.assertRaises(ValueError):collection_script(dict(run_id='other',project='/project'))


if __name__=='__main__':unittest.main()
