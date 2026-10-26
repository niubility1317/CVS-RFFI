from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.pamf_u_history.rx_mean_max_results import RUN,METHODS,REFERENCES,SEEDS,expected_run,locations,collection_script,summarize
from experiments.pamf_u_history.fonly_results import metrics,VIEWS
from experiments.pamf_u_history.covariance_three_results import TASKS,COUNTS
from experiments.pamf_u_history.rx_mean_max_audit import check_step,check_history


def bundle():
    scores=[]
    for method in (*METHODS,*REFERENCES):
        for i,seed in enumerate(SEEDS):
            cells=[]
            for task,n in COUNTS.items():
                bonus=.02 if method in METHODS else .005 if method=='RXMean-U' else 0.
                rate=(.8 if task=='cross_rx' else .6)+i*.02+bonus;each=n//6;correct=round(each*rate);cm=np.zeros((6,6),np.int64)
                for k in range(6):cm[k,k]=correct;cm[k,(k+1)%6]=each-correct
                m=metrics(cm)
                for view in VIEWS:cells.append(dict(test=task,view=view,readout='raw',dimension='overall',stratum='ALL',count=n,accuracy=m['accuracy']/100,macro_f1=m['macro_f1']/100,confusion=cm.tolist()))
            scores.append(dict(method=method,model_seed=seed,score=dict(status='SCORED',truth_last=True,target_feedback_forbidden=True,model_seed=seed,
                run_id=expected_run(method,seed),row_id='F0-'+method+'-s'+str(seed),results=cells)))
    checks=[dict(method=m,model_seed=s,reference=r,task=t,count=COUNTS[t],exact_ID_order=True) for m,s,r,t in itertools.product(METHODS,SEEDS,REFERENCES,TASKS)]
    return dict(status='THREE_SEED_FROZEN_RESULTS_COLLECTED',scores=scores,pairing_checks=checks,logical_candidate_raw_decisions=3528000,truth_reads=0,IQ_reads=0,weight_reads=0)


class RXMeanMaxResultsTests(unittest.TestCase):
    def test_three_references_and_same_seed_weighted_comparison(self):
        result,tables=summarize(bundle())
        self.assertEqual(result['logical_models'],3);self.assertEqual(result['physical_pairings'],18);self.assertFalse(result['full_goal_complete'])
        summary=next(r for r in result['summary'] if r['method']==METHODS[0] and r['population']=='overall_168000')
        self.assertAlmostEqual(summary['raw_clean_mean'],79.);self.assertAlmostEqual(summary['raw_clean_sample_sd'],2.)
        paired=next(r for r in result['comparisons'] if r['reference']=='RXMean-U' and r['population']=='overall_168000')
        self.assertAlmostEqual(paired['raw_clean_delta_mean_pp'],1.5);self.assertAlmostEqual(paired['raw_clean_delta_sample_sd_pp'],0.)
        self.assertEqual(len(tables['paired_summary']),9)

    def test_wrong_reused_run_missing_seed_and_pairing_rejected(self):
        a=bundle();a['scores'].pop()
        with self.assertRaises(ValueError):summarize(a)
        a=bundle();a['scores'][-1]['score']['run_id']=RUN
        with self.assertRaises(ValueError):summarize(a)
        a=bundle();a['pairing_checks'].pop()
        with self.assertRaises(ValueError):summarize(a)

    def test_waiting_matrix_cannot_be_summarized_as_complete(self):
        self.assertEqual(len(locations(dict(project='/home/research/project'))),12)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'complete.json').write_text(json.dumps(dict(status='RX_MEAN_MAX_THREE_SEED_COMPLETE',models=3,methods=list(METHODS),model_seeds=list(SEEDS),raw_only=True,truth_last=True)))
            (root/'pipeline_state.json').write_text(json.dumps(dict(status='WAITING_FOR_PREDECESSOR',jobs={})))
            script=collection_script(dict(run_id=RUN,project='/project',run_root=str(root)))
            with self.assertRaisesRegex(ValueError,'must finish first'):exec(script,{})

    def test_candidate_eval_cost_and_maximum_are_checked_per_step(self):
        m=dict(RXM_calls=2221,RXM_seen=567256,RXM_L_pairs=142144,RXM_H_base=0,RXM_teacher_U_seen=3119780,
            RXM_nonzero_shift_U=567256,T1_L_packets=768,T1_As_KL_weight=0,T1_PL_seen=256,T1_PL_accepted=240,RXM_weight=1.,
            RXW_candidate_U_packets=2269024,RXW_selected_KL=.01,RXW_random_KL=.003,RXW_all_style_KL_mean=.005)
        check_step(m,2221,2221,1.)
        for key,bad in [('RXW_candidate_U_packets',567256),('RXM_teacher_U_seen',850756),('RXW_selected_KL',.001)]:
            with self.assertRaises(ValueError):check_step(dict(m,**{key:bad}),2221,2221,1.)

    def test_selected_style_subset_is_allowed_without_losing_any_receiver(self):
        rxs=(1,3,4,6,8);styles={str(r)+'->'+str(rxs[(i+1)%5]):340200 for i,r in enumerate(rxs)}
        h=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,mean_shift_U_seen=1701000,
            mean_shift_L_pairs=426240,raw_base_U=850500,H_base_U=850500,student_U_packets=3402000,teacher_U_packets=9355500,
            extra_L_teacher_packets=426240,extra_L_student_packets=426240,nonzero_shift_U=1701000,stored_training_statistics=20,
            candidate_U_packets=6804000,candidate_variants=4,selection_uses_U_TX=False,U_truth_read=False,target_reads=0,model_EMA_updates=0,
            new_inference_parameters=0,default_head_joint=True,selection_policy='per_packet_max_teacher_to_candidate_KL_over_other4_sourceRX',
            by_source_RX={str(r):dict(seen=340200,mean_shift=340200,accepted=340000) for r in rxs},accepted=1700000,by_RX_style=styles,by_replica=[283500]*3,by_scene=[283500]*3)
        check_history(h)
        bad=deepcopy(h);bad['by_RX_style']['1->1']=bad['by_RX_style'].pop('1->3')
        with self.assertRaises(ValueError):check_history(bad)


if __name__=='__main__':unittest.main()
