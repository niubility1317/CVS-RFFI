from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pamf_u_history.rx_trust_p_results import RUN,METHODS,SEEDS,PREDECESSOR,expected_run,summarize,collection_script
from experiments.pamf_u_history.rx_trust_p_audit import check_step


def bundle():
    from test_pamf_rx_mean_max_results import bundle as original
    b=original();names={'RXMean-MaxU':'RXTrust-P','RXMean-U':'RXMean-MaxU'}
    for row in b['scores']:
        row['method']=names.get(row['method'],row['method']);s=row['score'];s.update(run_id=expected_run(row['method'],row['model_seed']),row_id='F0-'+row['method']+'-s'+str(row['model_seed']))
    for row in b['pairing_checks']:row['method']=names.get(row['method'],row['method']);row['reference']=names.get(row['reference'],row['reference'])
    return b


class PartialResultsTest(unittest.TestCase):
    def test_matched_three_seed_references_and_weighted_population(self):
        r,t=summarize(bundle());self.assertEqual(r['physical_pairings'],18);self.assertEqual(r['logical_models'],3)
        row=next(x for x in r['summary'] if x['method']==METHODS[0] and x['population']=='overall_168000')
        self.assertAlmostEqual(row['raw_clean_mean'],79);self.assertAlmostEqual(row['raw_clean_sample_sd'],2)
        self.assertEqual(expected_run('RXMean-MaxU',SEEDS[0]),PREDECESSOR)
        delta=next(x for x in r['comparisons'] if x['reference']=='RXMean-MaxU' and x['population']=='overall_168000')
        self.assertAlmostEqual(delta['raw_clean_delta_mean_pp'],1.5);self.assertFalse(r['full_goal_complete'])

    def test_missing_seed_wrong_parent_and_pairing_fail(self):
        b=bundle();b['scores'].pop()
        with self.assertRaises(ValueError):summarize(b)
        b=bundle();b['scores'][-1]['score']['run_id']=RUN
        with self.assertRaises(ValueError):summarize(b)
        b=bundle();b['pairing_checks'].pop()
        with self.assertRaises(ValueError):summarize(b)

    def test_queued_pipeline_is_not_a_complete_result(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);(p/'complete.json').write_text(json.dumps(dict(status='RX_TRUST_P_THREE_SEED_COMPLETE',models=3,methods=list(METHODS),model_seeds=list(SEEDS),raw_only=True,truth_last=True)))
            (p/'pipeline_state.json').write_text(json.dumps(dict(status='WAITING_FOR_PREDECESSOR',jobs={})))
            with self.assertRaisesRegex(ValueError,'must finish first'):exec(collection_script(dict(run_id=RUN,project='/project',run_root=str(p))),{})

    def test_step_routes_follow_H_parity_gate_and_tail_batch(self):
        m=dict(RXM_calls=2220,RXM_seen=567000,RXM_L_pairs=142080,RXM_H_base=1,RXM_teacher_U_seen=3118500,RXM_nonzero_shift_U=567000,
            T1_L_packets=768,T1_As_KL_weight=0,T1_PL_seen=567000,T1_PL_accepted=550000,RXM_weight=1.,RXW_candidate_U_packets=2268000,
            RXW_selected_KL=.3,RXW_random_KL=.2,RXW_all_style_KL_mean=.1,RXP_H_singleton_seen=200000,RXP_H_pair_seen=70000,RXP_H_soft_fallback_seen=13500,
            RXP_current_singleton=90,RXP_current_pair=30,RXP_current_soft_fallback=4,RXM_raw_PL_accepted=120)
        check_step(m,2220,1,1.)
        with self.assertRaises(ValueError):check_step(dict(m,RXP_current_pair=31),2220,1,1.)
        with self.assertRaises(ValueError):check_step(dict(m,RXM_raw_PL_accepted=121),2220,1,1.)
        with self.assertRaises(ValueError):check_step(dict(m,RXP_H_soft_fallback_seen=13501),2220,1,1.)


if __name__=='__main__':unittest.main()
