import unittest
import numpy as np
from experiments.pamf_u_history.source_confidence_metrics import probabilities,statistics,grouped
from experiments.pamf_u_history.source_confidence_readout import validate_context,source_folder,METHODS,SEEDS
from experiments.pamf_u_history.publish_source_confidence_readout import required


class SourceConfidenceTests(unittest.TestCase):
    def test_softmax_shift_invariance_and_probability_normalization(self):
        x=np.asarray([[50,2,-40,0,3,-3],[1,0,-1,2,-2,3]],float)
        np.testing.assert_allclose(probabilities(x),probabilities(x+500));np.testing.assert_allclose(probabilities(x).sum(1),1)

    def test_accepted_precision_error_concentration_and_smoothing_direction(self):
        p=np.asarray([[.999,*([.0002]*5)],[.9,*([.02]*5)],[.96,*([.008]*5)]])
        row=statistics(np.log(p),np.asarray([0,1,0]),role='V')
        self.assertEqual(row['accepted'],2);self.assertEqual(row['accepted_precision_pct'],100)
        self.assertEqual(row['rejected_errors'],1);self.assertEqual(row['rejected_share_of_all_errors_pct'],100)
        self.assertEqual(row['accepted_LS001_opposes_hard_pseudo_CE_pct'],50)
        self.assertLess(row['accepted_pseudo_CE_logit_gradient_norm_mean'],row['rejected_pseudo_CE_logit_gradient_norm_mean'])

    def test_empty_accepted_pool_and_illegal_role_do_not_invent_precision(self):
        row=statistics(np.zeros((6,6)),np.arange(6),role='V')
        self.assertEqual(row['accepted'],0);self.assertIsNone(row['accepted_precision_pct']);self.assertIsNone(row['accepted_LS001_opposes_hard_pseudo_CE_pct'])
        with self.assertRaises(ValueError):statistics(np.zeros((6,6)),np.arange(6),role='U_s')
        with self.assertRaises(ValueError):statistics(np.zeros((6,6)),np.full(6,-1),role='V')

    def test_group_population_counts_and_no_silent_metadata_misalignment(self):
        z=np.zeros((30,6));y=np.arange(30)%6;rx=np.tile([1,3,4,6,8],6);day=np.arange(30)%3+1
        rows=grouped(z,y,rx,day);self.assertEqual(rows[0]['count'],30)
        self.assertEqual(sum(r['count'] for r in rows if r['dimension']=='RX'),30)
        with self.assertRaises(ValueError):grouped(z,y,rx[:-1],day)

    def test_context_permissions_and_remote_paths(self):
        c=dict(project='/home/research/project',python='/python',frozen_sources={k:'/'+k for k in ('F','AF','BASE','FOLD')},channel_code='/physics',source_contract='/contract',source_cache='/cache')
        validate_context(c);self.assertTrue(all('\\' not in source_folder(c,m,s) for m in METHODS for s in SEEDS))
        self.assertTrue(all('\\' not in p for p in required(c)))
        with self.assertRaises(ValueError):validate_context(dict(c,tests={}))
        with self.assertRaises(ValueError):validate_context(dict(c,frozen_sources=dict(c['frozen_sources'],G='/G')))


if __name__=='__main__':unittest.main()
