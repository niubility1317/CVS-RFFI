import unittest
import numpy as np
from experiments.pamf_u_history.rx_mean_reference_diagnostics import paired_statistics,diagnose,summarize,RUN,SCENES


class SourceReferenceTest(unittest.TestCase):
    def data(self):
        raw=np.array([[9,0,0,0,0,0],[1,0,1,0,0,0],[0,0,9,0,0,0]],float)
        H=np.array([[0,9,0,0,0,0],[0,9,0,0,0,0],[9,0,0,0,0,0]],float)
        return raw,H,np.array([0,1,0])

    def test_same_packet_wrong_confident_H_reference_is_counted(self):
        raw,H,y=self.data();r=paired_statistics(raw,H,y,role='V')
        self.assertEqual(r['raw_correct_H_wrong'],1);self.assertEqual(r['raw_wrong_H_correct'],2)
        self.assertEqual(r['H_confident_wrong'],1);self.assertEqual(r['raw_H_disagreements'],3)
        self.assertAlmostEqual(r['H_confident_precision_pct'],200/3);self.assertGreater(r['raw_H_KL_mean'],0)

    def test_raw_accepted_rejected_partition_and_view_identity(self):
        raw,H,y=self.data();z=np.stack([raw,H,H,H]);r=diagnose(z,y,np.array([1,1,3]),np.array([1,2,1]),['raw',*SCENES])
        selected=[v for v in r if v['dimension']=='overall' and v['scene']==SCENES[0]]
        counts={v['scope']:v['count'] for v in selected};self.assertEqual(counts,dict(all=3,raw_accepted=2,raw_rejected=1))
        with self.assertRaises(ValueError):diagnose(z,y,np.array([1,1,3]),np.array([1,2,1]),['raw',*reversed(SCENES)])

    def test_source_role_and_alignment_are_required(self):
        raw,H,y=self.data()
        with self.assertRaises(ValueError):paired_statistics(raw,H,y,role='U_s')
        with self.assertRaises(ValueError):paired_statistics(raw,H,y-10,role='V')
        with self.assertRaises(ValueError):paired_statistics(raw,H[:2],y,role='V')

    def test_incomplete_seed_matrix_cannot_be_summarized(self):
        b=dict(run_id=RUN,status='PAIRED_SOURCE_REFERENCE_DIAGNOSTICS_COLLECTED',new_F_forwards=0,new_training_updates=0,U_TX_truth_reads=0,target_reads=0,G_forwards=0,metadata_checks=[],rows=[])
        with self.assertRaisesRegex(ValueError,'all three'):summarize(b)


if __name__=='__main__':unittest.main()
