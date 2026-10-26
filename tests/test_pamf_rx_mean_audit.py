import unittest
from experiments.pamf_u_history.rx_mean_audit import expected_U,check_step,check_history


class RXMeanAuditTests(unittest.TestCase):
    def test_partial_last_batch_and_H_epoch_boundaries(self):
        self.assertEqual(expected_U(221),(56576,0));self.assertEqual(expected_U(222),(56700,0))
        self.assertEqual(expected_U(223),(56956,256));self.assertEqual(expected_U(444),(113400,56700))
        self.assertEqual(expected_U(6660),(1701000,850500))
        with self.assertRaises(ValueError):expected_U(6661)

    def test_segment_counter_reset_preserves_global_exposure(self):
        # First update of own segment50: new U epoch11 uses raw, while native
        # PL counters restart and the method's physical history continues.
        m=dict(RXM_calls=2221,RXM_seen=567256,RXM_L_pairs=142144,RXM_H_base=0,RXM_teacher_U_seen=850756,
            RXM_nonzero_shift_U=567256,T1_L_packets=768,T1_As_KL_weight=0,T1_PL_seen=256,T1_PL_accepted=240,RXM_weight=1.)
        check_step(m,2221,2221,1.)
        for key,bad in [('T1_PL_seen',567256),('RXM_H_base',1),('RXM_nonzero_shift_U',567255),('RXM_weight',.5)]:
            with self.assertRaises(ValueError):check_step(dict(m,**{key:bad}),2221,2221,1.)

    def test_RX_style_counts_cannot_hide_missing_U_coverage(self):
        rxs=(1,3,4,6,8)
        h=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,mean_shift_U_seen=1701000,
            mean_shift_L_pairs=426240,raw_base_U=850500,H_base_U=850500,student_U_packets=3402000,teacher_U_packets=2551500,
            extra_L_teacher_packets=426240,extra_L_student_packets=426240,nonzero_shift_U=1701000,stored_training_statistics=20,
            U_truth_read=False,target_reads=0,model_EMA_updates=0,new_inference_parameters=0,default_head_joint=True,accepted=1700000,
            by_source_RX={str(r):dict(seen=340200,mean_shift=340200,accepted=340000) for r in rxs},
            by_RX_style={str(r)+'->'+str(s):85050 for r in rxs for s in rxs if r!=s},by_replica=[283500]*3,by_scene=[283500]*3)
        check_history(h)
        h['by_RX_style']['1->3']-=1
        with self.assertRaises(ValueError):check_history(h)


if __name__=='__main__':unittest.main()
