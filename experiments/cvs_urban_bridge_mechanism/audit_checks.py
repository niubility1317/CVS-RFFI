"""Focused reporting-contract negatives; no model or data imports."""
import tempfile,unittest
from experiments.cvs_urban_semantic50._audit_worker import main_forward_packets
from ._audit_worker import audit_bridge


class Checks(unittest.TestCase):
    def counts(self):
        return dict(clean_L_packets=1420800,teacher_L_clean_packets=1420800,native_U_packets=963900)

    def test_cost_profiles_count_real_HQ_student_and_teacher_and_preserve_old_profiles(self):
        counts=self.counts();self.assertEqual(main_forward_packets(counts,False),4769400)
        counts['satellite_L_packets']=1420800;self.assertEqual(main_forward_packets(counts,True),6190200)
        counts.update(HQ_L_packets=1420800,teacher_HQ_packets=1420800)
        self.assertEqual(main_forward_packets(counts,True,HQ=True),9031800)
        with self.assertRaises(ValueError):main_forward_packets(counts,True)
        del counts['teacher_HQ_packets']
        with self.assertRaises(ValueError):main_forward_packets(counts,True,HQ=True)

    def test_incomplete_matrix_never_reads_partial_logs_or_target_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            def no_access(row):raise AssertionError('Partial row accessed')
            value=audit_bridge(dict(base=directory,rows=[dict(source='forbidden')]),no_access)
        self.assertEqual(value['status'],'WAITING_FOR_MATRIX');self.assertEqual(value['new_updates_parsed'],0)
        self.assertFalse(value['reads_IQ_or_checkpoints']);self.assertFalse(value['reads_target_scores'])


if __name__=='__main__':unittest.main()
