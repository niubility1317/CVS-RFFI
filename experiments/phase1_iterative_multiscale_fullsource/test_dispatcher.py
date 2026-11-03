import unittest
from experiments.phase1_iterative_multiscale_fullsource.dispatcher import registered_processes

class DispatchTest(unittest.TestCase):
 def test_wait_and_launch_states(self):
  self.assertIsNone(registered_processes({'state':'WAITING_CPU_LOAD'}))
  self.assertEqual(registered_processes({'state':'LAUNCHED','pids':{'source':11,'source_audit':12}}),(11,12))
 def test_original_comparator_requires_fixed_independently_audited_source(self):
  from experiments.phase1_iterative_multiscale_fullsource.dispatcher import validate_original,ORIGINAL
  source={'state':'SOURCE_SCORE_COMPLETE'}
  meta={'status':'SOURCE_PREDICTIONS_FIXED','count':11761,'pilot_replayed':672,'source_models':[ORIGINAL],'TX_truth_reads':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'model_updates':0}
  audit={'status':'PASS','count':11761,'comparisons':96,'fields':480}
  validate_original(source,meta,audit)
  for bad in (dict(meta,status='RUNNING'),dict(meta,source_models=[]),dict(meta,U_s_truth_reads=1)):
   with self.assertRaises(ValueError):validate_original(source,bad,audit)
  with self.assertRaises(ValueError):validate_original(source,meta,dict(audit,status='FAILED'))
 def test_fails_on_terminal_failure_and_invalid_process_ids(self):
  with self.assertRaises(RuntimeError):registered_processes({'state':'FAILED','error':'failed'})
  for pids in ({'source':11},{'source':0,'source_audit':12},{'source':11,'source_audit':11}):
   with self.assertRaises(ValueError):registered_processes({'state':'LAUNCHED','pids':pids})

if __name__=='__main__':unittest.main()
