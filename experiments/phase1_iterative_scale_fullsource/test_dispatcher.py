import unittest
from experiments.phase1_iterative_scale_fullsource.dispatcher import registered_processes

class DispatchTest(unittest.TestCase):
 def test_wait_and_launch_states(self):
  self.assertIsNone(registered_processes({'state':'WAITING_CPU_LOAD'}))
  self.assertEqual(registered_processes({'state':'LAUNCHED','pids':{'source':11,'source_audit':12}}),(11,12))
 def test_fails_on_terminal_failure_and_invalid_process_ids(self):
  with self.assertRaises(RuntimeError):registered_processes({'state':'FAILED','error':'failed'})
  for pids in ({'source':11},{'source':0,'source_audit':12},{'source':11,'source_audit':11}):
   with self.assertRaises(ValueError):registered_processes({'state':'LAUNCHED','pids':pids})

if __name__=='__main__':unittest.main()
