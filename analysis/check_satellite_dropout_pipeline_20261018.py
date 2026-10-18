"""Wait semantics tests;does not inspect or mutate real run processes."""
import tempfile,unittest
from pathlib import Path
import cvs_satellite_dropout_pipeline_20261018 as p
import cvs_satellite_dropout_results_20261018 as a

class WaitChecks(unittest.TestCase):
 def test_verified_live_wait_then_ready(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'out';out.mkdir();calls=[];seen=[False,True]
   p.wait_until(td,'SOURCE',out,readiness=lambda paths:seen.pop(0),probe=lambda project:([123],False),sleep=lambda seconds:calls.append(seconds))
   self.assertEqual(calls,[15]);self.assertEqual(a.read(out/'state.json')['live_owned_PIDs'],[123])
 def test_observation_unknown_is_not_terminal(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'out';out.mkdir();seen=[False,True];calls=[];p.wait_until(td,'FINAL',out,readiness=lambda paths:seen.pop(0),probe=lambda project:([],True),sleep=lambda seconds:calls.append(seconds));self.assertEqual(calls,[15]);self.assertTrue(a.read(out/'state.json')['observation_unknown'])
 def test_complete_metadata_does_not_require_live_process(self):
  with tempfile.TemporaryDirectory() as td:
   def forbidden(project):raise AssertionError('Do not probe aftercomplete')
   p.wait_until(td,'FINAL',Path(td),readiness=lambda paths:True,probe=forbidden,sleep=lambda seconds:self.fail('No waiting'))
 def test_absent_owned_handles_is_incomplete_not_restart(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(a.AwaitingArtifacts):p.wait_until(td,'SOURCE',Path(td),readiness=lambda paths:False,probe=lambda project:([],False),sleep=lambda seconds:self.fail('No restart orsleep'))
 def test_known_failure_preserved(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td)/'runs'/a.RUN;base.mkdir(parents=True);a.write(base/'failure.json',dict(status='SYNTHETIC'))
   with self.assertRaisesRegex(RuntimeError,'technical failure preserved'):p.wait_until(td,'SOURCE',Path(td),readiness=lambda paths:False,probe=lambda project:([123],False),sleep=lambda seconds:self.fail('No restart'))
   self.assertEqual(a.read(base/'failure.json')['status'],'SYNTHETIC')
if __name__=='__main__':unittest.main()
