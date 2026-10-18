"""Wait semantics tests;does not inspect or mutate real run processes."""
import ast,json,sys,tempfile,unittest
from unittest.mock import patch
import cvs_satellite_dropout_publish_analysis_20261018 as publisher
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

class OutputChecks(unittest.TestCase):
 def test_registered_attempt_names_are_accepted(self):
  with tempfile.TemporaryDirectory() as td:
   project=Path(td);base=project/'runs'/a.RUN
   for name in ('analysis_auto_r01','analysis_auto_r02','analysis_auto_r03'):
    with self.subTest(name=name):self.assertEqual(p.validate_output(project,base/name),base/name)
 def test_other_run_or_parent_is_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   project=Path(td)
   for parent in (project,project/'runs'/'other-run',project/'outside'/a.RUN):
    with self.subTest(parent=parent),self.assertRaisesRegex(ValueError,'Only own immutable analysis root'):p.validate_output(project,parent/'analysis_auto_r03')
 def test_non_attempt_names_are_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   project=Path(td);base=project/'runs'/a.RUN
   for name in ('analysis_auto_r3','analysis_auto_r003','analysis_auto_r03-launch','analysis_auto_r03-exports','arbitrary','analysis_auto_rNN'):
    with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Only own immutable analysis root'):p.validate_output(project,base/name)
 def test_existing_attempt_is_preserved(self):
  with tempfile.TemporaryDirectory() as td:
   project=Path(td);out=project/'runs'/a.RUN/'analysis_auto_r03';out.mkdir(parents=True);marker=out/'existing.txt';marker.write_text('preserve',encoding='utf8')
   with self.assertRaisesRegex(FileExistsError,'no relaunch'):p.validate_output(project,out)
   self.assertEqual(marker.read_text(encoding='utf8'),'preserve')
 def test_publisher_emitted_output_passes_pipeline_guard(self):
  with tempfile.TemporaryDirectory() as td:
   project=Path(td)/'synthetic-project';artifacts=Path(td)/'artifacts';runtime=Path(td)/'runtime.json';rt=dict(project=str(project),python='/SYNTHETIC/python',ssh_config='/SYNTHETIC/ssh',user='ordinary',host='SYNTHETIC');runtime.write_text(json.dumps(rt),encoding='utf8');seen=[]
   def remote(script,unused):
    tree=ast.parse(script);configs=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='c' for x in n.targets)];self.assertEqual(len(configs),1);config=configs[0];seen.append(config);out=p.validate_output(project,Path(config['output']));self.assertEqual(out,project/'runs'/a.RUN/'analysis_auto_r03');self.assertEqual(Path(config['launch']),out.with_name(out.name+'-launch'));self.assertEqual(Path(config['release']).name,'cvs_F_satellite_dropout_analysis_20261018_r03');return dict(status='SUBMITTED',pid=123,NN=0)
   argv=['publisher','--stage','launch','--runtime',str(runtime),'--source-root',str(project),'--artifact-root',str(artifacts),'--commit','SYNTHETIC_COMMIT']
   with patch.object(sys,'argv',argv),patch.object(publisher,'remote',side_effect=remote):publisher.main()
   self.assertEqual(len(seen),1);self.assertEqual(json.loads((artifacts/'launch.json').read_text(encoding='utf8'))['status'],'SUBMITTED');self.assertFalse(Path(seen[0]['output']).exists())

if __name__=='__main__':unittest.main()
