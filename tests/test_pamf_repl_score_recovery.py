import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_repl_completion import recovery,runner

class ScoreEntryTest(unittest.TestCase):
    def test_CPU_score_does_not_initialize_F_or_G(self):
        c=dict(prediction_output='already_frozen',output_root='new_score')
        with patch.object(runner,'activate',side_effect=AssertionError('Model bootstrap forbidden')),patch('experiments.pamf_repl_completion.evaluation.score',return_value='CPU_ONLY') as score:
            self.assertEqual('CPU_ONLY',runner.worker(c,'score'));score.assert_called_once_with(c)

class RecoveryTest(unittest.TestCase):
    def test_exact_dispatch_only_not_smoke_or_worker(self):
        rt=dict(python='/python',runtime_file='/runtime',release_root='/release')
        argv=['/python','-u','-m','experiments.pamf_repl_completion.runner','dispatch','--config','/runtime']
        recovery.validate_owner(dict(cwd='/release',argv=argv),rt)
        for action in ('smoke','worker'):
            wrong=list(argv);wrong[4]=action
            with self.assertRaises(ValueError):recovery.validate_owner(dict(cwd='/release',argv=wrong),rt)
    def test_attach_preserves_same_PID_cwd_argv_GPU(self):
        item=dict(status='RUNNING',pid=123,gpu=4,cwd='/old',argv=['python','worker'],log='/old/log')
        with patch.object(recovery,'process_info',return_value=dict(cwd='/old',argv=['python','worker'],gpu='4')):
            active,children,fail=recovery.attach([dict(id='F')],dict(jobs={'F':item}))
            self.assertEqual(123,children['F'].pid);self.assertEqual(item,active['F']);self.assertIsNone(children['F'].poll());self.assertFalse(fail)
        with patch.object(recovery,'process_info',return_value=dict(cwd='/other',argv=['python','worker'],gpu='4')):
            with self.assertRaises(ValueError):recovery.attach([dict(id='F')],dict(jobs={'F':item}))

    def test_only_two_failed_CPU_outputs_replaced_no_scientific_job_change(self):
        def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'old';new=base/'new';old.mkdir();new.mkdir();jobs=[];states={}
            for i in range(733):
                rid=recovery.FAILED_ROWS[i] if i<2 else 'row'+str(i);kind='score' if i<228 else 'F';key=kind+'__'+rid
                config=put(old/'configs'/(key+'.json'),dict(row_id=rid,output_root=str(old/'outputs'/rid),prediction_output='FROZEN'))
                jobs.append(dict(id=key,kind=kind,config=config,completion=str(old/'outputs'/rid/'complete.json'),successful_statuses=['SCORED_COMPLETE'],dependencies=[]))
                states[key]=dict(status='FAILED',failure=recovery.FINGERPRINT) if i<2 else dict(status='QUEUED')
            put(old/'jobs.json',jobs);put(old/'sources.json',{})
            put(old/'smoke.json',dict(status='PASS',target_reads=0,U_truth_reads=0))
            put(old/'pipeline_state.json',dict(pid=99,code_commit='oldcommit',systemic_stop_new_launches=True,failures={recovery.FINGERPRINT:2},jobs=states))
            path=put(base/'old_runtime.json',dict(run_root=str(old),release_root='/oldrelease',code_commit='oldcommit'))
            rt=dict(run_id='same_science',recover_previous_runtime=path,run_root=str(new),release_root='/newrelease',previous_execution_commit='oldcommit',code_commit='newcommit')
            with patch.object(recovery,'process_info',return_value=None):updated,state=recovery.prepare(rt)
            self.assertEqual(733,len(updated));self.assertEqual(228,sum(j['kind']=='score' for j in updated));self.assertFalse(state['failures'])
            for before,after in zip(jobs,updated):
                if before['id'] in ('score__'+r for r in recovery.FAILED_ROWS):
                    self.assertNotEqual(before['completion'],after['completion']);self.assertFalse(Path(before['completion']).exists())
                else:
                    self.assertEqual(before,{k:v for k,v in after.items() if k!='execution_release'})
                    self.assertEqual('/newrelease' if before['kind']=='score' else '/oldrelease',after['execution_release'])
            self.assertEqual(2,len(json.loads((new/'score_entry_recovery.json').read_text())['replacements']))

if __name__=='__main__':unittest.main()
