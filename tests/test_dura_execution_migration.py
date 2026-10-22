import copy,tempfile,unittest
from pathlib import Path
from experiments.cric_dura_comparison.io import write,read
from experiments.cric_dura_comparison.execution_migration import promote_pending_G
from experiments.cric_dura_comparison.multiprocess_control import command_for

ROOT=Path(__file__).resolve().parents[1]/'local_artifacts/dura_training_acceleration_20261022/tests'
ROOT.mkdir(parents=True,exist_ok=True)


class PendingOnlyAcceleration(unittest.TestCase):
    def test_only_still_queued_unstarted_G_is_changed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);rt=dict(control_root=str(root/'new'),release_root='/new/release',code_commit='new')
            jobs=[];previous=dict(jobs={});active={}
            for key,kind,status,exists in [('pending','G','QUEUED',False),('racing','G','RUNNING',False),('complete','G','QUEUED',True),('partial','G','QUEUED',True),('F','F','QUEUED',False)]:
                path=root/(key+'.json');output=root/key
                if exists:output.mkdir()
                write(path,dict(stage=kind,code_commit='old',output_root=str(output),model_seed=1,cache={'roles':'L'}))
                job=dict(id=key,kind=kind,config=str(path),completion=str(output/'complete.json'),successful_statuses=['SOURCE_FROZEN'],log_name=key)
                if key=='complete':write(job['completion'],dict(status='SOURCE_FROZEN'))
                jobs.append(job);previous['jobs'][key]=dict(status=status)
                if key=='racing':active[key]=dict(pid=123,gpu=0)
            before=copy.deepcopy(jobs)
            got=promote_pending_G(rt,jobs,previous,active)
            self.assertEqual([j['id'] for j in got],['pending'])
            self.assertEqual(jobs[1:],before[1:]);self.assertEqual(len(jobs),5)
            config=read(jobs[0]['config']);original=read(before[0]['config'])
            self.assertEqual(config['cache'],original['cache']);self.assertEqual(config['model_seed'],1)
            self.assertEqual(config['output_root'],original['output_root'])
            self.assertEqual(jobs[0]['worker_release'],'/new/release')
    def test_worker_release_is_per_job_and_other_commands_are_unchanged(self):
        rt=dict(python='/python',worker_release='/science3e')
        old=dict(kind='F',config='/original.json')
        new=dict(kind='G',config='/new/config.json',worker_release='/acceleration')
        self.assertEqual(command_for(rt,old,2)[1],'/science3e')
        command,cwd,env=command_for(rt,new,2)
        self.assertEqual(command[-1],'/new/config.json');self.assertEqual(cwd,'/acceleration')
        self.assertEqual(env['PYTHONPATH'],'/acceleration');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2')
    def test_target_config_is_rejected_without_creating_model_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);config=root/'bad.json';output=root/'never'
            write(config,dict(stage='G',code_commit='old',output_root=str(output),target_inputs='FORBIDDEN'))
            job=dict(id='bad',kind='G',config=str(config),completion=str(output/'complete.json'),successful_statuses=['SOURCE_FROZEN'],log_name='bad')
            with self.assertRaisesRegex(ValueError,'source-only'):
                promote_pending_G(dict(control_root=str(root/'control'),release_root='/r',code_commit='new'),[job],dict(jobs={'bad':dict(status='QUEUED')}),{})
            self.assertFalse(output.exists())
    def test_runtime_preserves_baseline_and_resolves_jobs_in_new_control_root(self):
        from experiments.cric_dura_comparison.publish_teacher_acceleration import runtime_for,BASE
        old=dict(project='/p',run_id='20261022-phase1-T1-DURA-multiprocess-control-r03',worker_commit=BASE,
            worker_release='/p/original3e',release_root='/p/r03',control_root='/p/control-r03',
            runtime_file='/p/r03/runtime.json',jobs='/p/original-jobs.json',load_balance=dict(max_gpu_workers=26))
        new=runtime_for(old,'new-commit')
        self.assertEqual(new['worker_release'],old['worker_release']);self.assertEqual(new['worker_commit'],BASE)
        self.assertEqual(new['source_jobs'],old['jobs']);self.assertEqual(new['load_balance'],old['load_balance'])
        self.assertTrue(new['jobs'].startswith(new['control_root']+'/'))
        self.assertEqual(new['previous_controller_cwd'],'/p/r03')
        with self.assertRaisesRegex(ValueError,'prior owner'):runtime_for(dict(old,run_id='UNRELATED'),'new')
    def test_resource_handover_retains_slowdown_decisions_and_trial_cohort(self):
        from experiments.cric_dura_comparison.resource_balance import LoadPolicy
        policy=LoadPolicy(dict(boost_gpus=[0,6],max_gpu_workers=26))
        prior=dict(disabled_fourth_slots={'0':'slow'},trials={'6':dict(assessed=False,cohort={'F-job':dict(count=5,phase='L',seconds=10)})})
        policy.restore_handover(prior)
        self.assertEqual(policy.disabled,{0:'slow'});self.assertIn(6,policy.trials)
        self.assertEqual(policy.trials[6]['cohort'],prior['trials']['6']['cohort'])
        self.assertEqual(policy.history,{})


if __name__=='__main__':unittest.main()
