import json,tempfile,unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_channel_pl import receiver_balance_confirmation as c
from tests.test_pamf_clean_h_source import config as original_config
from tests.test_pamf_no_pair_source import context as previous_context
from tests.test_pamf_receiver_balance_source import context as source_context


def fixture(project='/project'):
    s=c.source;n=s.reference;old=json.loads(json.dumps(original_config(n.BASE_METHOD)).replace('/project',project))
    with patch.object(n,'read',return_value=old):previous=n.template(previous_context(project),'old')
    with patch.object(s,'read',return_value=previous),patch.object(n,'read',return_value=old):source=s.template(source_context(project),'source_commit',s.METHODS[0])
    return old,previous,source


def runtime(project='/project'):
    release=project+'/releases/'+c.RELEASE
    return dict(run_id=c.RUN,project=project,python='python',release_root=release,run_root=project+'/runs/'+c.RUN,context=release+'/context.json',runtime_file=release+'/runtime.json',
        code_commit='commit',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},allowed_gpus=list(range(8)),launch_owner=c.OWNER)


class ReceiverBalanceConfirmationTests(unittest.TestCase):
    def test_original_E20_own_seeds_and_exact_method_no_second_training_seed01(self):
        old,previous,source=fixture();winner=source['paper_method'];path='/project/runs/'+c.source.RUN+'/complete.json'
        with patch.object(c,'selection',return_value=(winner,path)),patch.object(c,'read',return_value=source),patch.object(c.source,'read',return_value=previous),patch.object(c.source.reference,'read',return_value=old):
            for seed in c.SEEDS[1:]:
                cfg=c.template('/project','commit',seed);c.validate_config(cfg)
                self.assertEqual(cfg['shared_SIG1_E20'],c.prefix('/project',seed));self.assertEqual(cfg['receiver_balance'],source['receiver_balance'])
                for changes in [dict(shared_SIG1_E20=c.prefix('/project',c.SEEDS[0])),dict(epochs=40),dict(plain_PL_weight=7),dict(source_G={})]:
                    with self.assertRaises(ValueError):c.validate_config(dict(cfg,**changes))
                for offset,n in [(20,10),(30,20)]:
                    part=dict(cfg,segment_epochs=n,epoch_offset=offset,step_offset=offset*222,parent_source=cfg['shared_SIG1_E20'] if offset==20 else cfg['output_root']+'/segment30',output_root=cfg['output_root']+'/segment'+str(offset+n))
                    c.validate_config(part);part['parent_source']='foreign'
                    with self.assertRaises(ValueError):c.validate_config(part)
            with self.assertRaises(ValueError):c.template('/project','commit',c.SEEDS[0])

    def test_exactly_nine_jobs_and_all_control_predictions_reused(self):
        with tempfile.TemporaryDirectory() as td:
            project=td.replace('\\','/');old,previous,source=fixture(project);winner=source['paper_method'];rt=runtime(project);Path(rt['run_root']).mkdir(parents=True)
            ctx=dict(project=project,python='python',frozen_sources=rt['frozen_sources'],classes=list('abcdef'),tests={t:dict(views_root=t,truth='truth_'+t,kind=t) for t in c.evaluation.TASK_COUNTS})
            def reader(p):return ctx if str(p)==rt['context'] else source
            with patch.object(c,'selection',return_value=(winner,project+'/runs/'+c.source.RUN+'/complete.json')),patch.object(c,'read',reader),patch.object(c.source,'read',return_value=previous),patch.object(c.source.reference,'read',return_value=old):jobs=c.prepare(rt)
            self.assertEqual(Counter(j['kind'] for j in jobs),dict(train=2,predict=3,score=3,smoke=1))
            for job in jobs:
                cfg=json.loads(Path(job['config']).read_text())
                if job['kind'] in ('predict','score'):
                    self.assertEqual(cfg['method'],winner);self.assertEqual(set(cfg['tests']),set(c.evaluation.TASK_COUNTS))
                if job['kind']=='predict':
                    self.assertNotIn('truth',str(cfg));self.assertEqual(cfg['readouts'],['raw'])
                    if cfg['model_seed']==c.SEEDS[0]:self.assertEqual(cfg['source_run_id'],c.source.RUN)
                if job['kind']=='score':self.assertTrue(c.evaluation.shared.PREDICT_ONLY.isdisjoint(cfg))
            refs=json.loads((Path(rt['run_root'])/'reused_reference_scores.json').read_text());self.assertEqual(set(refs),{'SIG1-Sup','SIG1-PL'})
            self.assertEqual(sum(len(v) for v in refs.values()),6)

    def test_source_gate_denies_before_new_root_or_target_construction(self):
        with tempfile.TemporaryDirectory() as td:
            rt=runtime(td.replace('\\','/'));Path(td,'runs').mkdir()
            with patch.object(c,'selection',side_effect=ValueError('no source pass')),patch.object(c,'prepare') as prepare:
                with self.assertRaises(ValueError):c.dispatch(rt)
            prepare.assert_not_called();self.assertFalse(Path(rt['run_root']).exists())

    def test_scorer_worker_never_activates_F_and_train_uses_same_RB_factory(self):
        with patch.object(c.evaluation,'score',return_value='CPU') as score,patch.object(c.source.base,'activate') as activate:
            self.assertEqual(c.worker({'score':'only'},'score'),'CPU')
        score.assert_called_once();activate.assert_not_called()
        cfg=dict(frozen_sources={})
        with patch.object(c.source.base,'activate'),patch.object(c.source.base,'train') as train:c.worker(cfg,'train')
        train.assert_called_once_with(cfg,c.validate_config,c.ReceiverBalancedObjective)

    def test_global_capacity_preserves_queued_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            project=td.replace('\\','/');Path(td,'runs').mkdir();rt=runtime(project)
            jobs=[dict(id='smoke',kind='smoke',dependencies=[],completion='absent',successful_statuses=['PASS'],gpu_pool=list(range(8)))]
            class Snapshot(Exception):pass
            with patch.object(c,'selection',return_value=('winner','source')),patch.object(c,'prepare',return_value=jobs),patch('experiments.pamf_repl_completion.runner.capacity',return_value=({101,102},20000)),patch('experiments.pamf_repl_completion.runner.processes',return_value={}),patch('experiments.pamf_peer_supervised.runner.launch') as launch,patch.object(c.time,'sleep',side_effect=Snapshot):
                with self.assertRaises(Snapshot):c.dispatch(rt)
            launch.assert_not_called();state=json.loads((Path(rt['run_root'])/'pipeline_state.json').read_text());self.assertEqual(state['counts'],{'QUEUED':1})

    def run_summary(self,td,six_delta=None,broken=False):
        winner=c.source.METHODS[0];mapping={};jobs=[]
        def scored(method,seed,tasks,selected):
            rows=[]
            for task in tasks:
                n=c.evaluation.TASK_COUNTS[task]
                for view in c.evaluation.VIEWS:
                    pct=({'cross_rx':90.,'cross_both':70.,'cross_date':10.}[task] if selected else 80.)
                    if six_delta is not None:pct=(85. if view=='clean' else 80.+six_delta) if selected else (84. if view=='clean' else 80.)
                    correct=round(n*pct/100);cm=[[0]*6 for _ in range(6)];cm[0][0]=correct;cm[0][1]=n-correct
                    rows.append(dict(test=task,readout='raw',dimension='overall',view=view,count=n,confusion=cm))
            if broken:rows[-1]=deepcopy(rows[-2])
            return dict(status='SCORED',row_id='F0-'+method+'-s'+str(seed),model_seed=seed,truth_last=True,target_feedback_forbidden=True,results=rows)
        for seed in c.SEEDS:
            name='selected_'+str(seed);out=Path(td)/str(seed);cfg=dict(model_seed=seed,output_root=str(out));mapping[name]=cfg;jobs.append(dict(kind='score',config=name))
            mapping[str(out/'scores.json')]=scored(winner,seed,tuple(c.evaluation.TASK_COUNTS),True)
        for method in ('SIG1-Sup','SIG1-PL'):
            for seed in c.SEEDS:
                paths=c.reference_score_paths(td,method,seed);mapping[paths['primary']]=scored(method,seed,('cross_rx','cross_both'),False);mapping[paths['cross_date']]=scored(method,seed,('cross_date',),False)
        with patch.object(c,'selection',return_value=(winner,'source')),patch.object(c,'read',side_effect=lambda p:mapping[str(p)]):c.analyze(dict(project=td,run_root=td),jobs)
        return json.loads((Path(td)/'analysis/complete.json').read_text())

    def test_primary_counts_weighted168000_cross_date_separate_and_controls_not_repredicted(self):
        with tempfile.TemporaryDirectory() as td:end=self.run_summary(td)
        selected=[r for r in end['results'] if r['method']==c.source.METHODS[0]]
        self.assertEqual([r['mean'] for r in selected if r['population']=='overall_168000'],[85.,85.])
        self.assertEqual([r['mean'] for r in selected if r['population']=='cross_date'],[10.,10.])
        self.assertFalse(end['target_goals_pass']);self.assertEqual(end['new_reference_predictions'],0);self.assertEqual(end['new_frozen_class_decisions'],4158000)

    def test_six_scene_gain_strictly_more_than1p5_and_clean_at_least1(self):
        with tempfile.TemporaryDirectory() as td:end=self.run_summary(td,six_delta=1.5)
        self.assertFalse(end['matched_baseline_goal_pass'])
        with tempfile.TemporaryDirectory() as td:end=self.run_summary(td,six_delta=1.51)
        self.assertTrue(end['matched_baseline_goal_pass']);self.assertFalse(end['target_floors_pass'])

    def test_duplicate_missing_view_scores_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):self.run_summary(td,broken=True)


if __name__=='__main__':unittest.main()
