import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_channel_pl import clean_h_confirmation as c
from tests.test_pamf_clean_h_source import config as source_config


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.old=source_config('REPL_H2_LU_Pair');self.method=self.old['paper_method']
        self.select=patch.object(c,'selection',return_value=(self.method,'/project/runs/source/complete.json'));self.select.start();self.addCleanup(self.select.stop)

    def test_only_missing_seeds_exact_selected_recipe_own_parent_and_budget(self):
        with patch.object(c,'read',return_value=self.old):
            for seed in c.SEEDS[1:]:
                cfg=c.template('/project','commit',seed);c.validate_config(cfg)
                self.assertEqual(cfg['shared_SIG1_E20'],c.prefix('/project',seed));self.assertEqual(cfg['clean_H'],self.old['clean_H'])
                for k,v in [('shared_SIG1_E20',c.prefix('/project',c.SEEDS[0])),('epochs',40),('plain_PL_weight',7),('clean_H',dict(dual_L=False))]:
                    bad=deepcopy(cfg);bad[k]=v
                    with self.assertRaises(ValueError):c.validate_config(bad)
                for offset,n in ((20,10),(30,20)):
                    segment=dict(cfg,segment_epochs=n,epoch_offset=offset,step_offset=222*offset,parent_source=cfg['shared_SIG1_E20'] if offset==20 else cfg['output_root']+'/segment30',output_root=cfg['output_root']+'/segment'+str(offset+n))
                    c.validate_config(segment);segment['parent_source']=c.prefix('/project',c.SEEDS[0])
                    with self.assertRaises(ValueError):c.validate_config(segment)
            with self.assertRaises(ValueError):c.template('/project','commit',c.SEEDS[0])

    def test_complete_matrix_reuses_seed01_and_only_missing_reference_cross_date(self):
        with tempfile.TemporaryDirectory() as td:
            td=td.replace('\\','/')
            self.old=json.loads(json.dumps(source_config(self.method)).replace('/project',td))
            root=Path(td)/'runs'/c.RUN;root.mkdir(parents=True)
            release=td+'/releases/'+c.RELEASE
            rt=dict(run_id=c.RUN,project=td,python='python',release_root=release,run_root=td+'/runs/'+c.RUN,context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='commit',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},allowed_gpus=list(range(8)),launch_owner='codex-pamf-clean-h-confirm-20261027')
            ctx=dict(project=td,python='python',frozen_sources=rt['frozen_sources'],classes=list('abcdef'),tests={t:dict(views_root=t,truth='truth_'+t,kind=t) for t in c.evaluation.TASK_COUNTS})
            def reader(path):
                if str(path)==rt['context']:return ctx
                if str(path).endswith('complete.json'):return dict(config=dict(source_contract='roles'))
                return self.old
            with patch.object(c,'read',reader),patch.object(c,'selection',return_value=(self.method,td+'/runs/'+c.source.RUN+'/complete.json')):jobs=c.prepare(rt)
            self.assertEqual(Counter(j['kind'] for j in jobs),dict(train=2,predict=9,score=9,smoke=1))
            for job in jobs:
                cfg=json.loads(Path(job['config']).read_text())
                if job['kind']=='predict':
                    self.assertEqual(cfg['readouts'],['raw']);self.assertNotIn('truth',str(cfg))
                    if cfg['purpose']=='baseline_cross_date':self.assertEqual(set(cfg['tests']),{'cross_date'})
                    elif cfg['model_seed']==c.SEEDS[0]:self.assertIn(c.source.RUN,cfg['source_output'])
                if job['kind']=='score':self.assertNotIn('source_output',cfg);self.assertNotIn('frozen_sources',cfg)

    def test_truth_worker_never_activates_or_loads_F(self):
        with patch.object(c.evaluation,'score',return_value='CPU') as score,patch.object(c.source,'activate') as activate:
            self.assertEqual(c.worker({'scorer':'only'},'score'),'CPU')
        score.assert_called_once();activate.assert_not_called()

    def test_failed_source_selection_precedes_preparing_or_target(self):
        self.select.stop()
        with patch.object(c.evaluation,'selected_method',side_effect=ValueError('no source pass')),patch.object(c,'read') as read:
            with self.assertRaises(ValueError):c.template('/project','commit',c.SEEDS[1])
        read.assert_not_called()

    def test_primary_is_weighted_168000_and_cross_date_is_separate(self):
        def data(tasks,selected):
            results=[]
            for task,n in tasks.items():
                pct=({'cross_rx':90,'cross_both':70,'cross_date':10}[task] if selected else 80)
                correct=n*pct//100;cm=[[0]*6 for _ in range(6)];cm[0][0]=correct;cm[0][1]=n-correct
                for view in c.evaluation.VIEWS:results.append(dict(test=task,readout='raw',dimension='overall',view=view,count=n,confusion=cm))
            return dict(truth_last=True,target_feedback_forbidden=True,results=results)
        with tempfile.TemporaryDirectory() as td:
            jobs=[];mapping={};rt=dict(project=td,run_root=td)
            for method in (self.method,'SIG1-Sup','SIG1-PL'):
                for seed in c.SEEDS:
                    selected=method==self.method;rid='F0-'+method+'-s'+str(seed);out=Path(td)/rid
                    cfg=dict(method=method,model_seed=seed,purpose='selected_method' if selected else 'baseline_cross_date',output_root=str(out))
                    name=str(out)+'.json';jobs.append(dict(kind='score',config=name));mapping[name]=cfg
                    mapping[str(out/'scores.json')]=data(c.evaluation.TASK_COUNTS if selected else {'cross_date':30000},selected)
                    if not selected:
                        old=Path(td)/'runs'/c.evaluation.BASELINE_RUN/'readouts'/rid/'scores/scores.json'
                        mapping[str(old)]=data({'cross_rx':126000,'cross_both':42000},False)
            with patch.object(c,'read',side_effect=lambda p:mapping[str(p)]):c.analyze(rt,jobs)
            end=json.loads((Path(td)/'analysis/complete.json').read_text())
            primary=[r for r in end['results'] if r['method']==self.method and r['population']=='overall_168000']
            self.assertEqual([r['mean'] for r in primary],[85.,85.]);self.assertFalse(end['target_goals_pass'])
            diag=[r for r in end['results'] if r['method']==self.method and r['population']=='cross_date']
            self.assertEqual([r['mean'] for r in diag],[10.,10.])


from collections import Counter
if __name__=='__main__':unittest.main()
