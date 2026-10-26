import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_channel_pl import sig1_matched_baselines as b


def make(method,seed,configuration=b.config):
    context=dict(project='/project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
        cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='/old/segment20',
            source_config=dict(ablation='F21',num_classes=6),expected_source_contract='roles') for s in b.SEEDS])
    old=dict(num_classes=6,model_seed=seed,source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='cache',source_contract='roles',hard_pl=dict(weight=6))
    with patch.object(b,'read',return_value=dict(config=old)):return configuration(context,'commit',method,seed)


class MatchedBaselineTests(unittest.TestCase):
    def test_each_seed_uses_its_own_prefix_and_original_budget(self):
        for seed in b.SEEDS:
            for method in b.METHODS:
                c=make(method,seed);b.validate_config(c)
                self.assertEqual(c['model_seed'],seed)
                self.assertEqual((c['epochs'],c['optimizer_updates']),(50,11100))
                self.assertEqual(c['shared_SIG1_E20'],b.prefix('/project',seed) if c['U_use'] else None)
                if c['U_use']:
                    other=next(s for s in b.SEEDS if s!=seed)
                    with self.assertRaises(ValueError):b.validate_config(dict(c,shared_SIG1_E20=b.prefix('/project',other)))

    def test_supervised_cannot_inherit_U_or_enable_any_U_loop(self):
        for seed in b.SEEDS:
            c=make('SIG1-Sup',seed)
            for change in (dict(shared_SIG1_E20=b.prefix('/project',seed)),dict(U_use=True),dict(own_EMA=True),dict(native_U_gate=True)):
                with self.subTest(change=change),self.assertRaises(ValueError):b.validate_config(dict(c,**change))

    def test_training_rejects_target_and_foreign_objectives(self):
        c=make('SIG1-PL',b.SEEDS[1])
        for change in (dict(tests={}),dict(truth='private'),dict(hard_pl={}),dict(sig_time_views=3),dict(plain_PL_weight=6),dict(posthoc_head_fit=True)):
            with self.subTest(change=change),self.assertRaises(ValueError):b.validate_config(dict(c,**change))

    def test_only_own_segments_and_exact_continuation(self):
        for method in b.METHODS:
            c=make(method,b.SEEDS[2]);root=c['output_root'];use=c['U_use']
            for offset,length in ([(20,10),(30,20)] if use else [(0,20),(20,10),(30,20)]):
                parent=(c['shared_SIG1_E20'] if use else root+'/segment20') if offset==20 else root+'/segment30' if offset==30 else None
                s=dict(c,output_root=root+'/segment'+str(offset+length),parent_source=parent,segment_epochs=length,epoch_offset=offset,step_offset=222*offset)
                b.validate_config(s)
                with self.assertRaises(ValueError):b.validate_config(dict(s,parent_source='/foreign'))

    def test_exact_six_models_then_independent_prediction_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=dict(project='/project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
                cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='/old/segment20',source_config=dict(ablation='F21',num_classes=6),expected_source_contract='roles') for s in b.SEEDS])
            target=dict(classes=list(range(6)),same_G=source['G'],restored_caches=dict(LU='same_cache'),tests={t:dict(views_root=t,truth=t+'_truth') for t in b.TASKS})
            (root/'source.json').write_text(json.dumps(source));(root/'target.json').write_text(json.dumps(target))
            with patch.object(b,'config',side_effect=lambda ctx,commit,method,seed:make(method,seed)):
                jobs=b.prepare(dict(context=str(root/'source.json'),evaluation_context=str(root/'target.json'),run_root=str(root),code_commit='commit',release_root='release',python='python',frozen_sources={}))
            self.assertEqual(len(jobs),19)
            self.assertEqual({kind:sum(j['kind']==kind for j in jobs) for kind in ('train','predict','score','smoke')},dict(train=6,predict=6,score=6,smoke=1))
            for j in jobs:
                c=json.loads(Path(j['config']).read_text())
                if j['kind']=='train':
                    self.assertNotIn('tests',c);self.assertEqual(j['dependencies'],[str(root/'smoke.json')])
                if j['kind']=='predict':
                    b.validate_evaluation(c);self.assertTrue(all(set(t)=={'views_root'} for t in c['tests'].values()))
                    self.assertEqual(Path(j['dependencies'][0]).parent.name,'source')
                if j['kind']=='score':
                    b.validate_evaluation(c,scoring=True);self.assertNotIn('source_output',c);self.assertNotIn('frozen_sources',c)
                    self.assertEqual(Path(j['dependencies'][0]).parent.name,'predictions')

    def test_prediction_rejects_truth_and_partial_population(self):
        c=dict(run_id=b.RUN,row_id='F0-SIG1-PL-s2026092701',model_seed=2026092701,method='SIG1-PL',protocol_id='F0',
            source_days=[1,2,3],classes=list(range(6)),readouts=b.READOUTS,restored_caches=dict(LU='cache'),U=True,tests={t:dict(views_root=t) for t in b.TASKS})
        b.validate_evaluation(c)
        for changes in (dict(tests={'cross_rx':{}}),dict(truth='truth'),dict(readouts=['raw']),dict(U=False)):
            with self.assertRaises(ValueError):b.validate_evaluation(dict(c,**changes))
        bad=deepcopy(c);bad['tests']['cross_rx']['truth']='truth'
        with self.assertRaises(ValueError):b.validate_evaluation(bad)

    def test_combined_metric_uses_physical_counts_and_paired_seeds(self):
        views=('clean','h1','h2','h3','h4','h5','h6')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=[]
            for seed in b.SEEDS:
                for method in b.METHODS:
                    folder=root/(method+str(seed));folder.mkdir();cells=[]
                    for task,count in [('cross_rx',126000),('cross_both',42000)]:
                        # Sup: 100% crossRX,0% both ->75%;PL:both100% ->100%.
                        correct=count if task=='cross_rx' or method=='SIG1-PL' else 0
                        cm=[[0]*6 for _ in range(6)];cm[0][0]=correct;cm[0][1]=count-correct
                        cells.extend(dict(readout=route,dimension='overall',test=task,view=view,count=count,confusion=cm) for route in b.READOUTS for view in views)
                    (folder/'scores.json').write_text(json.dumps(dict(truth_last=True,target_feedback_forbidden=True,results=cells)))
                    cfg=folder/'config.json';cfg.write_text(json.dumps(dict(output_root=str(folder),method=method,model_seed=seed)))
                    jobs.append(dict(kind='score',config=str(cfg)))
            b.analyze(dict(run_root=str(root)),jobs)
            result=json.loads((root/'analysis/complete.json').read_text())
            cells=[v for v in result['results'] if v['population']=='overall_168000']
            self.assertTrue(all(v['mean']==(75 if v['method']=='SIG1-Sup' else 100) and v['sample_sd']==0 for v in cells))
            self.assertTrue(all(v['mean_delta_pp']==25 and v['sample_sd_delta_pp']==0 for v in result['paired_PL_minus_Sup'] if v['population']=='overall_168000'))


if __name__=='__main__':unittest.main()
