import copy,json,unittest,tempfile
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.pamf_repl_view_shuffle.plan import make_plan,configuration,validate_config,SEEDS
from experiments.pamf_repl_view_shuffle.runner import eligible

class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx=json.loads(Path('local_artifacts/pamf_repl_view_shuffle_20261027/context.json').read_text())
    def test_only_view_count_differs_in_each_seed_pair(self):
        rows=make_plan()['rows'];self.assertEqual(len(rows),16)
        for seed in SEEDS:
            cs=[configuration(self.ctx,r,'/project','commit') for r in rows if r['model_seed']==seed]
            for c in cs:
                self.assertNotIn('shared_prefix',c);self.assertEqual(c['model_seed'],seed);self.assertEqual(c['trajectory'],[20,10,20])
                for k in ('run_id','row_id','output_root','view_comparison'):c.pop(k)
                c['repl_row'].pop('row_id')
            self.assertEqual(*cs)
    def test_scientific_changes_and_forbidden_inputs_fail(self):
        c=configuration(self.ctx,make_plan()['rows'][0],'/project','commit')
        for key,value in [('truth','x'),('tests',{'x':'target'}),('shared_prefix','foreign'),('checkpoint','foreign'),('own_EMA',True),('U_use',False),('epochs',20),('unlabeled_batch',128)]:
            changed=copy.deepcopy(c);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(changed,self.ctx)
        changed=copy.deepcopy(c);changed['hard_pl_receiver_extrapolation']['alpha']=.5
        with self.assertRaises(ValueError):validate_config(changed,self.ctx)
    def test_foreign_user_cards_and_capacity_are_excluded(self):
        self.assertFalse(eligible(dict(blocked=True,pids=set(),free=24576),0,1))
        self.assertFalse(eligible(dict(blocked=False,pids={1,2},free=24000),0,1))
        self.assertFalse(eligible(dict(blocked=False,pids=set(),free=11000),0,1))
        self.assertTrue(eligible(dict(blocked=False,pids={1},free=16000),0,1))
        self.assertFalse(eligible(dict(blocked=False,pids={1},free=16000),1,1))
        self.assertTrue(eligible(dict(blocked=False,pids={1},free=16000),1,2))
    def test_single_view_waits_for_all_three_view_scores(self):
        from experiments.pamf_repl_view_shuffle.runner import prepare
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ctx=root/'context.json';ctx.write_text(json.dumps(self.ctx))
            jobs=prepare(dict(context=str(ctx),run_root=str(root),project='/project',code_commit='test',frozen_sources={}))
            tri_scores={j['completion'] for j in jobs if j['kind']=='score' and '3view-' in j['id']}
            self.assertEqual(len(tri_scores),8)
            for j in jobs:
                if j['kind']=='train':self.assertEqual(set(j['dependencies']),tri_scores if '1view-' in j['id'] else set())
    def test_new_seed_reaches_core_runtime_and_native_arguments(self):
        from experiments.pamf_repl_view_shuffle.source import seed_scope
        fake=SimpleNamespace(original_config=lambda seed:dict(model_seed=seed),arguments=lambda c,device:SimpleNamespace(seed=c['model_seed']))
        with patch.dict('sys.modules',{'experiments.cric_dura_comparison.f_train':fake}):
            with seed_scope():
                self.assertEqual(fake.original_config(2026092708)['model_seed'],2026092708)
                self.assertEqual(fake.arguments(dict(model_seed=2026092708),'cpu').seed,2026092708)
    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires verified project Torch environment')
    def test_actual_T1_hook_uses_segment_local_step_not_offset(self):
        from experiments.cric_dura_comparison.t1_model import hook
        model=SimpleNamespace(training=True,_feature_l_steps=1,_cd_config=dict(ablation='F21',step_offset=4440),_cd_pending=None)
        kw=dict(model=model,labeled_output=None,labels=None,extra=None,domains=None,optimizer_step=1,identity_loss=None,labeled_count=128,labeled_inputs=None)
        # Correct local step passes the ordering guard and reaches the IQ check.
        with self.assertRaisesRegex(ValueError,'Actual ordinary IQ absent'):hook(model,**kw)
        model._feature_l_steps=4440
        with self.assertRaisesRegex(ValueError,'Ordered true L128'):hook(model,**kw)

if __name__=='__main__':unittest.main()
