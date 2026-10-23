import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.practical_full_2x2.plan import make_plan,NEW_RX,SCENES
from experiments.practical_full_2x2.cache import probability,key,configuration,transform

class Scope(unittest.TestCase):
    def test_scope_and_native_budgets(self):
        p=make_plan();r=p['rows'];self.assertEqual(528,sum(c['kind']=='external' for c in r))
        self.assertEqual(66,sum(c.get('regime')=='matched_sat_supervised' for c in r))
        self.assertEqual(6,sum(c.get('regime')=='native_cric' for c in r));self.assertEqual(27,len(p['reused_rows']))
        for cell in 'ABCD':self.assertEqual(132,sum(c.get('cell')==cell and c['kind']=='external' for c in r))
        for c in r:
            if c['kind']=='external':self.assertEqual('full',c['variant']);self.assertEqual(c['U_s_use'],c['cell'] in 'BD')
    def test_rx_roles_and_cancelled_projects(self):
        p=make_plan();sets=[set(p['protocols'][k]['source_rxs']) for k in ('F0',*NEW_RX)]
        for a,b in ((0,1),(0,2),(1,2)):self.assertEqual(1,len(sets[a]&sets[b]))
        self.assertEqual(12,len(set.union(*sets)))
        self.assertNotIn('chrono',p['protocols']);self.assertNotIn('m5-natural',p['protocols'])
    def test_same_native_T1_and_pamf_off(self):
        p=make_plan()
        for c in p['rows']:
            if c.get('regime')=='native_cric':self.assertTrue(c['own_EMA']);self.assertFalse(c['satellite_concat']);self.assertEqual([20,10,20],c['trajectory'])
            if c.get('regime')=='matched_sat_supervised':self.assertFalse(c['own_EMA']);self.assertFalse(c['U_use']);self.assertTrue(c['satellite_concat'])
    def test_schedule(self):
        self.assertEqual([.3,.3,.6,.6,.8,.8],[probability(v) for v in (0,.199,.2,.449,.45,1)])
        for bad in (-.1,1.1):
            with self.assertRaises(ValueError):probability(bad)
        with self.assertRaises(ValueError):key(0,1,.3)
    def test_practical_route_and_function(self):
        for scene in SCENES:
            c=configuration(scene);self.assertEqual('residual',c.processing_route);self.assertEqual('post_sync',c.mode);self.assertFalse(c.equalization_enabled)
        x=np.random.default_rng(2).normal(size=(2,256)).astype(np.float32)
        a,s,l=transform(x,'source-only',3,2,1,1,.8);b,t,_=transform(x,'source-only',3,2,1,1,.8)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(s,t);self.assertEqual(np.float32,a.dtype)

class Permissions(unittest.TestCase):
    def test_runner_bootstrap_precedes_legacy_plan_import(self):
        import ast
        source=Path(__file__).resolve().parents[1]/'experiments/practical_full_2x2/runner.py'
        tree=ast.parse(source.read_text())
        imports=[n for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.module in ('cache','plan') for n in imports))
        dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
        first_call=next(n for n in dispatch.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call))
        self.assertEqual('activate',first_call.value.func.id)

    def test_supervised_never_opens_U(self):
        from experiments.practical_full_2x2.source import array_loader
        with tempfile.TemporaryDirectory() as root:
            Path(root,'metadata.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'cannot iterate U'):array_loader(dict(source_cache=root,U_s_use=False),'U_s')
    def test_source_rejects_target_and_legacy_CE(self):
        from experiments.practical_full_2x2.source import check_training
        c=next(c for c in make_plan()['rows'] if c['kind']=='external')
        with self.assertRaises(ValueError):check_training(dict(c,truth='forbidden'))
        with self.assertRaises(ValueError):check_training(dict(c,variant='ce'))
    def test_prediction_rejects_truth(self):
        from experiments.cric_dura_comparison.evaluation import reject_truth
        with self.assertRaises(ValueError):reject_truth({'tests':{'cross_date':{'truth':'not_permitted'}}})

    def test_no_channel_frozen_reader_accepts_zero_applied(self):
        import torch
        from unittest.mock import patch
        from experiments.practical_full_2x2 import source
        c=dict(satellite_train_augmentation=False,U_s_use=False)
        net=torch.nn.Linear(2,2)
        with tempfile.TemporaryDirectory() as d:
            Path(d,'freeze.json').write_text(json.dumps(dict(config=c,status='SOURCE_TRAINED_FROZEN',source_contract={},
                target_access=False,target_score_used=False,satellite_applied_examples=0,checkpoint='mock')))
            payload=dict(default_head_joint_training=True,separate_head_training=False,state_dict=net.state_dict(),training_telemetry={})
            with patch.object(source,'check_training'),patch.object(source.native,'load',return_value=payload),patch.object(source,'build',return_value=net):
                self.assertIs(net,source.frozen_model(d,{},'cpu')[0])

class SharedCache(unittest.TestCase):
    def test_same_FP32_function_permuted_consumers(self):
        from experiments.practical_full_2x2.cache import produce,Reader,read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir();ids=np.asarray(['L0','L1','L2','L3']);n=4
            x=np.random.default_rng(7).normal(size=(n,2,256)).astype(np.float32)
            np.save(src/'L_s_x.npy',x);np.savez(src/'L_s_metadata.npz',ids=ids,rx_i=np.asarray([1,1,3,3]),day_i=np.asarray([1,2,1,2]))
            np.savez(src/'L_s_index.npz',ids=ids)
            contract=dict(role_ids={'L_s':ids.tolist()});Path(src/'source_contract.json').write_text(json.dumps(contract))
            c=dict(protocol_id='test',source_cache=str(src),source_contract=str(src/'source_contract.json'),augmentation_cache=str(root/'cache'))
            request=dict(protocol_id='test',source_cache=str(src),source_contract=c['source_contract'],cache_root=c['augmentation_cache'],epoch=1,stage=1,probability=.8)
            produce(request,workers=1)
            reader=Reader(c);out,_=reader.get([2,0,2],1,1,.5,1)
            expected=np.stack([transform(x[j],str(ids[j]),[1,1,3,3][j],[1,2,1,2][j],1,1,.8)[0] for j in (2,0,2)])
            self.assertTrue(np.array_equal(out,expected));self.assertEqual(4,read(root/'cache/test'/key(1,1,.8)/'complete.json')['count'])
            reader.arrays.clear()

class ResolvedQueue(unittest.TestCase):
    def test_all_stage_configs_and_legacy_G_descriptor(self):
        from unittest.mock import patch
        from experiments.practical_full_2x2.prepare import prepare
        from experiments.practical_full_2x2 import data
        from experiments.cric_dura_comparison.io import read,write
        p=make_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);handles={};templates={};oldjobs=[]
            for pid,q in p['protocols'].items():
                contract=root/(pid+'.json');classes=[str(i) for i in range(10 if pid=='ManyRx' else 6)]
                write(contract,dict(role_ids={'L_s':['L'],'U_s':['U'],'V':['V']},counts=q['expected_counts'],
                    source_rxs=q['source_rxs'],source_days=q['source_days'],classes=classes))
                h=dict(source_cache=str(root/pid),source_contract=str(contract),classes=classes,counts=q['expected_counts'],
                    tests={t:dict(views_root='blind-'+t,truth='independent-'+t) for t in ('cross_date','cross_rx','cross_both')})
                h['external']={r:dict(source_cache=h['source_cache'],source_contract=str(contract)) for r in ('supervised','pl')};handles[pid]=h
                # Historic descriptors really omit source_contract inside G.
                templates[pid]=dict(source_contract=str(contract),readouts={f'DURA-{role}':dict(kind='contract_matched',roles=role,checkpoint='frozen.pt') for role in ('L','LU')})
            for i,r in enumerate(p['reused_rows']):
                src=root/f'old{i}';src.mkdir();write(src/'complete.json',dict(status='SOURCE_FROZEN',config=dict(source_contract=handles[r['protocol_id']]['source_contract'])))
                c=root/f'old-config{i}.json';write(c,dict(run_id='20261021-phase1-T1-DURA-comparison-m153-r01',row_id=f"{r['protocol_id']}-native_cric-s{r['model_seed']}",source_output=str(src)))
                oldjobs.append(dict(kind='predict',config=str(c)))
            write(root/'oldjobs.json',oldjobs)
            run=root/'new';run.mkdir();rt=dict(run_root=str(run),old_jobs=str(root/'oldjobs.json'),G_templates=templates,
                code_commit='test',frozen_sources={},external_releases=[],python='cpu',external_python='cpu',
                ibrff_model_root='author-IB',sdg_model_root='author-SDG',augmentation_cache=str(run/'cache'))
            with patch.object(data,'prepare',return_value=handles),patch.object(data,'old_G_selector',return_value={}):jobs=prepare(rt,p)
            self.assertEqual(1880,len(jobs))
            for job in jobs:
                c=read(job['config']);self.assertEqual(job['kind'],c['stage'])
                if job['kind']=='F':
                    self.assertNotIn('tests',c);self.assertNotIn('truth',c)
                    if c['kind']=='pamf':self.assertIn('ablation',c);self.assertIsNone(c['ablation'])

if __name__=='__main__':unittest.main()
