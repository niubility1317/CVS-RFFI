"""Small synthetic source roles and fake frozen checkpoints only."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COUNTS = dict(L=90, OE=15, V=90, novel=15)


class StarRepresentationCacheTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_representation_cache.py').is_file(),
                        'Star representation cache is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_representation_cache')

    def fixture(self, root):
        from experiments.pamf_phase3_open_world import star_relative_source as source
        reader=dict(base_release='fixed-base',source_output='original-F0/source',model_seed=2026092701,
                    frozen_sources={k:'frozen-'+k for k in ('F','AF','BASE','FOLD','G')})
        mix=root/'mixture.json';mix.write_text((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(),encoding='utf-8')
        out=root/'run/cache'
        nested=dict(run_id='20261029-phase3-star-representation-s2026092701-r01',code_commit='a'*40,
            reader_config=reader,source_cache='original-source-cache',source_contract='original-contract.json',
            raw_mainsig='original-ManySig.pkl',mixture_config_path=str(mix),episode=source.EPISODE.copy(),
            dura_checkpoint='frozen-G/final.pt',dura_release='frozen-G-release',output_root=str(out),batch_size=128)
        config=dict(source_config=nested,exposure_public_root='G10-OE-public',exposure_cap='OE-cap',exposure_split='OE-split',
            novel_public_root='G10-novel-public',novel_cap='novel-cap',novel_split='novel-split',output_root=str(out),batch_size=128)
        roles={}
        for role,n in COUNTS.items():
            domain=np.arange(n,dtype=np.int64)%15
            meta=dict(rx_i=np.asarray([1,3,4,6,8],np.int64)[domain//3],day_i=domain%3+1,domain=domain,
                      ids=np.asarray([role+'-physical-'+str(i) for i in range(n)]))
            if role in ('L','V'):meta.update(y=np.arange(n,dtype=np.int64)//15,eq_i=np.ones(n,np.int64),
                sig_i=np.arange(n,dtype=np.int64),base_index=np.arange(n,dtype=np.int64))
            roles[role]=meta
        return config,roles

    @staticmethod
    def backend():
        import torch
        models=[torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False) for _ in range(2)]
        for model in models:model.register_buffer('fixed_buffer',torch.tensor([1.],dtype=torch.float32))
        return dict(reader=models[0],dura=(models[1],None),reader_identity={'model':'original_F0_E50','source_output':'original-F0/source','model_seed':2026092701},
                    dura_identity={'deployment_id':'dura_v1','precision':'FP32','TF32':False})

    def mocks(self, m, config, roles, backend, events):
        stack=contextlib.ExitStack()
        stack.enter_context(patch.object(m,'COUNTS',COUNTS))
        stack.enter_context(patch.object(m,'_backend',return_value=backend))
        def smoke(*args):events.append('smoke');return dict(status='PASS',synthetic_physical_rows=2,synthetic_F0_rows=4,synthetic_G_rows=2)
        stack.enter_context(patch.object(m,'_smoke',side_effect=smoke))
        def metadata(*args):events.append('metadata');return {'counts':{'L_s':90,'V':90}}, {'L_s':roles['L'],'V':roles['V']}
        stack.enter_context(patch.object(m,'_source_metadata',side_effect=metadata))
        def raw(*args):events.append('raw');return 'synthetic-payload'
        stack.enter_context(patch.object(m,'_load_raw',side_effect=raw))
        def observations(payload,meta,contract,cfg,mixture):
            events.append('channel:'+str(meta['ids'][0]).split('-')[0]);return np.full((len(meta['ids']),2,256),3.,np.float32),np.full(len(meta['ids']),'practical_mid_urban')
        stack.enter_context(patch.object(m,'_observations',side_effect=observations))
        def public(role,cfg):
            events.append('public:'+role);meta=roles[role]
            return {'capsule_id':cfg['exposure_cap' if role=='OE' else 'novel_cap'],'split_id':cfg['exposure_split' if role=='OE' else 'novel_split']},meta['ids'],np.full((len(meta['ids']),2,256),4.,np.float32),{k:v for k,v in meta.items() if k!='ids'}
        stack.enter_context(patch.object(m,'_public',side_effect=public))
        def restore(b,values,device):events.append(('G',len(values)));return values-.25
        stack.enter_context(patch.object(m,'_restore',side_effect=restore))
        def logits(b,values,device):events.append(('F',len(values)));return np.tile(np.arange(6,dtype=np.float32),(len(values),1))
        stack.enter_context(patch.object(m,'_logits',side_effect=logits))
        return stack

    def test_all_four_roles_exact_files_frozen_models_row_counts_and_no_truth(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config,roles=self.fixture(Path(tmp));backend=self.backend();events=[]
            with self.mocks(m,config,roles,backend,events):
                done=m.build(config,'cpu')
                self.assertEqual(events[:3],['smoke','metadata','raw'])
                self.assertEqual(done['status'],'STAR_REPRESENTATION_CACHE_COMPLETE')
                self.assertEqual(done['cache_model_forward_rows'],{'DURA':210,'F0':270})
                self.assertEqual(sum(x[1] for x in events if isinstance(x,tuple) and x[0]=='G'),210)
                self.assertEqual(sum(x[1] for x in events if isinstance(x,tuple) and x[0]=='F'),270)
                for role in COUNTS:
                    bundle=m.load_role(Path(config['output_root'])/role,role)
                    self.assertEqual(set(p.name for p in (Path(config['output_root'])/role).iterdir()),m.role_files(role))
                    np.testing.assert_array_equal(bundle['ids'],roles[role]['ids'])
                    np.testing.assert_allclose(bundle['restored'],bundle['received']-.25)
                    self.assertFalse(bundle['received'].flags.writeable);self.assertFalse(bundle['restored'].flags.writeable)
                    self.assertEqual(set(bundle['metadata']),m.METADATA_FIELDS[role])
                    self.assertEqual(bundle['manifest']['parent_F0_identity'],backend['reader_identity'])
                    if role=='L':self.assertEqual(bundle['teacher_logits'].shape,(90,6))
                    if role=='V':np.testing.assert_array_equal(bundle['baseline_closed'],np.full((90,2),5,np.int64))
                    m.close_role(bundle)
                self.assertEqual(done['original_U_s_reads'],0);self.assertEqual(done['target_reads'],0)
                self.assertEqual(done['private_truth_reads'],0);self.assertTrue(done['models_unchanged'])

    def test_load_L_and_OE_never_open_parent_complete_or_sibling_roles(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config,roles=self.fixture(Path(tmp));events=[]
            with self.mocks(m,config,roles,self.backend(),events):
                m.build(config,'cpu');out=Path(config['output_root']);(out/'complete.json').unlink()
                original=np.load
                def current_only(path,*args,**kwargs):
                    self.assertIn(Path(path).parent.name,{'L','OE'});return original(path,*args,**kwargs)
                with patch.object(np,'load',side_effect=current_only):
                    l=m.load_L(out/'L');oe=m.load_OE(out/'OE');m.close_role(l);m.close_role(oe)
                with self.assertRaises(ValueError):m.load_role(out/'V','L')
                manifest=out/'OE/manifest.json';data=json.loads(manifest.read_text());data['role']='L';manifest.write_text(json.dumps(data))
                with self.assertRaises(ValueError):m.load_OE(out/'OE')

    def test_invalid_inputs_and_output_collision_precede_models_or_any_source_read(self):
        m=self.module()
        for change in ('truth','target','batch','nested_run','nested_output','existing'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                config,_=self.fixture(Path(tmp))
                if change in ('truth','target'):config[change]='forbidden'
                elif change=='batch':config['batch_size']=64
                elif change=='nested_run':config['source_config']['run_id']='other'
                elif change=='nested_output':config['source_config']['output_root']='other'
                else:Path(config['output_root']).mkdir(parents=True)
                with patch.object(m,'_backend',side_effect=AssertionError('No model')),self.assertRaises((ValueError,FileExistsError)):m.build(config,'cpu')

    def test_smoke_failure_and_model_mutation_never_complete(self):
        m=self.module()
        for change in ('smoke','buffer'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                config,roles=self.fixture(Path(tmp));backend=self.backend();events=[]
                with self.mocks(m,config,roles,backend,events):
                    if change=='smoke':
                        with patch.object(m,'_smoke',side_effect=ValueError('synthetic smoke failed')),self.assertRaises(ValueError):m.build(config,'cpu')
                        self.assertNotIn('raw',events)
                    else:
                        def restore(b,values,device):b['dura'][0].fixed_buffer.add_(1);return values.copy()
                        with patch.object(m,'_restore',side_effect=restore),self.assertRaises(ValueError):m.build(config,'cpu')
                    self.assertFalse((Path(config['output_root'])/'complete.json').exists())

    def test_tensor_bridges_no_numpy_ABI_and_no_inference_mode(self):
        m=self.module()
        import torch
        class Reader(torch.nn.Module):
            def features(self,x):self.assert_grad=torch.is_grad_enabled();return x.mean((1,2))[:,None].repeat(1,160)
            def classify_features(self,x):return x[:,:6]
        reader=Reader().eval().requires_grad_(False)
        def engine(x):return (x*.75,None)
        backend={'reader':reader,'dura':(None,engine)};values=np.ones((2,2,256),np.float32)
        with patch.object(torch.Tensor,'numpy',side_effect=AssertionError('No NumPy ABI')),patch.object(torch,'inference_mode',side_effect=AssertionError('Use no_grad')):
            restored=m._restore(backend,values,'cpu');logits=m._logits(backend,restored,'cpu')
        self.assertEqual(restored.dtype,np.float32);self.assertEqual(logits.shape,(2,6));self.assertFalse(reader.assert_grad)

    def test_actual_synthetic_star_smoke_uses_two_rows_without_source_inputs(self):
        m=self.module()
        import torch
        calls=[]
        class Reader(torch.nn.Module):
            def features(self,x):calls.append(('F',len(x)));return x.mean((1,2))[:,None].repeat(1,160)
            def classify_features(self,x):return x[:,:6]
        def engine(x):calls.append(('G',len(x)));return (x*.75,None)
        backend={'reader':Reader().eval().requires_grad_(False),'dura':(None,engine)}
        with tempfile.TemporaryDirectory() as tmp:
            config,_=self.fixture(Path(tmp))
            with patch.object(m,'_source_metadata',side_effect=AssertionError('No source metadata')), \
                 patch.object(m,'_load_raw',side_effect=AssertionError('No source IQ')):
                done=m._smoke(backend,config['source_config'],'cpu')
        self.assertEqual(calls,[('G',2),('F',2),('F',2)])
        self.assertEqual((done['synthetic_G_rows'],done['synthetic_F0_rows'],done['source_IQ_reads']),(2,4,0))


if __name__=='__main__':unittest.main()
