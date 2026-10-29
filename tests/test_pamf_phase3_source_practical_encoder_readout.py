"""Synthetic source-only contracts for evaluation of the two new encoders."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_encoder_readout')


def geometry(mean=0.):
    from experiments.pamf_phase3_open_world import source_open_readout as core
    from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
    result=dict(schema=core.GEOMETRY_SCHEMA,class_count=6,branches={
        name:dict(mean=np.full(d,mean),scale=np.ones(d),prototypes=np.zeros((6,d)),whitener=np.eye(d))
        for name,d in DIMENSIONS.items()})
    result['branches']['reference29']['mean'][27]=0.
    return result


def state(nk,no):
    from experiments.pamf_phase3_open_world import source_open_readout as core
    return dict(schema=core.READOUT_SCHEMA,descriptor_dim=100,method_names=list(core.METHODS),
        mean=np.zeros(100),scale=np.ones(100),models={
            'linear_oe':dict(weight=np.zeros((1,100),np.float32),bias=np.zeros(1,np.float32)),
            'mlp16_oe':dict(weight1=np.zeros((16,100),np.float32),bias1=np.zeros(16,np.float32),
                weight2=np.zeros((1,16),np.float32),bias2=np.zeros(1,np.float32))},
        training=dict(steps=200,known_count=nk,oe_count=no,parameter_counts={'linear_oe':101,'mlp16_oe':1633},
            losses={k:[.5]*200 for k in ('linear_oe','mlp16_oe')}))


class EncoderReadoutTests(unittest.TestCase):
    def fixture(self,m,root,arm='clean_increment'):
        import torch
        cfg=dict(run_id=m.RUN,code_commit='a'*40,arm=arm,train_root=str(root/arm/'train'),
            source_row_root=str(root/'20261029-phase3-source-embedding-readout-s2026092701-r01/rows/uniform_oe'),
            risk_prediction_root=str(root/'20261029-phase3-source-baseline-risk-s2026092701-r01/rows/uniform_oe/predict'),
            oe_data_root=str(root/'source_oe'),dura_checkpoint=str(root/'G/final.pt'),dura_release=str(root/'G'),
            output_root=str(root/arm/'source'),batch_size=128)
        old={'run_id':'old-E51','updates':222,'cumulative_updates':11322}
        ident=dict(run_id=m.RUN,arm=arm,code_commit=cfg['code_commit'],checkpoint=str(root/arm/'train/final.pt'),
            updates=222,cumulative_updates=11544,parent_encoder_identity=old)
        models=[torch.nn.Linear(1,1).eval().requires_grad_(False) for _ in range(3)]
        setup=dict(model=models[0],reader=models[0],baseline=models[1],dura=models[2],engine=None,
            source=dict(marker=dict(encoder_state_identity=old,parent_export_identity={'baseline':'originalF0'}),
                geometry=geometry(99.),state=state(1,1),thresholds=np.zeros(7)),contract={},
            channels={},api=None,sessions={},identity=ident)
        roles={};n=30
        for role in m.COUNTS:
            raw=np.zeros((n,2,256),np.float32);raw[:,0,0]=np.tile(np.arange(6),5)
            raw[:,0,1]=list(m.COUNTS).index(role)+10
            meta=dict(ids=np.asarray([role+'-'+str(i) for i in range(n)]),
                rx_i=np.repeat(np.asarray(m.RECEIVERS,np.int64),6),day_i=np.ones(n,np.int64),
                domain=np.zeros(n,np.int64))
            if role in ('source_L','source_V'):meta['y']=np.tile(np.arange(6),5).astype(np.int64)
            roles[role]=(raw,meta)
        return cfg,setup,roles

    def patch_pipeline(self,m,cfg,setup,roles,events,*,bad_budget=False,mutate=False):
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.object(m,'COUNTS',{r:30 for r in m.COUNTS}))
        stack.enter_context(mock.patch.object(m,'_setup',return_value=setup))
        def load(config,current,role):
            if role=='novel_validation':
                self.assertTrue((Path(cfg['output_root'])/'calibration/complete.json').is_file())
                events.append(('novel_after_tau',))
            events.append(('load',role));return roles[role]
        stack.enter_context(mock.patch.object(m,'_load_role',side_effect=load))
        def encode(current,raw,meta,view,device,baseline=False):
            from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
            n=len(raw);y=np.asarray(raw)[:,0,0].astype(int)%6
            events.append(('encode',str(meta['ids'][0]),view))
            branches={name:np.full((n,d),float(np.asarray(raw)[0,0,1]),np.float32) for name,d in DIMENSIONS.items()}
            branches['reference29'][:,27]=0
            if mutate and str(meta['ids'][0]).startswith('source_V'):
                with __import__('torch').no_grad():setup['model'].weight.add_(1.)
            return dict(branches=branches,logits=np.eye(6,dtype=np.float32)[y]*2,
                baseline_closed=y.astype(np.int64) if baseline else np.full(n,-2,np.int64))
        stack.enter_context(mock.patch.object(m,'_encode_batch',side_effect=encode))
        def fit_geometry(branches,y):
            events.append(('geometry',));self.assertEqual(len(y),30)
            self.assertTrue(np.all(branches['time160']==10.))
            return geometry(3.)
        stack.enter_context(mock.patch.object(m.core,'fit_geometry',side_effect=fit_geometry))
        def fit_readouts(known,oe,seed):
            events.append(('readouts',));self.assertEqual(known.shape,(120,100));self.assertEqual(oe.shape,(120,100))
            out=state(len(known),len(oe))
            if bad_budget:out['training']['steps']=199
            return out
        stack.enter_context(mock.patch.object(m.core,'fit_readouts',side_effect=fit_readouts))
        return stack

    def test_config_exact_roles_identity_and_source_output_before_IO(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,setup,roles=self.fixture(m,Path(td))
            self.assertEqual(m.validate_config(cfg),cfg)
            for key,value in (('unknown_count',3),('target_U_root','forbidden'),('arm','wrong'),('batch_size',64)):
                with self.subTest(key=key),mock.patch.object(m,'_setup') as backend:
                    bad=dict(cfg);bad[key]=value
                    with self.assertRaises(ValueError):m.run(bad,'cpu')
                    backend.assert_not_called()

    def test_smoke_new_cleanL_geometry_fit_L_OE_only_tau_before_novel_and_roundtrip(self):
        m=module();events=[]
        with tempfile.TemporaryDirectory() as td:
            cfg,setup,roles=self.fixture(m,Path(td))
            with self.patch_pipeline(m,cfg,setup,roles,events):
                result=m.run(cfg,'cpu')
                self.assertEqual(result['status'],m.COMPLETE)
                self.assertEqual(events[0][0],'encode')
                self.assertTrue(events[0][1].startswith('encoder-source-synthetic'))
                self.assertEqual(sum(e[0]=='geometry' for e in events),1)
                self.assertLess(events.index(('geometry',)),events.index(('readouts',)))
                self.assertLess(events.index(('readouts',)),events.index(('load','source_V')))
                self.assertIn(('novel_after_tau',),events)
                prior=list(events);bundle=m.load_bundle(cfg['output_root'])
                self.assertEqual(events,prior)
                self.assertEqual(bundle['encoder_identity'],setup['identity'])
                self.assertEqual(bundle['thresholds'].shape,(7,))
                self.assertEqual(bundle['geometry']['branches']['time160']['mean'][0],3.)
                calibration=bundle['calibration_marker']
                self.assertEqual(calibration['known_count'],120)
                self.assertEqual(calibration['unknown_calibration_scores_read'],0)
                for method in calibration['methods'].values():
                    self.assertEqual(set(method['by_view']),set(m.VIEWS))
                    self.assertEqual(len(method['by_receiver']),5)
                    self.assertEqual(sum(len(v) for v in method['by_receiver_view'].values()),20)
                self.assertEqual(set(bundle),{'geometry','state','thresholds','encoder_identity','dura_identity',
                    'fit_marker','calibration_marker','source_marker','method_names'})
                root=Path(cfg['output_root'])
                self.assertFalse(any('features__' in k for f in root.rglob('*.npz') for k in np.load(f,allow_pickle=False).files))

    def test_wrong_encoder_parent_arm_updates_or_checkpoint_rejected(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,current,roles=self.fixture(m,Path(td))
            parent=current['source']['marker']['encoder_state_identity']
            for key,value in (('parent_encoder_identity',{'wrong':'parent'}),('arm','practical_increment'),
                    ('updates',221),('cumulative_updates',11322),('checkpoint',str(Path(td)/'other/final.pt'))):
                with self.subTest(key=key):
                    bad=dict(current['identity']);bad[key]=value
                    with self.assertRaises(ValueError):m._encoder_identity(bad,cfg,parent)

    def test_smoke_failure_and_unknown_y_stop_before_fit_or_novel(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,setup,roles=self.fixture(m,Path(td));events=[]
            with self.patch_pipeline(m,cfg,setup,roles,events),mock.patch.object(m,'_smoke',side_effect=ValueError('smoke')):
                with self.assertRaises(ValueError):m.run(cfg,'cpu')
                self.assertFalse(any(e[0]=='load' for e in events))
        with tempfile.TemporaryDirectory() as td:
            cfg,setup,roles=self.fixture(m,Path(td));events=[]
            roles['exposure_train'][1]['y']=np.zeros(30,np.int64)
            with self.patch_pipeline(m,cfg,setup,roles,events):
                with self.assertRaises(ValueError):m.run(cfg,'cpu')
                self.assertNotIn(('readouts',),events);self.assertNotIn(('load','novel_validation'),events)

    def test_bad_readout_budget_and_frozen_parameter_mutation_cannot_complete(self):
        m=module()
        for which in ('budget','mutation'):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as td:
                cfg,setup,roles=self.fixture(m,Path(td));events=[]
                with self.patch_pipeline(m,cfg,setup,roles,events,bad_budget=which=='budget',mutate=which=='mutation'):
                    with self.assertRaises(ValueError):m.run(cfg,'cpu')
                    self.assertFalse((Path(cfg['output_root'])/'complete.json').exists())
                    self.assertNotIn(('load','novel_validation'),events)

    def test_independent_score_requires_fit_calibration_and_all_views_before_truth(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,setup,roles=self.fixture(m,Path(td));events=[]
            with self.patch_pipeline(m,cfg,setup,roles,events):
                m.run(cfg,'cpu');root=Path(cfg['output_root'])
                truth=Path(td)/'truth.npz';np.savez(truth,ids=roles['novel_validation'][1]['ids'],tx=np.asarray(['N0','N1','N2']*10))
                for rel in ('fit/complete.json','calibration/complete.json','predict/'+m.VIEWS[-1]+'/predictions.npz'):
                    file=root/rel;raw=file.read_bytes();file.unlink()
                    with self.subTest(missing=rel),mock.patch.object(m,'_novel_truth') as private:
                        with self.assertRaises((OSError,ValueError)):m.score_source(root,truth,Path(td)/'bad')
                        private.assert_not_called()
                    file.write_bytes(raw)
                scored=m.score_source(root,truth,Path(td)/'score')
                self.assertTrue(scored['truth_last']);self.assertFalse(scored['final_phase3_complete'])
                self.assertEqual(set(scored['methods']),set(m.METHODS))
                one=scored['methods']['mlp16_oe']
                self.assertEqual(set(one['by_receiver_view']),set(map(str,m.RECEIVERS)))
                self.assertEqual(set(one['by_view']),set(m.VIEWS))
                self.assertIsNone(one['overall']['confidence_intervals95'])

    def test_safe_numpy_bridge_no_inference_mode_and_recomputed_default_logits(self):
        m=module();import torch
        from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
        raw=np.ones((2,2,256),np.float32);meta=dict(ids=np.asarray(['a','b']),rx_i=np.asarray([1,3]),day_i=np.asarray([1,2]))
        def encode(reader,x):
            self.assertFalse(torch.is_grad_enabled());self.assertFalse(torch.is_inference_mode_enabled())
            self.assertEqual(x.dtype,torch.float32)
            result={'features__'+k:np.zeros((2,d),np.float32) for k,d in DIMENSIONS.items()}
            result['logits']=np.tile(np.arange(6,dtype=np.float32),(2,1))
            return result
        native=torch.as_tensor
        def no_ndarray(value,*a,**kw):
            if isinstance(value,np.ndarray):raise RuntimeError('NumPy ABI unsupported')
            return native(value,*a,**kw)
        with mock.patch.object(torch,'as_tensor',side_effect=no_ndarray),mock.patch.object(m.frozen,'_encode_new',side_effect=encode), \
                mock.patch.object(m.frozen,'_original_closed',return_value=np.asarray([5,5],np.int64)):
            result=m._encode_batch(dict(reader=object(),baseline=object()),raw,meta,'clean','cpu',True)
            self.assertTrue(np.all(result['logits'].argmax(1)==5))

    def test_cli_first_line_and_missing_source_bundle_cannot_read_rows(self):
        m=module()
        with mock.patch.object(m,'run',return_value={'status':'SYNTHETIC'}),mock.patch.object(m,'_read',return_value={}),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main(['run','--config','fake','--device','cpu']),0)
        event=json.loads(out.getvalue().splitlines()[0]);self.assertEqual(event['event'],'SOURCE_PRACTICAL_ENCODER_EVAL_PROCESS')
        self.assertTrue(event['pid']);self.assertTrue(event['cwd'])
        with tempfile.TemporaryDirectory() as td,self.assertRaises((OSError,ValueError)):m.load_bundle(td)


if __name__=='__main__':unittest.main()
