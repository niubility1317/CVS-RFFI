"""Synthetic-only source statistics, view permissions and aggregate deployment."""
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
    return importlib.import_module('experiments.pamf_phase3_open_world.unlabeled_registration_source')


def geometry():
    from experiments.pamf_phase3_open_world.source_open_readout import GEOMETRY_SCHEMA
    from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
    return dict(schema=GEOMETRY_SCHEMA,class_count=6,branches={k:dict(mean=np.zeros(d),scale=np.ones(d),
        prototypes=np.zeros((6,d)),whitener=np.eye(d)) for k,d in DIMENSIONS.items()})


def state():
    from experiments.pamf_phase3_open_world.source_open_readout import READOUT_SCHEMA
    return dict(schema=READOUT_SCHEMA,descriptor_dim=100,method_names=['linear_oe','mlp16_oe'],
        mean=np.zeros(100),scale=np.ones(100),models={
        'linear_oe':dict(weight=np.zeros((1,100),np.float32),bias=np.zeros(1,np.float32)),
        'mlp16_oe':dict(weight1=np.zeros((16,100),np.float32),bias1=np.zeros(16,np.float32),
            weight2=np.zeros((1,16),np.float32),bias2=np.zeros(1,np.float32))},training={})


class RegistrationSourceTests(unittest.TestCase):
    def fixture(self,m,root):
        import torch
        cfg=dict(run_id=m.RUN,code_commit='a'*40,source_row_root=str(root/'old/rows/uniform_oe'),
            risk_prediction_root=str(root/'risk/rows/uniform_oe/predict'),aug_fit_root=str(root/'aug/source/fit'),
            aug_calibration_root=str(root/'aug/source/calibration'),training_cache_root=str(root/'cache'),
            source_validation_export_root=str(root/'aug/source/export'),dura_checkpoint=str(root/'G/final.pt'),
            dura_release=str(root/'G'),output_root=str(root/'new-source'),batch_size=128)
        identity={'run_id':'original-E51','arm':'uniform_oe'};parent={'identity':'F0-E50'}
        g=dict(deployment_id='dura_v1',checkpoint=cfg['dura_checkpoint'],release_root=cfg['dura_release'])
        src=dict(marker=dict(encoder_state_identity=identity,parent_export_identity=parent),geometry=geometry(),state=state(),thresholds=np.ones(7)*999.)
        src['state']['mean'][:]=33.  # New source must copy aug mean0, never original clean state.
        aug=dict(geometry=geometry(),state=state(),thresholds=np.ones(7)*-999.,identities=dict(
            encoder_state_identity=identity,parent_export_identity=parent,dura_identity=g),arm='augmented_readout')
        cache=dict(encoder_state_identity=identity,parent_export_identity=parent,dura_identity=g,
            source_contract_handle=dict(counts=dict(L_s=180,U_s=56700,V=180)),source_only=True,
            counts=dict(source_L=180,source_OE=6134),same_physical_ID_order_across_views=True)
        exp=dict(encoder_state_identity=identity,parent_export_identity=parent,dura_identity=g,
            geometry_source_fit_root=cfg['source_row_root']+'/fit',descriptor_schema=m.consensus.DESCRIPTOR_SCHEMA,
            counts=dict(source_L=180,source_V=180,exposure_train=4200,novel_validation=3150),source_only=True)
        keys=np.asarray([(y,r,d) for y in range(6) for r in m.RECEIVERS for d in (1,2,3)],np.int64)
        labels=np.repeat(keys[:,0],2);rx=np.repeat(keys[:,1],2);day=np.repeat(keys[:,2],2);n=len(labels)
        meta=dict(ids=np.asarray(['L-'+str(i) for i in range(n)]),y=labels,rx_i=rx,day_i=day,
            domain=np.searchsorted(m.RECEIVERS,rx)*3+day-1,eq_i=np.ones(n,np.int64),
            sig_i=np.zeros(n,np.int64),base_index=np.arange(n,dtype=np.int64))
        arrays={view:np.full((n,2,256),j,np.float32) for j,view in enumerate(m.VIEWS)}
        validation={}
        for j,view in enumerate(m.VIEWS):
            validation[view]=dict(ids=np.asarray(['V-'+str(i) for i in range(n)]),y=labels.copy(),rx_i=rx.copy(),day_i=day.copy(),
                descriptor=np.full((n,100),j,np.float32),static_scores=np.tile(np.arange(n,dtype=np.float64)[:,None]/100+j,(1,5)),
                logits=np.eye(6,dtype=np.float32)[(labels+1)%6]*10,baseline_closed=labels.copy())
        models=[torch.nn.Linear(1,1).eval().requires_grad_(False) for _ in range(3)]
        return cfg,dict(source=src,aug=aug,cache=cache,export=exp),arrays,meta,validation,models,keys

    def patch(self,m,inputs,arrays,meta,validation,models,keys,events):
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.object(m,'L_COUNT',len(meta['ids'])))
        stack.enter_context(mock.patch.object(m,'V_COUNT',len(meta['ids'])))
        stack.enter_context(mock.patch.object(m.frozen,'_source_bundle',return_value=inputs['source']))
        stack.enter_context(mock.patch.object(m.readout,'load_bundle',return_value=inputs['aug']))
        stack.enter_context(mock.patch.object(m.cache_api,'cache_marker',return_value=inputs['cache']))
        stack.enter_context(mock.patch.object(m.exporter,'export_marker',return_value=inputs['export']))
        def backend(prepared,c,device):events.append('backend');return dict(reader=models[0],original=models[1],dura=models[2],engine=None)
        stack.enter_context(mock.patch.object(m,'_backend',side_effect=backend))
        def smoke(prepared,backend,device):events.append('smoke');return dict(status='PASS',real_checkpoint=True,synthetic_rows=2)
        stack.enter_context(mock.patch.object(m,'_smoke',side_effect=smoke))
        def load(root,role,views):
            events.append(('load',role));self.assertEqual(role,'source_L');self.assertIn('smoke',events)
            return arrays,meta
        stack.enter_context(mock.patch.object(m.cache_api,'load_role',side_effect=load))
        def features(prepared,backend,raw,device):
            n=len(raw);value=np.asarray(raw)[:,0,0].copy();events.append(('features',value.tolist()))
            f0=np.zeros((n,160),np.float32);f0[:,0]=1
            return np.tile(value[:,None],(1,688)).astype(np.float32),f0,np.tile(value[:,None],(1,7)).astype(np.float64)
        stack.enter_context(mock.patch.object(m,'_features',side_effect=features))
        def density(z,f0,y,rx,day):
            events.append(('density',len(z)));self.assertEqual(len(z),len(meta['ids']))
            return dict(eps=.2,radius=.3,anchor_units=np.tile(np.eye(1,160,0,dtype=np.float32),(90,1)),
                group_keys=keys.copy(),group_indices=tuple(np.flatnonzero((y==k)&(rx==r)&(day==d)) for k,r,d in keys),diagnostics={'count':len(z)})
        stack.enter_context(mock.patch.object(m,'_density',side_effect=density))
        def load_v(root,role,view,include_y):
            events.append(('validation',view));self.assertEqual(role,'source_V');self.assertTrue(include_y)
            return validation[view]
        stack.enter_context(mock.patch.object(m.exporter,'load_role_view',side_effect=load_v))
        return stack

    def test_invalid_config_and_cache_or_export_identity_before_backend(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,inputs,arrays,meta,val,models,keys=self.fixture(m,Path(td));events=[]
            for badkey in ('target_root','unknown_count'):
                bad=dict(cfg);bad[badkey]='forbidden'
                with self.subTest(key=badkey),mock.patch.object(m,'_backend') as backend:
                    with self.assertRaises(ValueError):m.run(bad,'cpu')
                    backend.assert_not_called()
            for kind in ('cache','export','aug'):
                with self.subTest(identity=kind),self.patch(m,inputs,arrays,meta,val,models,keys,events),mock.patch.object(m,'_backend') as backend:
                    if kind=='aug': old=inputs[kind]['identities']['encoder_state_identity'];inputs[kind]['identities']['encoder_state_identity']={'wrong':'F'}
                    else:old=inputs[kind]['encoder_state_identity'];inputs[kind]['encoder_state_identity']={'wrong':'F'}
                    try:
                        with self.assertRaises(ValueError):m.run(cfg,'cpu')
                        backend.assert_not_called()
                    finally:
                        if kind=='aug':inputs[kind]['identities']['encoder_state_identity']=old
                        else:inputs[kind]['encoder_state_identity']=old

    def test_invalid_fit_or_capsule_handle_stops_before_backend(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,inputs,arrays,meta,val,models,keys=self.fixture(m,Path(td));events=[]
            for owner,name in ((m.readout,'load_bundle'),(m.cache_api,'cache_marker'),(m.exporter,'export_marker')):
                with self.subTest(handle=name),self.patch(m,inputs,arrays,meta,val,models,keys,events), \
                        mock.patch.object(owner,name,side_effect=ValueError('wrong fit/role/capsule')),mock.patch.object(m,'_backend') as backend:
                    with self.assertRaises(ValueError):m.run(cfg,'cpu')
                    backend.assert_not_called()

    def test_smoke_wrapper_records_actual_pass_and_mutation_prevents_complete(self):
        m=module()
        with mock.patch.object(m.frozen,'_synthetic_smoke',return_value=None) as smoke:
            value=m._smoke({'source':{}},dict(reader=1,original=2,engine=3),'cpu')
            self.assertEqual(value['status'],'PASS');self.assertEqual(value['F0_forward_rows'],4)
            smoke.assert_called_once()
        with tempfile.TemporaryDirectory() as td:
            cfg,inputs,arrays,meta,val,models,keys=self.fixture(m,Path(td));events=[]
            def mutation(prepared,backend,raw,device):
                inputs['aug']['state']['mean'][0]+=1.
                n=len(raw);f0=np.zeros((n,160),np.float32);f0[:,0]=1.
                return np.ones((n,688),np.float32),f0,np.ones((n,7),np.float64)
            with self.patch(m,inputs,arrays,meta,val,models,keys,events),mock.patch.object(m,'_features',side_effect=mutation):
                with self.assertRaisesRegex(ValueError,'changed'):m.run(cfg,'cpu')
                self.assertFalse((Path(cfg['output_root'])/'complete.json').exists())

    def test_source_selection_one_view_per_physical_independent_of_labels(self):
        m=module();a=m.source_view_indices(6300,'P1');b=m.source_view_indices(6300,'P1')
        np.testing.assert_array_equal(a,b)
        self.assertEqual(set(a.tolist()),{1,2,3});self.assertEqual(len(a),6300)
        self.assertTrue(np.all(m.source_view_indices(6300,'C0')==0))
        with self.assertRaises(TypeError):m.source_view_indices(6300,'P1',y=np.zeros(6300))

    def test_complete_aggregates_f0_calibration_sigma_and_no_source_bank(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,inputs,arrays,meta,val,models,keys=self.fixture(m,Path(td));events=[]
            oldstate={k:v.copy() for k,v in inputs['aug']['state'].items() if isinstance(v,np.ndarray)}
            actual=m.risk.calibrate_known
            def calibrate(ids,y,closed,baseline,scores,rx,view):
                np.testing.assert_array_equal(closed,y);np.testing.assert_array_equal(baseline,y)
                # E51 logits intentionally predict a different class.
                return actual(ids,y,closed,baseline,scores,rx,view)
            with self.patch(m,inputs,arrays,meta,val,models,keys,events),mock.patch.object(m.risk,'calibrate_known',side_effect=calibrate):
                result=m.run(cfg,'cpu');self.assertEqual(result['status'],m.STATUS)
                self.assertEqual(sum(len(e[1]) for e in events if isinstance(e,tuple) and e[0]=='features'),2*len(meta['ids']))
                before=list(events);bundle=m.load_source(cfg['output_root']);self.assertEqual(events,before)
                stats=bundle['statistics'];self.assertEqual(stats['anchor_units'].shape,(2,90,160));self.assertEqual(stats['anchor_scores'].shape,(2,90,7))
                self.assertEqual(stats['sigma'].shape,(2,7));self.assertTrue(np.all(stats['sigma']>=1e-6))
                score=np.stack([m.risk.score_compact(val[v]['descriptor'],val[v]['static_scores'],inputs['aug']['state']) for v in m.VIEWS])
                np.testing.assert_allclose(stats['sigma'][0],np.maximum(score[0].std(axis=0),1e-6))
                np.testing.assert_allclose(stats['sigma'][1],np.maximum(score[1:].reshape(-1,7).std(axis=0),1e-6))
                self.assertEqual(set(bundle),{'geometry','state','thresholds','statistics','marker'})
                self.assertFalse(any(k in stats for k in ('ids','group_indices','source_features','z')))
                for k,v in oldstate.items():np.testing.assert_array_equal(bundle['state'][k],v)
                root=Path(cfg['output_root']);self.assertFalse(any(x.name in ('iq.npy','features.npy') for x in root.rglob('*')))
                file=root/'calibration/complete.json';payload=file.read_bytes();file.unlink()
                with self.assertRaises((OSError,ValueError)):m.load_source(root)
                file.write_bytes(payload)

    def test_smoke_failure_prevents_source_arrays_and_no_fit_call(self):
        m=module()
        with tempfile.TemporaryDirectory() as td:
            cfg,inputs,arrays,meta,val,models,keys=self.fixture(m,Path(td));events=[]
            with self.patch(m,inputs,arrays,meta,val,models,keys,events),mock.patch.object(m,'_smoke',side_effect=ValueError('smoke')),mock.patch.object(m.readout.core,'fit_readouts') as fit:
                with self.assertRaises(ValueError):m.run(cfg,'cpu')
                self.assertFalse(any(isinstance(e,tuple) and e[0]=='load' for e in events));fit.assert_not_called()

    def test_no_numpy_abi_or_inference_tensor_for_f0_unit_and_auxiliary(self):
        m=module();import torch
        from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
        class Original:
            def features(self,x):
                self_test.assertFalse(torch.is_inference_mode_enabled());self_test.assertFalse(torch.is_grad_enabled())
                return torch.ones((len(x),160),dtype=torch.float32)
        self_test=self
        def encoded(reader,x):
            d={'features__'+k:np.zeros((2,v),np.float32) for k,v in DIMENSIONS.items()};d['logits']=np.zeros((2,6),np.float32);return d
        def abi(*a,**kw):raise RuntimeError('ndarray ABI unavailable')
        with mock.patch.object(torch,'as_tensor',side_effect=abi),mock.patch.object(m.frozen,'_encode_new',side_effect=encoded), \
                mock.patch.object(m,'_embedding',return_value=np.zeros((2,688),np.float32)):
            prepared=dict(source=dict(geometry=geometry(),state=state()))
            z,f0,s=m._features(prepared,dict(reader=object(),original=Original()),np.zeros((2,2,256),np.float32),'cpu')
            self.assertEqual(z.shape,(2,688));np.testing.assert_allclose(np.linalg.norm(f0,axis=1),1.,rtol=1e-6)

    def test_cli_reports_actual_pid_cwd_before_run(self):
        m=module()
        with mock.patch.object(m,'_read',return_value={}),mock.patch.object(m,'run',return_value={'status':'SYNTHETIC'}),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main(['--config','fake','--device','cpu']),0)
        event=json.loads(out.getvalue().splitlines()[0]);self.assertEqual(event['event'],'UNLABELED_REGISTRATION_SOURCE_PROCESS');self.assertTrue(event['pid']);self.assertTrue(event['cwd'])


if __name__=='__main__':unittest.main()
