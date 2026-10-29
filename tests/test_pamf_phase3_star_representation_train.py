"""Synthetic permission/budget/checkpoint tests; no source IQ or weights."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FakeReader(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();self.core.id_backbone=nn.Module()
        b=self.core.id_backbone;b.sinc=nn.Module()
        b.sinc.low_coordinate=nn.Parameter(torch.ones(24,1)*.01);b.sinc.band_coordinate=nn.Parameter(torch.ones(24,1)*.02)
        b.cls_head=nn.Module();b.cls_head.weight=nn.Parameter(torch.randn(6,160)*.1)
        self.linear=nn.Linear(512,160);self.bn=nn.BatchNorm1d(160);self.dropout=nn.Dropout(.7)
    def features(self,x):
        q=self.core.id_backbone.sinc
        z=self.linear(x.flatten(1))+(q.low_coordinate.sum()+q.band_coordinate.sum())*x[:,0].mean(1)[:,None]
        return self.dropout(self.bn(z))
    def classify_features(self,z):return 30*F.linear(F.normalize(z,dim=1),F.normalize(self.core.id_backbone.cls_head.weight,dim=1))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=FakeReader()
        self._phase3_parent_args={'weight_decay':.0001};self._phase3_parent_identity={'parent_updates':11100,'source_run_id':'F0'}


class StarRepresentationTrainTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_representation_train')
    def config(self,root,arm='anchor'):
        from experiments.pamf_phase3_open_world import source_open_embedding_train as native
        run='20261029-phase3-star-representation-s2026092701-r01';out=str(root/arm)
        n=dict(run_id=run,row_id=arm+'-s2026092701',code_commit='a'*40,output_root=out,model_seed=2026092701,
            base_release='original-release',frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},
            source_output='original-F0/source',source_contract='contract.json',source_cache='legacy-cache-MUST-NOT-OPEN',
            registered_classes=native.CLASSES.copy(),batch_size=128)
        return dict(run_id=run,code_commit='a'*40,arm=arm,native_config=n,L_root=str(root/'cache/L'),
            OE_root=None if arm=='anchor' else str(root/'cache/OE'),output_root=out,steps=2220,lr=.0001,
            L_seed=2026092701,OE_seed=2026102907,view_seed=2026102908)
    def role(self,role):
        n=30;rng=np.random.default_rng(5 if role=='L' else 6)
        data=dict(manifest=dict(role=role,count=n,source_only=True,path='P1'),ids=np.array([role+str(i) for i in range(n)]),
            restored=rng.normal(size=(n,2,256)).astype(np.float32),received=object(),
            metadata=dict(rx_i=np.tile([1,3,4,6,8],6).astype(np.int64),day_i=np.tile([1,2,3],10).astype(np.int64),domain=np.tile(np.arange(15),2).astype(np.int64)))
        # Metadata domain must be the actual Main namespace formula.
        data['metadata']['rx_i']=np.array([1,3,4,6,8],np.int64)[data['metadata']['domain']//3]
        data['metadata']['day_i']=data['metadata']['domain']%3+1
        data['manifest'].update(restored_input_path='P1',teacher_input_path='P1' if role=='L' else None,
            parent_F0_identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,
                model_seed=2026092701,source_output='original-F0/source'),
            dura_identity=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940))
        if role=='L':data['metadata']['y']=(np.arange(n)%6).astype(np.int64);data['teacher_logits']=rng.normal(size=(n,6)).astype(np.float32)
        return data
    def build_optimizer(self,model,params,**kw):
        opt=torch.optim.AdamW(params,**kw);opt._sinc_unit_steps=0;opt._sinc_physical_clip_calls=0
        opt.register_step_post_hook(lambda o,args,kwargs:setattr(o,'_sinc_unit_steps',o._sinc_unit_steps+1))
        return opt
    def clip(self,opt,params):
        opt._sinc_physical_clip_calls+=1
        return nn.utils.clip_grad_norm_(params,5.)

    def test_config_strict_permissions_native_mapping_and_original_initialization(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));self.assertIs(m.validate_config(c),c)
            mapped=m._native_config(c)
            self.assertEqual(mapped['arm'],'supervised_only');self.assertEqual(mapped['steps'],2220)
            self.assertNotIn('diverse_data_root',mapped)
            for key in ('V_root','novel_root','truth','target_root','checkpoint'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            for key in ('truth','V_root','checkpoint'):
                bad=copy.deepcopy(c);bad['native_config'][key]='x'
                with self.subTest(native=key),self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['OE_root']='forbidden'
            with self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['output_root']=bad['L_root'];bad['native_config']['output_root']=bad['L_root']
            with self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['output_root']=str(Path(bad['L_root'])/'new-training');bad['native_config']['output_root']=bad['output_root']
            with self.assertRaises(ValueError):m.validate_config(bad)
            for value in (222,2220.0,True):
                with self.subTest(steps=value),self.assertRaises(ValueError):m.validate_config(dict(c,steps=value))

    def test_synthetic_smoke_is_observed_G10_before_F_and_restores_grad_state(self):
        m=self.module();calls=[]
        def observe(raw,**kwargs):
            calls.append(kwargs);return np.asarray(raw,np.float32)*.8,{'scenario':'synthetic'}
        model=FakeModel().eval().requires_grad_(True);before={k:v.clone() for k,v in model.state_dict().items()}
        with patch.object(m,'_observe',side_effect=observe):result=m._smoke(model,'cpu')
        self.assertEqual(len(calls),2);self.assertEqual(result['input_path'],'P0_synthetic_G10')
        self.assertEqual(result['G_forwards'],0);self.assertGreater(result['sinc_gradient_norm'],0)
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k])
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)

    def test_three_arms_role_permissions_eval_mode_budget_and_final_marker(self):
        m=self.module();L=self.role('L');OE=self.role('OE')
        for arm in ('anchor','oe','oe_vic'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
                root=Path(td);c=self.config(root,arm);c['steps']=2;events=[];model=FakeModel();before=model.id_backbone.encoder.bn.running_mean.clone()
                real_write=m._write
                def write(path,value):events.append(('write',Path(path).name));return real_write(path,value)
                with patch.object(m,'STEPS',2),patch.object(m,'COUNTS',{'L':30,'OE':30}),\
                     patch.object(m.native,'validate_config',side_effect=lambda c:c),\
                     patch.object(m.native,'make_trainable_model',return_value=model) as build,\
                     patch.object(m,'_smoke',side_effect=lambda *a:events.append(('smoke',)) or {'status':'PASS'}),\
                     patch.object(m,'load_L',side_effect=lambda *a:events.append(('L',)) or L),\
                     patch.object(m,'load_OE',side_effect=lambda *a:events.append(('OE',)) or OE) as load_oe,\
                     patch.object(m,'close_role'),patch.object(m,'_optimizer',side_effect=self.build_optimizer),\
                     patch.object(m,'_clip',side_effect=self.clip),patch.object(m,'_write',side_effect=write):
                    result=m.run(c,'cpu')
                self.assertEqual(events[0],('smoke',));self.assertEqual(load_oe.call_count,int(arm!='anchor'))
                self.assertEqual(result['updates'],2);self.assertEqual(result['L_forward_samples'],512)
                self.assertEqual(result['OE_forward_samples'],0 if arm=='anchor' else 512)
                self.assertEqual(result['vicreg_evaluations'],2 if arm=='oe_vic' else 0)
                self.assertEqual(events[-1],('write','complete.json'));self.assertFalse(model.training)
                torch.testing.assert_close(model.id_backbone.encoder.bn.running_mean,before)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                cp=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
                self.assertIn('optimizer',cp);self.assertEqual(cp['schema'],m.SCHEMA)
                with patch.object(m,'STEPS',2),patch.object(m.native,'validate_config',side_effect=lambda c:c),\
                     patch.object(m.native,'make_trainable_model',return_value=FakeModel()):
                    loaded=m.load_trained(c,root/arm/'final.pt','cpu')
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                self.assertFalse(loaded.training);self.assertEqual(loaded._star_representation_identity['arm'],arm)
                with patch.object(m,'STEPS',2),patch.object(m.native,'validate_config',side_effect=lambda c:c),self.assertRaises(FileExistsError):m.run(c,'cpu')

    def test_anchor_never_evaluates_OE_or_vic_and_OE_control_never_vic(self):
        m=self.module();L=self.role('L');OE=self.role('OE')
        for arm in ('anchor','oe'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
                c=self.config(Path(td),arm);model=FakeModel().eval();opt=self.build_optimizer(model,list(model.parameters()),lr=.0001)
                with patch.object(m.losses,'vicreg',side_effect=AssertionError('VIC must be off')),\
                     patch.object(m.losses,'uniform_oe',side_effect=AssertionError('OE must be off')) if arm=='anchor' else patch.object(m,'COUNTS',m.COUNTS):
                    trace=m._loop(model,L,None if arm=='anchor' else OE,c,'cpu',opt,self.clip,steps=2)
                self.assertEqual(len(trace),2);self.assertTrue(all(row['vicreg_total']==0 for row in trace))

    def test_failure_never_completes_and_load_rejects_old_E51_before_constructor(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));model=FakeModel()
            with patch.object(m.native,'make_trainable_model',return_value=model),patch.object(m,'_smoke',return_value={'status':'PASS'}),\
                 patch.object(m,'load_L',side_effect=ValueError('bad cache')),self.assertRaises(ValueError):m.run(c,'cpu')
            self.assertFalse((Path(c['output_root'])/'complete.json').exists())
            with patch('torch.load',return_value={'schema':'pamf_phase3_source_embedding_increment_v1'}),\
                 patch.object(m.native,'make_trainable_model',side_effect=AssertionError('old E51 may not initialize')),\
                 self.assertRaises(ValueError):m.load_trained(c,'oldE51.pt','cpu')

    def test_cache_P1_teacher_identity_and_unknown_label_boundary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td),'oe');good=self.role('OE')
            with patch.object(m,'COUNTS',{'L':30,'OE':30}):
                self.assertIs(m._role(good,'OE',c),good)
                for defect in ('y','path','teacher_parent'):
                    bad=copy.deepcopy(good)
                    if defect=='y':bad['metadata']['y']=np.zeros(30,np.int64)
                    elif defect=='path':bad['manifest']['restored_input_path']='P0'
                    else:bad['manifest']['parent_F0_identity']['epoch']=51
                    with self.subTest(defect=defect),self.assertRaises(ValueError):m._role(bad,'OE',c)

    def test_real_native_scratch_graph_raw_fused_not_unit_and_all_losses_differentiable(self):
        m=self.module();root=Path(__file__).resolve().parents[1]
        c=self.config(root,'oe_vic');cfg=m._native_config(c);cfg['base_release']=str(root)
        frozen=root/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        cfg['frozen_sources']={k:str(frozen/k) for k in ('F','AF','BASE','FOLD','G')}
        m.runner._activate(cfg)
        parent=dict(run_id='synthetic-parent',row_id='synthetic-parent',model_seed=2026092701,method_id='pamf_t1_e50',
            source_rxs=[1,3,4,6,8],source_days=[1,2,3],num_classes=6,view_comparison={'view_count':3},
            paper_method='synthetic graph only',repl_row={'group':'main','structure':'SHUFFLE_TIME','L_A':True,'L_B':True,'U':True})
        with patch('torch.load',side_effect=AssertionError('No real weight IO')):
            model=m.native._new_native(cfg,m.native._training_config(cfg,parent),'cpu').eval().requires_grad_(True)
        m.native._assert_trainable(model)
        x=torch.randn(4,2,256,generator=torch.Generator().manual_seed(71));v=m.losses.phase_gain_view(x,torch.Generator().manual_seed(72))
        z0,p0=m.forward(model,x);z1,p1=m.forward(model,v)
        self.assertFalse(torch.allclose(z0.norm(dim=1),torch.ones(4),rtol=1e-3,atol=1e-3))
        loss=m.losses.labeled(p0,p1,torch.arange(4),torch.zeros_like(p0))['total']
        loss=loss+.1*m.losses.uniform_oe(p0,p1)+.1*m.losses.vicreg(z0,z1)['total'];loss.backward()
        named=dict(model.named_parameters())
        for key in (m.native.SINC_PREFIX+'low_coordinate',m.native.SINC_PREFIX+'band_coordinate',m.native.HEAD_NAME,
                    'id_backbone.encoder.sequence_pool.output_projection.weight'):
            grad=named[key].grad
            self.assertIsNotNone(grad,key);self.assertTrue(bool(torch.isfinite(grad).all()),key);self.assertGreater(float(grad.norm()),0,key)


if __name__=='__main__':unittest.main()
