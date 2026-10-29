"""Synthetic P1-only evaluation ordering and original-baseline regressions."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class JointOvaEvalTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_ova_eval')

    def fixture(self,root):
        m=self.module();rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        parent=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
            source_output='original',source_contract='contract',base_release='release')
        dura=dict(deployment_id='dura_v1',parameters=266951)
        def bundle(role):
            nr=n if role=='V' else 90;rx=np.asarray([rows[i%n][1] for i in range(nr)],np.int64);day=(np.arange(nr)%3+1).astype(np.int64)
            received=np.full((nr,2,256),999.,np.float32);restored=np.zeros((nr,2,256),np.float32)
            restored[:,0,0]=np.asarray([rows[i%n][0] for i in range(nr)]) if role=='V' else 7
            meta=dict(rx_i=rx,day_i=day,domain=np.searchsorted(m.RECEIVERS,rx)*3+day-1)
            if role=='V':meta.update(y=np.asarray([r[0] for r in rows],np.int64),scenario=np.asarray([r[2] for r in rows]))
            result=dict(manifest=dict(schema=m.CACHE_SCHEMA,status=m.CACHE_STATUS,run_id=m.CACHE_RUN,role=role,count=nr,
                iq_shape=[nr,2,256],paths=['P0','P1'],source_only=True,all_new_IQ_satellite=True,
                parent_F0_identity=parent,dura_identity=dura),ids=np.asarray([role+str(i) for i in range(nr)]),
                received=received,restored=restored,metadata=meta)
            if role=='V':result['baseline_closed']=np.column_stack((meta['y'],meta['y'])).astype(np.int64)
            return result
        roles={name:bundle(name) for name in ('V','novel')}
        native=dict(source_output='original',source_contract='contract',base_release='release')
        train=dict(run_id=m.RUN_ID,code_commit='a'*40,arm='joint',native_config=native,L_root='L',OE_root='OE',
            teacher_root='teacher',output_root='train',steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
        config=dict(run_id=m.RUN_ID,code_commit='a'*40,arm='joint',train_config=train,checkpoint='final.pt',cache_root='cache',
            output_root=str(root/'eval'),batch_size=128)
        model=torch.nn.Linear(1,1).eval().requires_grad_(False);aux=torch.nn.Linear(160,6).eval().requires_grad_(False)
        identity=dict(run_id=m.RUN_ID,arm='joint',row_id='joint-s2026092701',code_commit='a'*40,checkpoint='final.pt',
            updates=2220,cumulative_updates=13320,original_parent_identity=parent,model_parameters=232993,aux_parameters=966)
        return config,roles,model,aux,identity,parent

    def execute(self,m,config,roles,model,aux,identity,parent,*,wrong=False,mutate=False):
        events=[];reader=object();original=object()
        def load(root,role):events.append('load_'+role);return roles[role]
        def predict(r,a,values,device,batch,keep_unit=False):
            self.assertFalse(np.any(values[:,0,0]==999));events.append('original' if r is original else 'candidate')
            n=len(values);c=np.where(values[:,0,0]<6,values[:,0,0],0).astype(np.int64)
            if wrong and r is reader:c=(c+1)%6
            z=np.full((n,6),-2.,np.float32);z[np.arange(n),c]=2.
            g=np.full((n,6),-1.,np.float32);g[np.arange(n),c]=np.where(values[:,0,0]<6,1.,-1.)
            unit=np.zeros((n,160),np.float32);unit[:,0]=1.
            if mutate and r is reader:model.weight.add_(1.)
            return z,g if a is not None else None,unit if keep_unit else None
        def original_load(config,device):
            self.assertTrue((Path(config['output_root'])/'calibration/complete.json').is_file());events.append('load_original')
            return original,parent
        with patch.dict(m.COUNTS,{'V':180,'novel':90},clear=True),patch.object(m,'_backend',return_value=(model,reader,aux,identity)), \
             patch.object(m,'_smoke',side_effect=lambda *a:events.append('smoke') or dict(status='PASS')), \
             patch.object(m,'_load_role',side_effect=load),patch.object(m,'_close_role'),patch.object(m,'_predict',side_effect=predict), \
             patch.object(m,'_load_original',side_effect=original_load),patch.object(m,'_original_snapshot',return_value=None),patch.object(m,'_check_original'):
            result=m.run(config,'cpu')
        return result,events

    def test_candidate_and_tau_freeze_before_novel_original_only_novel_and_P1_only(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);config,roles,model,aux,identity,parent=self.fixture(root)
            done,events=self.execute(m,config,roles,model,aux,identity,parent)
            self.assertEqual(events,['smoke','load_V','candidate','load_original','load_novel','candidate','original'])
            self.assertTrue(done['all_thresholds_frozen_before_novel']);self.assertEqual(done['original_F0_novel_forward_rows'],90)
            self.assertEqual(done['real_candidate_P0_forward_rows'],0);self.assertEqual(done['source_OE_records_consumed'],0)
            with np.load(root/'eval/known_predictions.npz') as z:self.assertTrue(np.array_equal(z['baseline_closed'],roles['V']['baseline_closed'][:,1]))

    def test_original_F0_damage_infeasible_acceptall_still_predicts_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);done,events=self.execute(m,*v,wrong=True)
            cal=json.loads((root/'eval/calibration/complete.json').read_text())
            self.assertIn('load_novel',events)
            self.assertTrue(all(not r['no_rejection_feasible'] and not r['constraints_passed'] for r in cal['methods'].values()))
            with np.load(root/'eval/known_predictions.npz') as z:
                self.assertFalse(np.array_equal(z['closed'],z['baseline_closed']));self.assertTrue((z['scores']>=0).all())

    def test_private_config_wrong_arm_and_eval_OE_rejected_before_model_or_IQ(self):
        m=self.module()
        for mode in ('truth','arm','steps'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);config,*_=self.fixture(root)
                if mode=='truth':config['novel_truth']='forbidden'
                elif mode=='arm':config['train_config']['arm']='detach_control'
                else:config['train_config']['steps']=1
                with patch.object(m,'_backend',side_effect=AssertionError('backend')) as backend,self.assertRaises(ValueError):m.run(config,'cpu')
                self.assertEqual(backend.call_count,0)
        with self.assertRaises(ValueError):m._load_role('no-root','OE')

    def test_mutated_frozen_model_prevents_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with self.assertRaisesRegex(ValueError,'changed'):self.execute(m,*v,mutate=True)

    def test_aux_uses_unit_fused_and_safe_numpy_bridge_under_no_grad(self):
        m=self.module();aux=torch.nn.Linear(160,6,bias=False).eval().requires_grad_(False);aux.weight.fill_(1.)
        def extract(reader,x):
            raw=torch.ones((len(x),160),device=x.device)*3
            return {'fused160':raw},raw[:,:6]
        original_as_tensor=torch.as_tensor
        def unsafe(value,*a,**kw):
            if isinstance(value,np.ndarray):raise AssertionError('NumPy ABI path forbidden')
            return original_as_tensor(value,*a,**kw)
        with patch('experiments.pamf_phase3_open_world.source_open_export.extract_branches',side_effect=extract),patch.object(torch,'as_tensor',side_effect=unsafe):
            z,g,u=m._encode(object(),aux,np.ones((2,2,256),np.float32),'cpu',True)
        self.assertTrue(np.allclose(z,3.));self.assertTrue(np.allclose(g,np.sqrt(160),atol=1e-5))
        self.assertTrue(np.allclose(np.linalg.norm(u,axis=1),1.))

    def test_immutable_output_and_novel_private_metadata(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);config,roles,*_=v;(root/'eval').mkdir()
            with patch.object(m,'_backend',side_effect=AssertionError('backend')),self.assertRaises(FileExistsError):m.run(config,'cpu')
            roles['novel']['metadata']['y']=np.zeros(90,np.int64)
            with patch.dict(m.COUNTS,{'V':180,'novel':90},clear=True),self.assertRaises(ValueError):m._role(roles['novel'],'novel',v[-1],None)


if __name__=='__main__':unittest.main()
