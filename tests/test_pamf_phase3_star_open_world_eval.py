"""Synthetic PAMF-OW evaluation; no real checkpoints, IQ or private labels."""
from contextlib import contextmanager, ExitStack
import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'experiments/pamf_phase3_open_world/star_open_world_eval.py').is_file(), 'PAMF-OW eval is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.star_open_world_eval')


def config(root,arm='M'):
    run='20261030-phase3-star-open-episodic-s2026092701-r01';code='a'*40
    native=dict(base_release='base',frozen_sources='sources',source_output='original',source_contract='contract',model_seed=2026092701)
    train=dict(run_id=run,code_commit=code,arm=arm,native_config=native,L_root=str(root/'cache/L'),OE_root=str(root/'cache/OE'),
        output_root=str(root/'train'/arm),steps=2220,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018)
    return dict(run_id=run,code_commit=code,arm=arm,train_config=train,checkpoint=train['output_root']+'/final.pt',
        cache_root=str(root/'cache'),output_root=str(root/'predict'/arm),batch_size=128,
        Q_public_root=str(root/'pool/evaluation/public'),Q_capsule_id='capsule',Q_split_id='split',dura_checkpoint='G.pt',dura_release='G-release')


@contextmanager
def run_fixture(root,mutate_novel=False,missing_L_class=False):
    e=module();c=config(root);events=[]
    parent=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
        **{k:c['train_config']['native_config'][k] for k in ('base_release','source_output','source_contract')})
    dura=dict(deployment_id='dura_v1',checkpoint='G.pt',release_root='G-release',parameters=266951)
    identity=dict(run_id=e.RUN_ID,code_commit=c['code_commit'],arm='M',checkpoint=c['checkpoint'],updates=2220,cumulative_updates=13320,
        identifier_parameters=284834,open_world_parameters=51841,training_container_parameters=952563,
        original_parent_identity=dict(parent_epoch=50,parent_updates=11100,model_seed=2026092701,
            **{k:c['train_config']['native_config'][k] for k in ('base_release','source_output','source_contract')}),
        parent_F0_identity=parent,identity_head_replaced=False,input_path='cached_P1_only')
    def frozen():return torch.nn.Linear(1,1).eval().requires_grad_(False)
    backend=dict(model=frozen(),original=frozen(),dura=(frozen(),None),identity=identity,parent=parent,dura_identity=dura)
    rows=[(y,r,s) for y in range(6) for r in e.RECEIVERS for s in e.SCENES for _ in range(2)]
    vy=np.array([v[0] for v in rows],np.int64);vrx=np.array([v[1] for v in rows],np.int64);vs=np.array([v[2] for v in rows])
    labels=dict(L=np.arange(12,dtype=np.int64)%6,V=vy,novel=6+np.arange(18,dtype=np.int64)%3)
    if missing_L_class:labels['L'][labels['L']==5]=4
    def values(y):
        a=np.zeros((len(y),2,256),np.float32);a[:,0,0]=y;return a
    bundles={}
    for role,y in labels.items():
        n=len(y);rx=vrx if role=='V' else np.full(n,1,np.int64);day=np.ones(n,np.int64)
        meta=dict(rx_i=rx,day_i=day,domain=np.searchsorted(e.RECEIVERS,rx)*3+day-1)
        if role in ('L','V'):meta['y']=y
        if role=='V':meta['scenario']=vs
        bundle=dict(manifest=dict(parent_F0_identity=parent,dura_identity=dura,role=role,count=n),
                    ids=np.array([role+'-'+str(i) for i in range(n)]),metadata=meta,restored=values(y))
        if role=='V':bundle['baseline_closed']=np.column_stack((y,y)).astype(np.int64)
        bundles[role]=bundle
    qy=np.repeat(np.arange(9,dtype=np.int64),6);qs=np.tile(np.asarray(e.SCENES),9)
    qids=np.asarray(['Q-'+str(i) for i in range(len(qy))]);qvalues=values(qy)
    def load(cache,role):
        assert role in ('L','V','novel');events.append(role)
        if role=='V':assert (Path(c['output_root'])/'prototypes.npy').is_file()
        if role=='novel':assert (Path(c['output_root'])/'calibration/complete.json').is_file()
        return bundles[role]
    def encode(model,iq,device,prototypes=None):
        y=iq[:,0,0].astype(np.int64);closed=np.minimum(y,5);z=np.full((len(y),6),-2.,np.float32)
        z[np.arange(len(y)),closed]=2.;u=np.eye(64,dtype=np.float32)[closed]
        relation=None if prototypes is None else np.full((len(y),6),2.,np.float32)
        if mutate_novel and len(events) and events[-1]=='novel':
            with torch.no_grad():next(backend['model'].parameters()).add_(1.)
        return z,u,relation
    def opening(spec,role):
        assert role=='Q';assert spec['Q_expected_count']==len(qy)
        assert (Path(c['output_root'])/'calibration/complete.json').is_file()
        events.append('Q');return {},qids,qvalues
    def probe(b,device):events.append('smoke');return dict(status='PASS',synthetic_G10_rows=2,real_data_reads=0)
    with ExitStack() as stack:
        for name,replacement in (('_train_validate',lambda t:t),('_backend',lambda c,d:backend),('_probe',probe),
            ('_load_role',load),('_close_role',lambda x:None),('_encode',encode),
            ('_restored',lambda x,g,d:np.array(x,copy=True)),('_baseline',lambda r,x,d:np.minimum(x[:,0,0],5).astype(np.int64))):
            stack.enter_context(patch.object(e,name,replacement))
        stack.enter_context(patch.object(e.public,'_open',opening))
        stack.enter_context(patch.object(e,'COUNTS',{r:len(y) for r,y in labels.items()}))
        stack.enter_context(patch.object(e,'Q_COUNT',len(qy)))
        yield e,c,backend,bundles,events,dict(ids=qids,y=qy,receiver=np.full(len(qy),26,np.int64),scenario=qs)


class StarOpenWorldEvalTests(unittest.TestCase):
    def test_exact_config_rejects_truth_U_K_bad_arm_and_input_overwrite_before_backend(self):
        e=module()
        with patch.object(e,'_train_validate',lambda t:t):
            c=config(Path('synthetic'))
            self.assertEqual(e.validate_config(c),c)
            for field in ('truth','U_public_root','unknown_count','K'):
                with self.subTest(field=field),self.assertRaises(ValueError):e.validate_config(dict(c,**{field:'forbidden'}))
            for key,value in (('arm','best'),('batch_size',64),('output_root',c['cache_root']+'/bad'),('checkpoint','other.pt')):
                with self.subTest(key=key),self.assertRaises(ValueError):e.validate_config(dict(c,**{key:value}))

    def test_L_prototypes_are_six_unit_means_not_V_fit_and_require_all_classes(self):
        e=module();x=np.eye(64,dtype=np.float32)[np.arange(18)%6];y=np.arange(18,dtype=np.int64)%6
        p=e._prototypes(x,y);np.testing.assert_array_equal(p,np.eye(64,dtype=np.float32)[:6])
        for defect in ('class','zero','nan'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                a=x.copy();b=y.copy()
                if defect=='class':b[b==5]=4
                elif defect=='zero':a[b==5]=0
                else:a[0,0]=np.nan
                e._prototypes(a,b)

    def test_complete_predicts_novel_and_Q_even_when_novel_accepts_all(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp,run_fixture(Path(temp)) as (e,c,b,roles,events,truth):
            done=e.evaluate(c,'cpu')
            self.assertEqual(e._identity(done['model_identity'],c),b['parent'])
            self.assertEqual(events,['smoke','L','V','novel','Q'])
            self.assertEqual(done['status'],e.STATUS);self.assertEqual(done['primary_arm'],'M')
            self.assertEqual(done['target_U_reads'],0);self.assertFalse(done['target_Q_fit'])
            self.assertFalse(done['source_unknown_success_target_entry_gate'])
            output=Path(c['output_root'])
            with np.load(output/'novel_predictions.npz',allow_pickle=False) as z:
                self.assertEqual(set(z.files),e.NOVEL_FIELDS);self.assertTrue(np.all(z['scores']>=0))
                self.assertFalse({'y','tx','scenario'}&set(z.files))
            with np.load(output/'target_predictions.npz',allow_pickle=False) as z:
                self.assertEqual(set(z.files),e.TARGET_FIELDS);self.assertFalse({'y','tx','scenario','K'}&set(z.files))
            self.assertTrue(done['prototypes_frozen_before_V']);self.assertTrue(done['thresholds_frozen_before_novel_and_target'])
            with self.assertRaises(FileExistsError):e.evaluate(c,'cpu')

    def test_frozen_model_change_blocks_Q_and_missing_L_class_blocks_V(self):
        import tempfile
        for defect in ('mutation','coverage'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temp:
                with run_fixture(Path(temp),mutate_novel=defect=='mutation',missing_L_class=defect=='coverage') as (e,c,b,roles,events,t):
                    with self.assertRaises(ValueError):e.evaluate(c,'cpu')
                    self.assertNotIn('Q',events)
                    if defect=='coverage':self.assertNotIn('V',events)
                    self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_encode_uses_actual_open_model_APIs_no_grad_and_list_bridge_then_clear(self):
        e=module();from experiments.pamf_phase3_open_world import star_open_world_model as ow
        model=torch.nn.Module();model.add_module('open_world',ow.OpenParts());model.eval().requires_grad_(False)
        def forward(m,x):
            self.assertFalse(torch.is_grad_enabled())
            return dict(open64=torch.nn.functional.normalize(torch.arange(1,65).float().repeat(len(x),1),dim=1),
                        logits=torch.ones(len(x),6),raw_fused160=torch.ones(len(x),160),baseline_fused160=torch.ones(len(x),160))
        with patch.object(ow,'forward',forward),patch.object(torch.Tensor,'numpy',side_effect=AssertionError('No NumPy ABI')):
            z,u,r=e._encode(model,np.ones((2,2,256),np.float32),'cpu',np.eye(64,dtype=np.float32)[:6])
        self.assertEqual((z.shape,u.shape,r.shape),((2,6),(2,64),(2,6)))
        self.assertTrue(all(v.dtype==np.float32 and np.isfinite(v).all() for v in (z,u,r)))
        self.assertIsNone(model._open_world_last);self.assertEqual(model._open_world_handles,[])


if __name__=='__main__':unittest.main()
