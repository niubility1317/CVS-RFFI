"""Synthetic class-conditional evidence and fixed source-only shared readers."""
import copy
import importlib
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
PARAMETERS=('weight1','bias1','weight2','bias2')


class StarOVAEvidenceTests(unittest.TestCase):
    _fitted=None

    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_ova_evidence.py').is_file(),
                        'Class-conditional evidence core is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_ova_evidence')

    @staticmethod
    def geometry_data():
        from experiments.pamf_phase3_open_world import source_open_readout as old
        rng=np.random.default_rng(31);geometry=dict(schema=old.GEOMETRY_SCHEMA,class_count=6,branches={})
        branches={};other={}
        for name,d in old.DIMENSIONS.items():
            branches[name]=rng.normal(size=(4,d)).astype(np.float32)
            other[name]=rng.normal(size=(4,d)).astype(np.float32)
            mean=np.zeros(d);scale=np.ones(d);prototypes=rng.normal(size=(6,d))
            if name=='reference29':
                branches[name][:,27]=0;other[name][:,27]=0;prototypes[:,27]=0
            prototypes/=np.linalg.norm(prototypes,axis=1,keepdims=True)
            geometry['branches'][name]=dict(mean=mean,scale=scale,prototypes=prototypes,
                                            whitener=np.diag(np.linspace(.7,1.3,d)))
        logits=rng.normal(size=(4,6)).astype(np.float32)
        second_logits=rng.normal(size=(4,6)).astype(np.float32)
        return branches,logits,geometry,other,second_logits,copy.deepcopy(geometry)

    @staticmethod
    def data():
        rng=np.random.default_rng(47)
        kd=np.repeat(np.arange(15,dtype=np.int64),np.arange(15)%3+1)
        od=np.repeat(np.arange(15,dtype=np.int64),(np.arange(15)*2)%4+1)
        known=rng.normal(.7,.4,(len(kd),6,106)).astype(np.float32)
        oe=rng.normal(-.6,.5,(len(od),6,106)).astype(np.float32)
        y=np.arange(len(kd),dtype=np.int64)%6;closed=(y+np.arange(len(y))%2)%6
        # Duplicate evidence with distinct source labels exercises canonical ties.
        known[1]=known[0]
        return known,oe,y,closed,kd,od

    def fitted(self):
        m=self.module()
        if type(self)._fitted is None:
            import torch
            initial=[];updates=[0];old_init=torch.optim.AdamW.__init__;old_step=torch.optim.AdamW.step
            def initialize(opt,parameters,*args,**kwargs):
                parameters=list(parameters);old_init(opt,parameters,*args,**kwargs)
                initial.append({k:np.asarray(v.detach().cpu().tolist(),np.float32)
                                for k,v in zip(PARAMETERS,parameters)})
            def step(opt,*args,**kwargs):
                result=old_step(opt,*args,**kwargs);updates[0]+=1;return result
            arrays=self.data();before=[v.copy() for v in arrays]
            previous=torch.get_num_threads();torch.set_num_threads(1)
            try:
                with patch.object(m,'STEPS',8),patch.object(torch.optim.AdamW,'__init__',initialize), \
                     patch.object(torch.optim.AdamW,'step',step), \
                     patch.object(torch.Tensor,'numpy',side_effect=AssertionError('No NumPy ABI')), \
                     patch.object(torch,'from_numpy',side_effect=AssertionError('No ndarray bridge')):
                    state=m.fit(*arrays)
            finally:torch.set_num_threads(previous)
            for actual,expected in zip(arrays,before):np.testing.assert_array_equal(actual,expected)
            type(self)._fitted=(state,initial,updates[0])
        return copy.deepcopy(type(self)._fitted)

    def test_descriptor_has_manual_class_evidence_and_P1_then_P0_order(self):
        m=self.module();args=self.geometry_data();before=copy.deepcopy(args);actual=m.descriptor(*args)
        self.assertEqual((actual.shape,actual.dtype),((4,6,106),np.dtype('float32')))
        expected=[]
        for branches,logits,geometry in (args[:3],args[3:]):
            columns=[]
            for name in branches:
                b=geometry['branches'][name];u=(branches[name].astype(np.float64)-b['mean'])/b['scale']
                u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
                cosine=np.clip(u@b['prototypes'].T,-1,1)
                distance=np.square((u[:,None,:]-b['prototypes'][None,:,:])@b['whitener'].T).sum(-1)
                parts=[]
                for values in (cosine,-distance):
                    other=np.stack([np.max(np.delete(values,c,axis=1),axis=1) for c in range(6)],axis=1)
                    parts.extend((values,other,values-other))
                columns.append(np.stack(parts,axis=2))
            z=logits.astype(np.float64);top=z.max(1);exp=np.exp(z-top[:,None]);den=exp.sum(1)
            other=np.stack([np.max(np.delete(z,c,axis=1),axis=1) for c in range(6)],axis=1)
            columns.append(np.stack((exp/den[:,None],z,other,z-other,
                                     np.repeat((top+np.log(den))[:,None],6,axis=1)),axis=2))
            expected.append(np.concatenate(columns,axis=2))
        np.testing.assert_allclose(actual,np.concatenate(expected,axis=2).astype(np.float32),rtol=2e-6,atol=2e-6)
        for index in (0,3):
            for name in args[index]:np.testing.assert_array_equal(args[index][name],before[index][name])
        np.testing.assert_array_equal(args[1],before[1]);np.testing.assert_array_equal(args[4],before[4])

    def test_class_columns_are_equivariant_for_descriptor_and_shared_score(self):
        m=self.module();args=self.geometry_data();first=m.descriptor(*args);perm=np.array([4,0,5,1,3,2])
        changed=copy.deepcopy(args)
        for offset in (0,3):
            changed=list(changed);changed[offset+1]=changed[offset+1][:,perm]
            for b in changed[offset+2]['branches'].values():b['prototypes']=b['prototypes'][perm]
        second=m.descriptor(*changed);np.testing.assert_array_equal(second,first[:,perm])
        state,_,_=self.fitted();before=copy.deepcopy(state);score=m.score(first,state)
        self.assertEqual((score.shape,score.dtype),((4,2,6),np.dtype('float64')))
        np.testing.assert_array_equal(m.score(second,state),score[:,:,perm])
        for key in ('mean','scale'):np.testing.assert_array_equal(state[key],before[key])
        for name in state['models']:
            for key in PARAMETERS:np.testing.assert_array_equal(state['models'][name][key],before['models'][name][key])

    def test_loss_selects_original_closed_control_and_true_excluding_hard_negative(self):
        m=self.module();import torch
        known=torch.tensor([[9.,8.,7.,6.,5.,4.]],requires_grad=True)
        oe=torch.tensor([[-2.,-1.,3.,0.,1.,2.]],requires_grad=True)
        y=torch.tensor([0]);closed=torch.tensor([3])
        k,o=m._loss_vectors(known,oe,y,closed,'membership')
        self.assertAlmostEqual(k.item(),np.logaddexp(0,-6.),places=7)
        self.assertAlmostEqual(o.item(),np.logaddexp(0,3.),places=6)
        (k.sum()+o.sum()).backward()
        self.assertEqual(np.flatnonzero(known.grad.tolist()[0]).tolist(),[3])
        self.assertEqual(np.flatnonzero(oe.grad.tolist()[0]).tolist(),[2])
        known.grad=None;oe.grad=None;k,o=m._loss_vectors(known,oe,y,closed,'ova_hard')
        self.assertAlmostEqual(k.item(),.5*(np.logaddexp(0,-9.)+np.logaddexp(0,8.)),places=6)
        (k.sum()+o.sum()).backward()
        self.assertEqual(np.flatnonzero(known.grad.tolist()[0]).tolist(),[0,1])
        self.assertLess(known.grad[0,0].item(),0);self.assertGreater(known.grad[0,1].item(),0)
        self.assertEqual(np.flatnonzero(oe.grad.tolist()[0]).tolist(),[2])

    def test_shared_normalizer_domain_priors_initialization_budget_and_exact_risk(self):
        m=self.module();state,initial,updates=self.fitted();known,oe,y,closed,kd,od=self.data()
        self.assertEqual(m.STEPS,1000);self.assertEqual(m.MODEL_SEED,2026092701)
        self.assertEqual(updates,16);self.assertEqual(len(initial),2)
        for key in PARAMETERS:np.testing.assert_array_equal(initial[0][key],initial[1][key])
        wk=1/(15*np.bincount(kd)[kd]);wo=1/(15*np.bincount(od)[od]);weights=np.r_[.6*wk,.4*wo]
        raw=np.concatenate((known,oe)).astype(np.float64)
        mean=(raw.mean(1)*weights[:,None]).sum(0)/weights.sum()
        variance=(np.square(raw-mean).mean(1)*weights[:,None]).sum(0)/weights.sum()
        scale=np.sqrt(variance);scale=np.where(scale<=1e-8,1.,scale)
        np.testing.assert_allclose(state['mean'],mean,rtol=0,atol=2e-14)
        np.testing.assert_allclose(state['scale'],scale,rtol=0,atol=2e-14)
        x=((raw-state['mean'])/state['scale']).astype(np.float32)
        for i,name in enumerate(('membership','ova_hard')):
            model=initial[i];g=(np.maximum(x@model['weight1'].T+model['bias1'],0)@model['weight2'].T+model['bias2'])[:,:,0].astype(np.float64)
            if name=='membership':kl=np.logaddexp(0,-g[np.arange(len(y)),closed])
            else:
                negatives=g[:len(y)].copy();negatives[np.arange(len(y)),y]=-np.inf
                kl=.5*(np.logaddexp(0,-g[np.arange(len(y)),y])+np.logaddexp(0,negatives.max(1)))
            ol=np.logaddexp(0,g[len(y):].max(1));loss=.5*np.dot(wk,kl)+.5*np.dot(wo,ol)
            self.assertAlmostEqual(state['training']['losses'][name][0],loss,places=6)
        t=state['training'];self.assertEqual(t['steps_per_arm'],8);self.assertEqual(t['total_optimizer_updates'],16)
        self.assertEqual({k:len(v) for k,v in t['losses'].items()},{'membership':8,'ova_hard':8})
        self.assertEqual(t['parameter_counts'],{'membership':3457,'ova_hard':3457});self.assertTrue(t['initial_parameters_equal'])
        self.assertEqual(set(state),{'schema','descriptor_schema','descriptor_dim','class_count','model_names','mean','scale','models','training'})
        self.assertEqual(sum(v.size for model in state['models'].values() for v in model.values()),6914)
        json.dumps(t,allow_nan=False)

    def test_canonical_order_preserves_labels_and_exact_fit_under_row_permutation(self):
        m=self.module();import torch
        arrays=self.data();rng=np.random.default_rng(18);ki=rng.permutation(len(arrays[0]));oi=rng.permutation(len(arrays[1]))
        shuffled=(arrays[0][ki],arrays[1][oi],arrays[2][ki],arrays[3][ki],arrays[4][ki],arrays[5][oi])
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with patch.object(m,'STEPS',8):second=m.fit(*shuffled)
        finally:torch.set_num_threads(previous)
        first,_,_=self.fitted()
        for key in ('mean','scale'):np.testing.assert_array_equal(first[key],second[key])
        for name in first['models']:
            for key in PARAMETERS:np.testing.assert_array_equal(first['models'][name][key],second['models'][name][key])
        self.assertEqual(first['training'],second['training'])

    def test_bad_inputs_are_rejected_before_any_optimizer_or_hidden_label_argument(self):
        m=self.module();import torch
        for defect in ('shape','dtype','nan','y','closed','domain','seed'):
            with self.subTest(defect=defect),patch.object(torch.optim,'AdamW',side_effect=AssertionError('Invalid data entered fit')):
                a=list(self.data());seed=m.MODEL_SEED
                if defect=='shape':a[0]=a[0][:,:,:105]
                elif defect=='dtype':a[0]=a[0].astype(np.float64)
                elif defect=='nan':a[1][0,0,0]=np.nan
                elif defect=='y':a[2][0]=6
                elif defect=='closed':a[3][0]=-1
                elif defect=='domain':a[4][:]=0
                else:seed+=1
                with self.assertRaises(ValueError):m.fit(*a,model_seed=seed)
        signature=inspect.signature(m.fit)
        self.assertEqual(tuple(signature.parameters),('known_desc','oe_desc','known_y','known_closed','known_domains','oe_domains','model_seed'))
        self.assertEqual(tuple(inspect.signature(m.score).parameters),('desc','state'))
        with self.assertRaises(TypeError):m.fit(*self.data(),oe_y=np.zeros(1))
        with self.assertRaises(TypeError):m.fit(*self.data(),V=np.zeros(1))
        args=list(self.geometry_data());args[3]['reference29'][0,27]=1.
        with self.assertRaises(ValueError):m.descriptor(*args)

    def test_frozen_state_bad_shapes_and_nonfinite_forward_are_rejected(self):
        m=self.module();state,_,_=self.fitted();known=self.data()[0]
        for defect in ('scale','weight','extra','score_nan'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                s=copy.deepcopy(state);x=known.copy()
                if defect=='scale':s['scale'][0]=0.
                elif defect=='weight':s['models']['membership']['weight1']=np.zeros((6,32,106),np.float32)
                elif defect=='extra':s['source_rows']=known
                else:x[0,0,0]=np.nan
                m.score(x,s)


if __name__=='__main__':unittest.main()
