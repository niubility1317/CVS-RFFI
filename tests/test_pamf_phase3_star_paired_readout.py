"""Fixed-budget synthetic paired readouts; no real IQ, weights or unknown TX."""
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


class StarPairedReadoutTests(unittest.TestCase):
    _fitted=None

    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_paired_readout.py').is_file(),
                        'Paired evidence readout is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_paired_readout')

    @staticmethod
    def data():
        rng=np.random.default_rng(19)
        kd=np.repeat(np.arange(15,dtype=np.int64),np.arange(15)%3+1)
        od=np.repeat(np.arange(15,dtype=np.int64),(np.arange(15)*2)%4+1)
        known=rng.normal(1.,.4,(len(kd),200)).astype(np.float32)
        oe=rng.normal(-1.,.4,(len(od),200)).astype(np.float32)
        known[:,100:]+=1.;oe[:,100:]-=1.
        return known,oe,kd,od

    def fitted(self):
        m=self.module()
        if type(self)._fitted is None:
            import torch
            initial=[];snapshots={};updates=[0]
            old_init,old_step=torch.optim.AdamW.__init__,torch.optim.AdamW.step
            def initialize(opt,parameters,*args,**kwargs):
                parameters=list(parameters);old_init(opt,parameters,*args,**kwargs)
                initial.append({k:np.asarray(v.detach().cpu().tolist(),np.float32) for k,v in zip(PARAMETERS,parameters)})
            def step(opt,*args,**kwargs):
                result=old_step(opt,*args,**kwargs);updates[0]+=1
                current=int(opt.state[opt.param_groups[0]['params'][0]]['step'].item())
                if current in (200,1000):
                    arm='p1_only' if len(initial)==1 else 'paired'
                    snapshots[arm,current]={k:np.asarray(v.detach().cpu().tolist(),np.float32)
                        for k,v in zip(PARAMETERS,opt.param_groups[0]['params'])}
                return result
            known,oe,kd,od=self.data();before=(known.copy(),oe.copy())
            # Tiny synthetic matrices avoid CPU thread-launch overhead. Restore
            # the test process setting; production keeps its configured threads.
            previous_threads=torch.get_num_threads();torch.set_num_threads(1)
            try:
                with patch.object(torch.optim.AdamW,'__init__',initialize),patch.object(torch.optim.AdamW,'step',step), \
                     patch.object(torch.Tensor,'numpy',side_effect=AssertionError('No NumPy ABI')):
                    state=m.fit_readouts(known,oe,kd,od)
            finally:torch.set_num_threads(previous_threads)
            np.testing.assert_array_equal(known,before[0]);np.testing.assert_array_equal(oe,before[1])
            type(self)._fitted=(state,initial,snapshots,updates[0])
        return copy.deepcopy(type(self)._fitted)

    def test_200_and_1000_are_post_update_snapshots_of_two_exact_1000_step_trajectories(self):
        m=self.module();state,initial,snapshots,updates=self.fitted()
        self.assertEqual(updates,2000);self.assertEqual(len(initial),2)
        for key in PARAMETERS:np.testing.assert_array_equal(initial[0][key],initial[1][key])
        names=('p1_mlp16_step200','paired_mlp16_step200','p1_mlp16_step1000','paired_mlp16_step1000')
        for name,(arm,step) in zip(names,(('p1_only',200),('paired',200),('p1_only',1000),('paired',1000))):
            for key in PARAMETERS:np.testing.assert_array_equal(state['models'][name][key],snapshots[arm,step][key])
            self.assertEqual(sum(v.size for v in state['models'][name].values()),3233)
        t=state['training'];self.assertEqual(t['total_optimizer_updates'],2000);self.assertEqual(t['steps_per_arm'],1000)
        self.assertEqual(t['checkpoint_steps'],[200,1000]);self.assertTrue(t['initial_parameters_equal'])
        self.assertEqual({k:len(v) for k,v in t['losses'].items()},{'p1_only':1000,'paired':1000})
        self.assertEqual(t['parameter_counts'],{name:3233 for name in names});json.dumps(t,allow_nan=False)

    def test_control_ignores_arbitrary_finite_P0_changes_while_paired_can_use_them(self):
        m=self.module();state,_,_,_=self.fitted();known,_,_,_=self.data()
        changed=known.copy();changed[:,100:]=np.random.default_rng(3).normal(-3.,4.,changed[:,100:].shape)
        before=copy.deepcopy(state);first=m.score_readouts(known,state);second=m.score_readouts(changed,state)
        np.testing.assert_array_equal(first[:,[0,2]],second[:,[0,2]])
        self.assertTrue(np.any(np.abs(first[:,[1,3]]-second[:,[1,3]])>1e-5))
        self.assertEqual((first.shape,first.dtype),((len(known),4),np.dtype('float64')))
        for arm in state['normalizers']:
            for key in state['normalizers'][arm]:np.testing.assert_array_equal(state['normalizers'][arm][key],before['normalizers'][arm][key])
        for arm in state['models']:
            for key in state['models'][arm]:np.testing.assert_array_equal(state['models'][arm][key],before['models'][arm][key])
        combined=m.score_compact(known,np.arange(len(known),dtype=np.float64),state)
        np.testing.assert_array_equal(combined[:,1:],first);np.testing.assert_array_equal(combined[:,0],np.arange(len(known)))

    def test_normalizers_and_first_loss_use_domain_equal_mass_and_fixed_role_priors(self):
        m=self.module();state,initial,_,_=self.fitted();known,oe,kd,od=self.data()
        wk=1/(15*np.bincount(kd)[kd]);wo=1/(15*np.bincount(od)[od]);weights=np.r_[.6*wk,.4*wo]
        raw=np.concatenate((known,oe)).astype(np.float64)
        mean=(raw*weights[:,None]).sum(0)/weights.sum()
        variance=(np.square(raw-mean)*weights[:,None]).sum(0)/weights.sum()
        scale=np.sqrt(variance);scale=np.where(scale<=1e-8,1.,scale)
        np.testing.assert_allclose(state['normalizers']['paired']['mean'],mean,rtol=0,atol=2e-14)
        np.testing.assert_allclose(state['normalizers']['paired']['scale'],scale,rtol=0,atol=2e-14)
        control=state['normalizers']['p1_only'];np.testing.assert_array_equal(control['mean'][100:],np.zeros(100))
        np.testing.assert_array_equal(control['scale'][100:],np.ones(100))
        for key in ('mean','scale'):np.testing.assert_array_equal(control[key][:100],state['normalizers']['paired'][key][:100])
        for i,arm in enumerate(('p1_only','paired')):
            value=raw.copy()
            if arm=='p1_only':value[:,100:]=0
            value=((value-state['normalizers'][arm]['mean'])/state['normalizers'][arm]['scale']).astype(np.float32)
            params=initial[i];hidden=np.maximum(value@params['weight1'].T+params['bias1'],0)
            g=(hidden@params['weight2'].T+params['bias2']).reshape(-1).astype(np.float64)
            loss=.5*np.dot(wk,np.logaddexp(0,-g[:len(known)]))+.5*np.dot(wo,np.logaddexp(0,g[len(known):]))
            self.assertAlmostEqual(state['training']['losses'][arm][0],loss,places=6)

    def test_pair_order_and_bad_shapes_domains_seed_or_extra_roles_rejected(self):
        m=self.module();known,oe,kd,od=self.data()
        combined=m.pair_descriptors(known[:,:100],known[:,100:])
        np.testing.assert_array_equal(combined,known);self.assertEqual(combined.dtype,np.float32)
        for defect in ('columns','nan','domain','seed'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                k=known.copy();d=kd.copy();seed=2026092701
                if defect=='columns':k=k[:,:199]
                elif defect=='nan':k[0,0]=np.nan
                elif defect=='domain':d[:]=0
                else:seed+=1
                m.fit_readouts(k,oe,d,od,model_seed=seed)
        self.assertNotIn('steps',inspect.signature(m.fit_readouts).parameters)
        with self.assertRaises(TypeError):m.fit_readouts(known,oe,kd,od,V=known)
        with self.assertRaises(TypeError):m.fit_readouts(known,oe,kd,od,steps=200)
        state,_,_,_=self.fitted();state['normalizers']['p1_only']['mean'][100]=1.
        with self.assertRaises(ValueError):m.score_readouts(known,state)

    def test_five_method_known_only_calibration_preserves_90_percent_and_two_pp_limits(self):
        m=self.module();rows=[(y,r,s,k) for y in range(6) for r in m.RECEIVERS for s in m.SCENES for k in range(10)]
        y=np.array([r[0] for r in rows]);rx=np.array([r[1] for r in rows]);scenes=np.array([r[2] for r in rows])
        low=np.array([r[3]==0 for r in rows]);closed=y.copy();closed[low]=(closed[low]+1)%6
        values=np.tile(np.where(low,0.,1.)[:,None],(1,5));ids=np.array(['v'+str(i) for i in range(len(y))])
        thresholds,report=m.calibrate_known(ids,y,closed,closed,values,rx,scenes)
        np.testing.assert_array_equal(thresholds,np.ones(5));self.assertEqual(report['method_names'],list(m.METHODS))
        self.assertEqual(report['schema'],'pamf_phase3_star_paired_prediction_v1');self.assertEqual(report['status'],'STAR_PAIRED_THRESHOLDS_FROZEN')
        self.assertEqual(report['primary_method'],'paired_mlp16_step1000');self.assertTrue(report['constraints_passed'])
        self.assertEqual(report['limits']['known_macro_frr'],.1);self.assertEqual(report['limits']['original_closed_minus_new_open'],.02)


if __name__=='__main__':unittest.main()
