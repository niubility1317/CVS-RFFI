"""Synthetic source-evaluation contracts; no real checkpoint/cache is opened."""
import copy
import importlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


class RepresentationEvalTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_representation_eval')

    def config(self, root, arm='oe_vic'):
        m=self.module();train=dict(run_id=m.RUN_ID,code_commit='a'*40,arm=arm,native_config={},L_root='L',OE_root='OE',
            output_root='train',steps=2220,lr=.0001,L_seed=1,OE_seed=2,view_seed=3)
        return dict(run_id=m.RUN_ID,code_commit='a'*40,arm=arm,train_config=train,checkpoint='final.pt',
            cache_root=str(root/'cache'),output_root=str(root/'prediction'),batch_size=128)

    def inputs(self,m):
        rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        y=np.array([x[0] for x in rows],np.int64);n=len(rows)
        return (np.array(['k'+str(i) for i in range(n)]),y,np.column_stack((y,y)),np.ones((n,7),np.float64),
            np.array([x[1] for x in rows],np.int64),np.array([x[2] for x in rows]))

    def role(self,m,role):
        n=90 if role=='novel' else 180;ix=np.arange(n);y=(ix%6).astype(np.int64)
        manifest=dict(schema=m.CACHE_ROLE_SCHEMA,status=m.CACHE_ROLE_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            role=role,count=n,iq_shape=[n,2,256],paths=['P0','P1'],source_only=True,all_new_IQ_satellite=True,
            parent_F0_identity={'original':'F0'},dura_identity={'deployment_id':'dura_v1'})
        values=np.zeros((n,2,256),np.float32);values[:,0,0]=ix
        metadata=dict(rx_i=np.array([m.RECEIVERS[(i//6)%5] for i in ix],np.int64),
            day_i=(ix%3+1).astype(np.int64),domain=(ix%15).astype(np.int64))
        metadata['domain']=(np.searchsorted(m.RECEIVERS,metadata['rx_i'])*3+metadata['day_i']-1).astype(np.int64)
        if role!='novel':metadata['y']=y
        if role=='V':metadata['scenario']=np.array([m.SCENES[(i//30)%6] for i in ix])
        result=dict(manifest=manifest,ids=np.array([role+str(i) for i in ix]),received=values,restored=values.copy(),metadata=metadata)
        if role=='L':result['teacher_logits']=np.zeros((n,6),np.float32)
        if role=='V':result['baseline_closed']=np.column_stack((y,y))
        return result

    def test_configuration_rejects_target_truth_bad_arm_and_budget(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            good=self.config(Path(td));m.validate_config(good)
            for key,value in [('truth','hidden'),('target_root','hidden'),('arm','best_on_Q'),('batch_size',64)]:
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(good,**{key:value}))

    def test_original_baseline_damage_cannot_be_replaced_by_new_closed(self):
        m=self.module();ids,y,closed,scores,rx,scene=self.inputs(m);new=(y+1)%6
        threshold,report=m.calibrate_known(ids,y,new,y,scores,rx,scene)
        self.assertEqual(threshold.dtype,np.float64);self.assertFalse(report['constraints_passed'])
        for row in report['methods'].values():
            self.assertFalse(row['no_rejection_feasible']);self.assertTrue(row['zero_rejection_fallback'])
            self.assertEqual(row['threshold'],1.)
            self.assertAlmostEqual(row['groups']['overall']['original_closed_minus_new_open'],1.)

    def test_known_only_42groups_missing_class_and_intact_ties(self):
        m=self.module();ids,y,closed,scores,rx,scene=self.inputs(m)
        thresholds,report=m.calibrate_known(ids,y,y,y,scores,rx,scene)
        self.assertEqual(len(report['methods']['energy']['groups']),42)
        self.assertTrue(report['constraints_passed']);self.assertTrue(np.all(thresholds==1))
        keep=~((y==0)&(rx==1)&(scene==m.SCENES[0]))
        _,report=m.calibrate_known(ids[keep],y[keep],y[keep],y[keep],scores[keep],rx[keep],scene[keep])
        self.assertFalse(report['constraints_passed'])
        with self.assertRaises(ValueError):m.calibrate_known(ids,np.full(len(y),6),y,y,scores,rx,scene)

    def test_role_metadata_never_allows_novel_labels_or_OE_eval(self):
        m=self.module()
        with patch.object(m,'COUNTS',dict(L=180,V=180,novel=90)):
            m._validate_role(self.role(m,'L'),'L')
            for role,defect in [('novel','y'),('novel','scenario'),('V','baseline')]:
                with self.subTest(role=role,defect=defect),self.assertRaises(ValueError):
                    row=self.role(m,role)
                    if defect=='baseline':del row['baseline_closed']
                    else:row['metadata'][defect]=np.zeros(len(row['ids']),np.int64)
                    m._validate_role(row,role)
            with self.assertRaises(ValueError):m._validate_role(self.role(m,'OE'),'OE')

    def test_run_freezes_both_paths_and_thresholds_before_novel(self):
        m=self.module();events=[];reader=object();model=object()
        geometry={'token':np.array([1.])};metric={'token':np.array([2.])}
        identity=dict(run_id=m.RUN_ID,arm='oe_vic',code_commit='a'*40,checkpoint='final.pt',updates=2220,cumulative_updates=13320)
        def load(root,role):
            events.append(role)
            if role=='novel':
                out=Path(config['output_root'])
                for path in m.PATHS:self.assertTrue((out/'by_path'/path/'calibration/complete.json').is_file())
                self.assertFalse((out/'complete.json').exists())
            return self.role(m,role)
        def encode(_reader,x,device):
            n=len(x);logits=np.zeros((n,6),np.float32);logits[np.arange(n),x[:,0,0].astype(int)%6]=1
            return {'fused160':np.tile(np.eye(1,160,dtype=np.float32),(n,1))},logits
        def calibrate(*args):
            return np.zeros(7),dict(status=m.CALIBRATION_STATUS,schema=m.SCHEMA,methods={k:dict(threshold=0.,constraints_passed=True) for k in m.METHODS},constraints_passed=True)
        with tempfile.TemporaryDirectory() as td:
            config=self.config(Path(td))
            with patch.object(m,'COUNTS',dict(L=180,V=180,novel=90)),patch.object(m,'_backend',return_value=(model,reader,identity)),\
                 patch.object(m,'_snapshot_model',return_value={}),patch.object(m,'_check_model'),\
                 patch.object(m,'_load_role',side_effect=load),patch.object(m,'_close_role'),patch.object(m,'_encode',side_effect=encode),\
                 patch.object(m.geometry_core,'fit_geometry',return_value=geometry),patch.object(m.core,'fit_metric',return_value=metric),\
                 patch.object(m.core,'all_scores',side_effect=lambda b,l,g,t:np.ones((len(l),7))),\
                 patch.object(m,'_save_geometry',side_effect=lambda p,g:p.write_text('aggregate')),\
                 patch.object(m,'_save_metric',side_effect=lambda p,g:p.write_text('aggregate')),patch.object(m,'calibrate_known',side_effect=calibrate):
                done=m.run(config,'cpu')
            self.assertEqual(events,['L','V','novel']);self.assertTrue(done['all_thresholds_frozen_before_novel'])
            self.assertEqual(done['G_forward_rows'],0);self.assertEqual(done['source_OE_records_consumed'],0)
            with np.load(Path(config['output_root'])/'novel_predictions.npz') as rows:
                self.assertEqual(rows['scores'].shape,(90,2,7));self.assertEqual(rows['fused_unit'].shape,(90,2,160))

    def test_aggregate_or_model_mutation_stops_before_novel(self):
        m=self.module();state={'P0':dict(geometry={'a':np.array([1.])},metric={'b':np.array([2.])})}
        before=copy.deepcopy(state);state['P0']['metric']['b'][0]=3
        with self.assertRaises(ValueError):m._check_states(state,before)

    def test_tensor_bridge_uses_no_grad_and_never_inference_mode(self):
        import torch
        m=self.module()
        from experiments.pamf_phase3_open_world import source_open_export
        def extract(reader,x):
            self.assertFalse(torch.is_grad_enabled());self.assertFalse(torch.is_inference_mode_enabled())
            self.assertEqual(x.dtype,torch.float32);self.assertEqual(tuple(x.shape),(2,2,256))
            return {'fused160':torch.ones((2,160))},torch.ones((2,6))
        with patch.object(torch,'as_tensor',side_effect=RuntimeError('NumPy ABI unavailable')),\
             patch.object(torch,'inference_mode',side_effect=AssertionError('forbidden mode')),\
             patch.object(source_open_export,'extract_branches',side_effect=extract):
            branches,logits=m._encode(object(),np.zeros((2,2,256),np.float32),'cpu')
        self.assertEqual(branches['fused160'].dtype,np.float32);self.assertEqual(logits.shape,(2,6))


if __name__=='__main__':unittest.main()
