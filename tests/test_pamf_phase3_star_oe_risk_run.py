"""Synthetic ordering and frozen-parent checks for OE risk calibration."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class OERiskRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_oe_risk_run')

    def fixture(self,root):
        m=self.module();p=root/'parent';p.mkdir();(p/'calibration').mkdir()
        fixed=m.PARENT_COMMIT;parent0=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100)
        train=dict(run_id=m.PARENT_RUN,code_commit=fixed,arm='joint',native_config={},L_root='L',OE_root=str(root/'OE'),teacher_root='teacher',
            output_root=str(root/'train'),steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
        ec=dict(run_id=m.PARENT_RUN,code_commit=fixed,arm='joint',train_config=train,checkpoint='final.pt',cache_root=str(root/'cache'),output_root=str(p),batch_size=128)
        identity=dict(run_id=m.PARENT_RUN,code_commit=fixed,arm='joint',checkpoint='final.pt',updates=2220,cumulative_updates=13320,
            original_parent_identity=parent0,model_parameters=232993,aux_parameters=966)
        dura=dict(deployment_id='dura_v1',parameters=266951)
        marker=dict(schema=m.parent.SCHEMA,status=m.parent.STATUS,run_id=m.PARENT_RUN,code_commit=fixed,arm='joint',model_identity=identity,
            method_names=list(m.parent.METHODS),primary_method='ova_max',primary_arm='joint',classification_path='P1',source_only=True,target_reads=0,
            source_novel_truth_reads=0,known_count=180,novel_count=90,model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True,
            all_thresholds_frozen_before_novel=True,cache_identities=dict(parent_F0_identity=parent0,dura_identity=dura))
        (p/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        tau=np.arange(18,dtype=np.float64).reshape(3,6)/10-1
        groups={k:dict(constraints_passed=True) for k in m.GROUP_NAMES}
        reports={name:dict(conditional_thresholds=tau[j].tolist(),groups=groups,constraints_passed=True) for j,name in enumerate(m.parent.METHODS)}
        cal=dict(schema=m.parent.SCHEMA,status=m.parent.CALIBRATION_STATUS,run_id=m.PARENT_RUN,code_commit=fixed,arm='joint',known_only=True,
            method_names=list(m.parent.METHODS),methods=reports,group_count=42,conditional_thresholds=tau.tolist())
        (p/'calibration/complete.json').write_text(json.dumps(cal),encoding='utf-8');np.savez(p/'thresholds.npz',conditional=tau,**{'global':np.zeros(3,np.float64)})
        data={}
        for role,n in (('known',180),('novel',90)):
            y=np.arange(n,dtype=np.int64)%6;rx=np.resize(np.asarray(m.RECEIVERS,np.int64),n);day=np.arange(n,dtype=np.int64)%3+1
            z=np.full((n,6),-2.,np.float32);z[np.arange(n),y]=2;g=np.full((n,6),-1.,np.float32);g[np.arange(n),y]=1
            raw,c=m.parent.scalars(z,g);d=dict(ids=np.asarray([role+str(i) for i in range(n)]),rx_i=rx,day_i=day,default_logits=z,aux_logits=g,closed=c,scores=raw-tau[:,c].T)
            if role=='known':d.update(y=y,scenario=np.resize(np.asarray(m.SCENES),n),baseline_closed=(y+1)%6)
            else:
                u=np.zeros((n,160),np.float32);u[:,0]=1;d.update(candidate_fused_unit=u,baseline_fused_unit=u.copy())
            np.savez(p/(role+'_predictions.npz'),**d);data[role]=d
        rx=np.repeat(np.asarray(m.RECEIVERS,np.int64),6);day=np.tile(np.repeat(np.arange(1,4,dtype=np.int64),2),5)
        oe=dict(ids=np.asarray(['OE'+str(i) for i in range(30)]),metadata=dict(rx_i=rx,day_i=day,domain=np.searchsorted(m.RECEIVERS,rx)*3+day-1),
            manifest=dict(role='OE',count=30,parent_F0_identity=parent0,dura_identity=dura),received=np.full((30,2,256),999,np.float32),restored=np.zeros((30,2,256),np.float32))
        config=dict(run_id=m.RUN_ID,code_commit='b'*40,arm='joint',parent_eval_config=ec,parent_prediction_root=str(p),OE_root=str(root/'OE'),output_root=str(root/'out'),batch_size=128)
        model=torch.nn.Linear(1,1).eval().requires_grad_(False);aux=torch.nn.Linear(160,6).eval().requires_grad_(False)
        return config,oe,data,tau,model,aux,identity

    def execute(self,m,v,*,smoke='PASS',mutate=False,fail=False,scientific=False):
        config,oe,data,tau,model,aux,identity=v;events=[];original=m._load_parent_predictions
        def load(role,*args):
            events.append(role)
            if role=='novel':
                out=Path(config['output_root']);self.assertTrue((out/'calibration/complete.json').is_file());self.assertTrue((out/'known_predictions.npz').is_file());self.assertTrue((out/'thresholds.npz').is_file())
            return original(role,*args)
        def predict(reader,a,values,device,batch):
            events.append('OE_forward');self.assertFalse(np.any(values==999));n=len(values);z=np.zeros((n,6),np.float32);z[:,0]=1;g=np.full((n,6),-.5,np.float32)
            if mutate:model.weight.add_(1)
            return z,g,None
        def calibrate(*args):
            events.append('calibrate');self.assertEqual(len(args),11);self.assertEqual(set(args[7]),set(oe['ids']));self.assertTrue(np.array_equal(args[3],data['known']['baseline_closed']))
            if fail:raise RuntimeError('calibration failed')
            t=np.full(6,-10. if scientific else .25,np.float64)
            return {'conditional_thresholds':t},dict(conditional_thresholds=t.tolist(),groups={k:dict(constraints_passed=not scientific) for k in m.GROUP_NAMES},constraints_passed=not scientific,novel_inputs_used=False)
        with patch.dict(m.COUNTS,dict(known=180,novel=90,OE=30),clear=True),patch.object(m.parent,'_backend',return_value=(model,object(),aux,identity)), \
             patch.object(m.parent,'_smoke',side_effect=lambda *a:events.append('smoke') or dict(status=smoke)),patch.object(m,'_load_OE',side_effect=lambda *a:events.append('OE') or oe), \
             patch.object(m,'_close_OE'),patch.object(m.parent,'_predict',side_effect=predict),patch.object(m,'_calibrate',side_effect=calibrate),patch.object(m,'_load_parent_predictions',side_effect=load):
            done=m.run(config,'cpu')
        return done,events

    def test_smoke_first_freeze_before_novel_exact_parent_control_and_same_identity(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v);out=Path(v[0]['output_root'])
            self.assertEqual(events,['smoke','OE','known','OE_forward','calibrate','novel']);self.assertEqual(done['F_forward_rows'],32);self.assertEqual(done['G_forward_rows'],0)
            with np.load(out/'thresholds.npz') as z:self.assertTrue(np.array_equal(z['conditional'][0],v[3][2]))
            for role in ('known','novel'):
                with np.load(out/(role+'_predictions.npz')) as z:
                    self.assertTrue(np.array_equal(z['closed'],v[2][role]['closed']));self.assertTrue(np.array_equal(z['default_logits'],v[2][role]['default_logits']))
                    self.assertTrue(np.array_equal(z['scores'][:,0],v[2][role]['scores'][:,2]));self.assertEqual(z['scores'].shape[1],2)
            cal=json.loads((out/'calibration/complete.json').read_text());self.assertTrue(cal['source_OE_membership_used']);self.assertFalse(cal['known_only'])
            with np.load(out/'OE_scores.npz') as z:self.assertEqual(set(z.files),{'ids','rx_i','day_i','domain','closed','score','default_logits','aux_logits'})

    def test_failed_smoke_prevents_actual_source_and_partial_complete(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with self.assertRaises(ValueError):self.execute(m,v,smoke='FAIL')
            self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_private_or_mismatched_parent_config_rejected_before_backend(self):
        m=self.module()
        for kind in ('truth','parent_code','OE_path'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                c,*_=self.fixture(Path(td))
                if kind=='truth':c['novel_truth']='forbidden'
                elif kind=='parent_code':c['parent_eval_config']['code_commit']='c'*40
                else:c['OE_root']='other'
                with patch.object(m.parent,'_backend',side_effect=AssertionError('backend')),self.assertRaises(ValueError):m.run(c,'cpu')

    def test_parent_common_tau_or_identity_mismatch_not_recalibrated(self):
        m=self.module()
        for kind in ('tau','closed','parent_code'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));p=Path(v[0]['parent_prediction_root'])
                if kind=='tau':
                    cal=json.loads((p/'calibration/complete.json').read_text());cal['methods']['ova_max']['conditional_thresholds'][0]+=1;(p/'calibration/complete.json').write_text(json.dumps(cal))
                elif kind=='closed':
                    d=v[2]['known'];d['closed']=(d['closed']+1)%6;np.savez(p/'known_predictions.npz',**d)
                else:
                    d=json.loads((p/'complete.json').read_text());d['code_commit']='c'*40;(p/'complete.json').write_text(json.dumps(d))
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_frozen_model_mutation_and_calibration_failure_stop_before_novel(self):
        m=self.module()
        for kind in ('mutation','calibration'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td))
                with self.assertRaises((ValueError,RuntimeError)):self.execute(m,v,mutate=kind=='mutation',fail=kind=='calibration')
                self.assertFalse((Path(v[0]['output_root'])/'novel_predictions.npz').exists());self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_scientific_fallback_completes_and_output_is_immutable(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v,scientific=True);self.assertIn('novel',events)
            self.assertEqual(done['status'],m.STATUS);self.assertFalse(json.loads((Path(v[0]['output_root'])/'calibration/complete.json').read_text())['methods']['oe_risk_budget']['constraints_passed'])
            with patch.object(m.parent,'_backend',side_effect=AssertionError('backend')),self.assertRaises(FileExistsError):m.run(v[0],'cpu')

    def test_OE_private_labels_or_source_identity_mismatch_rejected(self):
        m=self.module()
        for kind in ('label','parent','role'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td))
                if kind=='label':v[1]['metadata']['tx']=np.zeros(30,np.int64)
                elif kind=='parent':v[1]['manifest']['parent_F0_identity']={}
                else:v[1]['manifest']['role']='novel'
                with self.assertRaises(ValueError):self.execute(m,v)


if __name__=='__main__':unittest.main()
