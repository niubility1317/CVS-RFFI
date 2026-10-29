"""Synthetic no-IQ orchestration tests for common-initialized risk budgets."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class CommonSeededRiskRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_common_seeded_risk_run')

    def fixture(self,root):
        # Reuse only a synthetic fixture, never another test or real artifact.
        oldtests=importlib.import_module('test_pamf_phase3_star_oe_risk_run')
        oldconfig,bundle,data,oldtau,model,aux,identity=oldtests.OERiskRunTests().fixture(root)
        m=self.module();p=root/'OE_parent';p.mkdir();(p/'calibration').mkdir();common=oldtau[2];zero=common+.1
        parent_marker=json.loads((Path(oldconfig['parent_prediction_root'])/'complete.json').read_text())
        marker=dict(schema=m.previous.SCHEMA,status=m.previous.STATUS,run_id=m.OE_PARENT_RUN,code_commit=m.OE_PARENT_COMMIT,arm='joint',
            method_names=list(m.previous.METHODS),source_only=True,target_reads=0,source_novel_truth_reads=0,known_count=180,novel_count=90,OE_count=30,
            parent_run_id=m.PARENT_RUN,parent_code_commit=m.PARENT_COMMIT,parent_prediction_root=oldconfig['parent_prediction_root'],
            parent_model_identity=identity,cache_identities=parent_marker['cache_identities'],model_parameters_buffers_unchanged=True,
            aux_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True,neural_updates=0,classification_path='P1')
        (p/'complete.json').write_text(json.dumps(marker),encoding='utf-8');np.savez(p/'thresholds.npz',conditional=np.stack((common,zero)))
        groups={k:dict(constraints_passed=True) for k in m.GROUP_NAMES}
        reports={name:dict(conditional_thresholds=tau.tolist(),constraints_passed=True,groups=groups) for name,tau in zip(m.previous.METHODS,(common,zero))}
        cal=dict(schema=m.previous.SCHEMA,status=m.previous.CALIBRATION_STATUS,run_id=m.OE_PARENT_RUN,code_commit=m.OE_PARENT_COMMIT,arm='joint',
            method_names=list(m.previous.METHODS),conditional_thresholds=np.stack((common,zero)).tolist(),methods=reports,group_count=42,
            source_only=True,target_reads=0,source_V_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,novel_inputs_used=False)
        (p/'calibration/complete.json').write_text(json.dumps(cal),encoding='utf-8')
        n=len(bundle['ids']);c=np.arange(n,dtype=np.int64)%6;z=np.full((n,6),-1.,np.float32);z[np.arange(n),c]=2;g=np.full((n,6),-.7,np.float32)
        oe=dict(ids=bundle['ids'],**bundle['metadata'],closed=c,score=g.astype(np.float64).max(1),default_logits=z,aux_logits=g)
        np.savez(p/'OE_scores.npz',**oe)
        config={k:v for k,v in oldconfig.items() if k!='OE_root'};config.update(run_id=m.RUN_ID,OE_prediction_root=str(p))
        return config,data,oldtau,zero,model,aux,identity,oe

    def execute(self,m,v,*,smoke='PASS',bad_monotone=False,scientific=False,mutate=False,fail=False):
        config,data,oldtau,zero,model,aux,identity,oe=v;events=[];old_loader=m._load_parent_predictions;oe_loader=m._load_OE_predictions
        def load(role,*args):
            events.append(role)
            if role=='novel':
                out=Path(config['output_root']);self.assertTrue(all((out/f).is_file() for f in ('thresholds.npz','calibration/complete.json','known_predictions.npz')))
            return old_loader(role,*args)
        def calibrate(*args,initial_thresholds):
            events.append('calibrate');self.assertEqual(len(args),11);self.assertTrue(np.array_equal(initial_thresholds,oldtau[2]));self.assertTrue(np.array_equal(args[3],data['known']['baseline_closed']))
            if fail:raise RuntimeError('fit failed')
            if mutate:aux.weight.add_(1)
            tau=initial_thresholds+(-.01 if bad_monotone else 0. if scientific else .2)
            return dict(conditional_thresholds=tau.copy()),dict(conditional_thresholds=tau.tolist(),constraints_passed=not scientific,
                groups={k:dict(constraints_passed=not scientific) for k in m.GROUP_NAMES},initialization='provided',initial_thresholds=initial_thresholds.tolist())
        with patch.dict(m.COUNTS,dict(known=180,novel=90,OE=30),clear=True),patch.object(m.parent,'_backend',return_value=(model,object(),aux,identity)), \
             patch.object(m.parent,'_smoke',side_effect=lambda *a:events.append('smoke') or dict(status=smoke)), \
             patch.object(m.parent,'_predict',side_effect=AssertionError('No real model forward permitted')),patch.object(m.previous,'_load_OE',side_effect=AssertionError('No IQ cache permitted')), \
             patch.object(m,'_load_parent_predictions',side_effect=load),patch.object(m,'_load_OE_predictions',side_effect=lambda *a:events.append('OE_public') or oe_loader(*a)),patch.object(m,'_calibrate',side_effect=calibrate):
            done=m.run(config,'cpu')
        return done,events

    def test_only_synthetic_smoke_two_exact_controls_and_freeze_before_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v);out=Path(v[0]['output_root'])
            self.assertEqual(events,['smoke','OE_public','known','calibrate','novel']);self.assertEqual(done['F_forward_rows'],2);self.assertEqual(done['real_P1_forward_rows'],0)
            self.assertTrue(done['monotone_thresholds_from_common']);self.assertEqual(done['parent_model_identity'],v[6])
            with np.load(out/'thresholds.npz') as z:
                self.assertTrue(np.array_equal(z['conditional'][0],v[2][2]));self.assertTrue(np.array_equal(z['conditional'][1],v[3]));self.assertTrue((z['conditional'][2]>=z['conditional'][0]).all())
            for role in ('known','novel'):
                with np.load(out/(role+'_predictions.npz')) as z:
                    self.assertTrue(np.array_equal(z['scores'][:,0],v[1][role]['scores'][:,2]));self.assertTrue(np.array_equal(z['closed'],v[1][role]['closed']))
                    raw=v[1][role]['aux_logits'].astype(float).max(1);self.assertTrue(np.array_equal(z['scores'][:,1],raw-v[3][z['closed']]))
                    self.assertTrue(np.all(~(z['scores'][:,2]>=0)|(z['scores'][:,0]>=0)))
            with np.load(out/'OE_scores.npz') as z:self.assertTrue(all(np.array_equal(z[k],v[7][k]) for k in z.files))

    def test_failed_smoke_does_not_consume_public_records(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with patch.object(m,'_load_OE_predictions',side_effect=AssertionError('OE records before smoke')),self.assertRaises(ValueError):self.execute(m,v,smoke='FAIL')
            self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_exact_config_and_parent_source_identity_required(self):
        m=self.module()
        for kind in ('truth','parent_code','parent_root'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));c=v[0]
                if kind=='truth':c['OE_root']='forbidden IQ'
                elif kind=='parent_code':c['parent_eval_config']['code_commit']='c'*40
                else:c['parent_eval_config']['output_root']='other'
                with patch.object(m.parent,'_backend',side_effect=AssertionError('backend')),self.assertRaises(ValueError):m.run(c,'cpu')

    def test_wrong_OE_parent_identity_code_common_or_private_field_rejected(self):
        m=self.module()
        for kind in ('identity','code','common','private'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));p=Path(v[0]['OE_prediction_root'])
                if kind in ('identity','code'):
                    x=json.loads((p/'complete.json').read_text());x['parent_model_identity' if kind=='identity' else 'code_commit']={} if kind=='identity' else 'c'*40;(p/'complete.json').write_text(json.dumps(x))
                elif kind=='common':
                    z=dict(np.load(p/'thresholds.npz'));z['conditional'][0,0]+=1;np.savez(p/'thresholds.npz',**z)
                else:
                    z=dict(np.load(p/'OE_scores.npz'));z['tx']=np.zeros(30,np.int64);np.savez(p/'OE_scores.npz',**z)
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_nonmonotone_thresholds_parameter_mutation_or_fit_error_cannot_complete(self):
        m=self.module()
        for kind in ('monotone','mutation','fit'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td))
                with self.assertRaises((ValueError,RuntimeError)):self.execute(m,v,bad_monotone=kind=='monotone',mutate=kind=='mutation',fail=kind=='fit')
                out=Path(v[0]['output_root']);self.assertFalse((out/'complete.json').exists());self.assertFalse((out/'novel_predictions.npz').exists())

    def test_scientific_invalid_initial_state_still_predicts_and_output_is_immutable(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v,scientific=True);self.assertIn('novel',events)
            cal=json.loads((Path(v[0]['output_root'])/'calibration/complete.json').read_text());self.assertFalse(cal['methods']['common_initialized_budget']['constraints_passed'])
            with patch.object(m.parent,'_backend',side_effect=AssertionError('backend')),self.assertRaises(FileExistsError):m.run(v[0],'cpu')


if __name__=='__main__':unittest.main()
