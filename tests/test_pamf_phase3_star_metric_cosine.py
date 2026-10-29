"""Synthetic cosine readout, freeze ordering, and CPU pipeline checks."""
import importlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


class MetricCosineTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_metric_cosine')

    def fixture(self,root):
        m=self.module();parent=root/'parent';(parent/'deployment').mkdir(parents=True);(parent/'calibration').mkdir()
        counts=dict(L=6300,OE=6134,V=180,novel=30);code='a'*40
        config=dict(run_id=m.RUN_ID,code_commit=code,parent_prediction_root=str(parent),output_root=str(root/'out'))
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        handles={r:dict(role=r,count=n,parent_F0_identity=identity,dura_identity=dura,source_only=True) for r,n in counts.items()}
        pm=dict(schema=m.PARENT_SCHEMA,status=m.PARENT_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,arm=m.PARENT_ARM,
            method_names=list(m.oldscore.METHODS),primary_arm=m.PARENT_ARM,primary_method=m.oldscore.PRIMARY,
            known_count=180,novel_count=30,source_L_count=6300,source_OE_count=6134,g_parameters=20704,g_training_steps=1000,
            g_training_L_exposures=120000,g_training_OE_exposures=128000,g_forward_rows=254510,F_forward_rows=2,synthetic_P0_forward_rows=2,
            G_forward_rows=0,real_IQ_values_read=0,source_L_metadata_labels_read=6300,neural_F_updates=0,G_updates=0,source_U_reads=0,
            target_reads=0,source_novel_truth_reads=0,source_only=True,classification_path='P1',default_head_replaced=False,
            model_parameters_buffers_unchanged=True,metric_parameters_buffers_frozen_for_V_novel=True,aggregate_frozen_before_V=True,
            all_thresholds_frozen_before_novel=True,source_free_deployment_state=True,per_record_source_rows_retained=0,parent_public_features_reused=True,
            model_identity=identity,dura_identity=dura,cache_identities=handles,cache_root='already-validated-cache-NOT-READ',
            teacher_run_id='20261029-phase3-star-joint-ova-s2026092701-r02',teacher_code_commit='5aa041122396d48c49b8cb3d5912649bdc3217e1',teacher_root='teacher-NOT-READ',
            fresh_closed_vs_cached_baseline_changed_count=0)
        (parent/'complete.json').write_text(json.dumps(pm),encoding='utf-8')
        model=m.oldscore.core.MetricNet().float();torch.save(model.state_dict(),parent/'deployment/model.pt')
        means=np.eye(6,160,dtype=np.float64)
        np.savez(parent/'deployment/metric.npz',class_means=means,background_mean=np.zeros(160,np.float64),within_whitener=np.eye(160),background_whitener=np.eye(160))
        tau=np.full((2,6),-1.,np.float64);np.savez(parent/'deployment/thresholds.npz',conditional=tau)
        deployment=dict(schema=m.oldscore.DEPLOYMENT_SCHEMA,status=m.oldscore.DEPLOYMENT_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,
            arm=m.PARENT_ARM,parameters=20704,source_free=True,per_record_source_rows_retained=0)
        (parent/'deployment/complete.json').write_text(json.dumps(deployment),encoding='utf-8')
        cal=dict(schema=m.PARENT_SCHEMA,status=m.oldscore.evaluation.CALIBRATION_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,arm=m.PARENT_ARM,
            method_names=list(m.oldscore.METHODS),conditional_thresholds=tau.tolist(),methods={name:dict(conditional_thresholds=tau[j].tolist()) for j,name in enumerate(m.oldscore.METHODS)},
            group_count=42,source_only=True,target_reads=0,known_only=True,unknown_inputs_used=False,source_V_neural_fit=False)
        (parent/'calibration/complete.json').write_text(json.dumps(cal),encoding='utf-8')
        rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        for role,n in (('known',180),('novel',30)):
            y=np.asarray([r[0] for r in rows],np.int64) if role=='known' else np.arange(n,dtype=np.int64)%6
            x=np.eye(6,160,dtype=np.float32)[y];logits=np.full((n,6),-2.,np.float32);logits[np.arange(n),y]=2
            data=dict(ids=np.asarray([role+str(i) for i in range(n)]),unit_features=x,metric_unit_features=x.copy(),default_logits=logits,closed=y,
                rx_i=np.asarray([r[1] for r in rows],np.int64) if role=='known' else np.resize(np.asarray(m.RECEIVERS,np.int64),n),
                day_i=np.arange(n,dtype=np.int64)%3+1,raw_scores=np.zeros((n,2),np.float64),scores=np.ones((n,2),np.float64))
            if role=='known':data.update(y=y.copy(),baseline_closed=y.copy(),scenario=np.asarray([r[2] for r in rows]))
            np.savez(parent/(role+'_predictions.npz'),**data)
        return config,counts

    def execute(self,m,v,smoke_fail=False):
        config,counts=v;events=[];original_np_load=np.load;original_smoke=m._reference_smoke
        class NoScores:
            def __init__(self,value):self.value=value;self.files=value.files
            def __enter__(self):return self
            def __exit__(self,*a):self.value.close()
            def __getitem__(self,key):
                if key in ('raw_scores','scores'):raise AssertionError('Producer must not consume parent score values')
                return self.value[key]
        def load(path,*a,**kw):
            path=Path(path);events.append('load:'+path.name)
            if 'score' in path.parts:raise AssertionError('Parent truth-derived score directory forbidden')
            if path.name.endswith('_predictions.npz'):
                self.assertIn('smoke',events)
                if path.name=='novel_predictions.npz':
                    out=Path(config['output_root']);self.assertTrue((out/'known_predictions.npz').is_file());self.assertTrue((out/'calibration/complete.json').is_file())
                    self.assertEqual({p.name for p in (out/'deployment').iterdir()},{'model.pt','prototypes.npz','thresholds.npz','complete.json'})
                return NoScores(original_np_load(path,*a,**kw))
            return original_np_load(path,*a,**kw)
        def smoke(net,marker,path):
            events.append('smoke')
            if smoke_fail:raise ValueError('smoke failure')
            return original_smoke(net,marker,path)
        with patch.dict(m.COUNTS,counts,clear=True),patch.object(m.np,'load',side_effect=load),patch.object(m,'_reference_smoke',side_effect=smoke), \
             patch.object(m.oldscore,'_prediction',side_effect=AssertionError('Full parent prediction validation opens novel early')):
            result=m.run(config)
        return result,events

    def test_hand_cosine_routes_rejection_by_original_closed_and_preserves_rows(self):
        m=self.module();means=np.eye(6,160,dtype=np.float64)*np.arange(1,7)[:,None];p=m.prototypes(means)
        x=np.eye(2,160,dtype=np.float32);closed=np.array([1,1],np.int64);before=p.copy()
        score=m.cosine_scores(x,closed,p);self.assertTrue(np.array_equal(score,np.array([[0.,1.],[1.,1.]],np.float64)))
        self.assertTrue(np.array_equal(p,before));self.assertTrue(np.array_equal(score,np.vstack([m.cosine_scores(x[i:i+1],closed[i:i+1],p) for i in range(2)])))
        self.assertTrue(np.array_equal(score[::-1],m.cosine_scores(x[::-1],closed[::-1],p)))

    def test_invalid_mean_unit_label_or_state_is_rejected(self):
        m=self.module();p=np.eye(6,160,dtype=np.float64);x=np.eye(2,160,dtype=np.float32);closed=np.array([0,1],np.int64)
        for kind in ('zero','nan','mean_dtype','unit','closed_dtype','closed_range','proto','huge_mean'):
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                if kind=='zero':m.prototypes(np.zeros((6,160),np.float64))
                elif kind=='nan':m.prototypes(np.full((6,160),np.nan,np.float64))
                elif kind=='mean_dtype':m.prototypes(p.astype(np.float32))
                elif kind=='unit':m.cosine_scores(x*.5,closed,p)
                elif kind=='closed_dtype':m.cosine_scores(x,closed.astype(float),p)
                elif kind=='closed_range':m.cosine_scores(x,np.array([0,6],np.int64),p)
                elif kind=='huge_mean':
                    with np.errstate(over='ignore'):m.prototypes(np.full((6,160),1e300,np.float64))
                else:m.cosine_scores(x,closed,p*.5)

    def test_producer_smoke_then_known_freeze_then_novel_without_parent_scores(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v);out=Path(v[0]['output_root']);parent=Path(v[0]['parent_prediction_root'])
            self.assertLess(events.index('smoke'),events.index('load:known_predictions.npz'));self.assertLess(events.index('load:known_predictions.npz'),events.index('load:novel_predictions.npz'))
            self.assertEqual(done['g_forward_rows'],2);self.assertEqual(done['new_neural_updates'],0);self.assertEqual(done['F_forward_rows'],0);self.assertEqual(done['primary_method'],'maximum_cosine')
            self.assertTrue(done['deployed_g_unchanged']);self.assertTrue(done['source_free_deployment_state']);self.assertEqual(done['cpu_readout_rows'],210)
            deploy=json.loads((out/'deployment/complete.json').read_text());self.assertEqual(set(deploy),m.DEPLOYMENT_FIELDS)
            self.assertEqual((out/'deployment/model.pt').read_bytes(),(parent/'deployment/model.pt').read_bytes())
            with np.load(out/'known_predictions.npz') as z,np.load(parent/'known_predictions.npz') as old:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS)
                for key in m.KNOWN_FIELDS-{'raw_scores','scores'}:self.assertTrue(np.array_equal(z[key],old[key]))
                self.assertEqual(z['scores'].shape,(180,2))

    def test_wrong_parent_arm_state_or_source_permissions_cannot_complete(self):
        m=self.module()
        for kind in ('arm','status','code','F_update','target'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));p=Path(v[0]['parent_prediction_root'])/'complete.json';pm=json.loads(p.read_text())
                if kind=='arm':pm['arm']='compact_oe'
                elif kind=='status':pm['status']='RUNNING'
                elif kind=='code':pm['code_commit']='b'*40
                elif kind=='F_update':pm['neural_F_updates']=1
                else:pm['target_reads']=1
                p.write_text(json.dumps(pm))
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_smoke_failure_illegal_config_and_output_collision_preserve_inputs(self):
        m=self.module()
        for kind in ('smoke','truth','overlap','exists'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));c=v[0]
                if kind=='truth':c['novel_truth']='forbidden'
                elif kind=='overlap':c['output_root']=c['parent_prediction_root']+'/new'
                elif kind=='exists':Path(c['output_root']).mkdir()
                with self.assertRaises(FileExistsError if kind=='exists' else ValueError):self.execute(m,v,smoke_fail=kind=='smoke')
                self.assertFalse((Path(c['output_root'])/'complete.json').is_file())

    def runtime(self,m,root):
        return dict(run_id=m.RUN_ID,code_commit='a'*40,release_root=str(root/'release'),run_root=str(root/'run'),python='python',
            launch_owner='test-owner',parent_prediction_root=str(root/'parent'),novel_truth='truth-private-only-scorer.npz')

    def test_cpu_pipeline_keeps_truth_out_of_config_and_predict_argv_and_checks_terminals(self):
        m=self.module()
        for failed in (False,True):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as td:
                root=Path(td);r=self.runtime(m,root);Path(r['run_root']).mkdir();(Path(r['run_root'])/'logs').mkdir();runtime=root/'runtime.json';runtime.write_text(json.dumps(r));calls=[]
                def job(r,state,path,name,command,gpu):
                    calls.append((name,command,gpu));state['jobs'][name]=dict(status='COMPLETE',pid=123,argv=command,returncode=0)
                    if failed:return False
                    target=Path(r['run_root'])/('predict' if name=='predict' else 'score');target.mkdir()
                    value=dict(run_id=m.RUN_ID,code_commit=r['code_commit'],status=m.STATUS if name=='predict' else 'STAR_METRIC_COSINE_SCORES_COMPLETE',
                        primary_method='maximum_cosine',source_gate_passed=False,eligible_primary=False,next_step='NO_PROMOTION')
                    (target/'complete.json').write_text(json.dumps(value));return True
                with patch.object(m,'_job',side_effect=job):result=m.pipeline(runtime)
                self.assertEqual(result['status'],'FAILED_PRESERVED' if failed else 'ANALYZED_SOURCE_ONLY')
                self.assertEqual([v[0] for v in calls],['predict'] if failed else ['predict','score'])
                self.assertNotIn(r['novel_truth'],calls[0][1]);self.assertTrue(all(gpu is None for _,_,gpu in calls))
                cfg=json.loads((Path(r['run_root'])/'config.json').read_text());self.assertEqual(set(cfg),m.FIELDS)
                self.assertNotIn('novel_truth',cfg)

    def test_start_is_cpu_only_and_cannot_relaunch_existing_root(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r=self.runtime(m,root);runtime=root/'runtime.json';runtime.write_text(json.dumps(r))
            with patch.object(m.subprocess,'Popen',return_value=types.SimpleNamespace(pid=321)) as launch:m.start(runtime)
            args,kwargs=launch.call_args;self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kwargs['cwd'],r['release_root'])
            self.assertIn('pipeline',args[0]);self.assertTrue((Path(r['run_root'])/'launch.json').is_file())
            with self.assertRaises(FileExistsError):m.start(runtime)


if __name__=='__main__':unittest.main()
