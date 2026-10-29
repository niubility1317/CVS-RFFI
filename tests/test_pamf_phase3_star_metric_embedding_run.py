"""Small synthetic checks for fixed-budget metric training and source-free export."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class MetricEmbeddingRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_metric_embedding_run')

    def fixture(self,root,arm='identity'):
        m=self.module();out=str(root/'out')
        source=dict(run_id=m.RUN_ID,row_id='star-metric-embedding-'+arm+'-s2026092701',model_seed=2026092701,
            base_release='release-F0',frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},source_output='F0/source',source_contract='contract.json',
            source_cache='old-clean-cache-NOT-OPEN',registered_classes=m.base.base.base.native.CLASSES.copy(),code_commit='a'*40,batch_size=128,output_root=out)
        config=dict(run_id=m.RUN_ID,code_commit='a'*40,arm=arm,source_config=source,parent_prediction_root=str(root/'parent'),cache_root=str(root/'cache'),output_root=out,batch_size=128)
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
            source_output='F0/source',source_contract='contract.json',base_release='release-F0')
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        counts=dict(L=120,OE=128,V=180,novel=30);parent=Path(config['parent_prediction_root']);parent.mkdir()
        cache=Path(config['cache_root']);(cache/'L').mkdir(parents=True);rng=np.random.default_rng(91)
        ly=np.repeat(np.arange(6,dtype=np.int64),20);lrx=np.tile(np.repeat(np.asarray(m.RECEIVERS,np.int64),4),6)
        def units(y):
            x=rng.normal(0,.05,(len(y),160)).astype(np.float32);x[np.arange(len(y)),y]+=1
            return (x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)
        bank=dict(L_ids=np.asarray(['L'+str(i) for i in range(120)]),L_unit_features=units(ly),L_rx_i=lrx,
            OE_ids=np.asarray(['OE'+str(i) for i in range(128)]),OE_unit_features=units(np.arange(128)%6+20),OE_rx_i=np.resize(np.asarray(m.RECEIVERS,np.int64),128))
        np.savez(parent/'source_bank.npz',**bank);np.save(cache/'L/ids.npy',bank['L_ids'])
        np.savez(cache/'L/metadata.npz',y=ly,rx_i=lrx,day_i=np.arange(120,dtype=np.int64)%3+1,domain=np.zeros(120,np.int64))
        handles={r:dict(role=r,count=n,parent_F0_identity=identity,dura_identity=dura,source_only=True) for r,n in counts.items()}
        (cache/'L/manifest.json').write_text(json.dumps(handles['L']),encoding='utf-8')
        marker=dict(schema=m.base.PARENT_SCHEMA,status=m.base.PARENT_STATUS,run_id=m.base.PARENT_RUN,code_commit=m.base.PARENT_COMMIT,
            model_identity=identity,dura_identity=dura,cache_root=config['cache_root'],cache_identities=handles,
            teacher_run_id=m.base.base.TEACHER_RUN,teacher_code_commit=m.base.base.TEACHER_COMMIT,teacher_root='original-F0-teacher',
            known_count=180,novel_count=30,bank_L_count=120,bank_OE_count=128,classification_path='P1',source_only=True,
            target_reads=0,source_novel_truth_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,source_U_reads=0,default_head_replaced=False,
            source_bank_frozen_before_V=True,source_bank_unchanged=True,model_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True,
            fresh_closed_vs_cached_baseline_changed_count=0)
        (parent/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        for role,n in (('known',180),('novel',30)):
            y=np.asarray([r[0] for r in rows],np.int64) if role=='known' else np.arange(n,dtype=np.int64)%6
            rx=np.asarray([r[1] for r in rows],np.int64) if role=='known' else np.resize(np.asarray(m.RECEIVERS,np.int64),n)
            z=np.full((n,6),-2.,np.float32);z[np.arange(n),y]=2
            data=dict(ids=np.asarray([role+str(i) for i in range(n)]),rx_i=rx,day_i=np.arange(n,dtype=np.int64)%3+1,unit_features=units(y),
                default_logits=z,closed=y,raw_scores=np.zeros((n,3),np.float64),scores=np.ones((n,3),np.float64))
            if role=='known':data.update(y=y.copy(),baseline_closed=y.copy(),scenario=np.asarray([r[2] for r in rows]))
            np.savez(parent/(role+'_predictions.npz'),**data)
        reader=torch.nn.Linear(1,1).eval().requires_grad_(False);backend=dict(reader=reader,dura=None,reader_identity=identity)
        bank['L_y']=ly
        return config,counts,backend,bank

    def execute(self,m,v,*,smoke='PASS',metric_change=None):
        config,counts,backend,bank=v;events=[];real_public=m._public;real_scores=m._metric_scores
        def public(config,role,seen):
            events.append('public:'+role);out=Path(config['output_root'])
            if role=='known':
                self.assertTrue((out/'training_complete.json').is_file());self.assertTrue((out/'deployment/model.pt').is_file())
            else:
                self.assertEqual({p.name for p in (out/'deployment').iterdir()},{'model.pt','metric.npz','thresholds.npz','complete.json'})
                self.assertTrue((out/'calibration/complete.json').is_file());self.assertTrue((out/'known_predictions.npz').is_file())
            return real_public(config,role,seen)
        def scores(features,closed,metric):
            if metric_change=='aggregate':metric['class_means'][0,0]+=.1
            return real_scores(features,closed,metric)
        with patch.dict(m.COUNTS,counts,clear=True),patch.object(m,'TRAIN_STEPS',2),patch.object(m,'_backend',return_value=(backend,{})), \
             patch.object(m,'_smoke',side_effect=lambda *args:events.append('smoke') or dict(status=smoke)), \
             patch.object(m,'_public',side_effect=public),patch.object(m,'_metric_scores',side_effect=scores):
            done=m.run(config,'cpu')
        return done,events

    def test_identity_exports_only_fixed_network_aggregates_and_thresholds_before_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v);out=Path(v[0]['output_root'])
            self.assertEqual(events,['smoke','public:known','public:novel']);self.assertEqual((out/'training.jsonl').read_text(),'')
            self.assertEqual(done['g_training_steps'],0);self.assertEqual(done['g_forward_rows'],330)
            self.assertEqual(done['primary_method'],'predicted_identity_mahalanobis')
            self.assertEqual(done['method_names'],['minimum_relative','predicted_identity_mahalanobis'])
            self.assertEqual(done['F_forward_rows'],2);self.assertEqual(done['real_IQ_values_read'],0);self.assertEqual(done['neural_F_updates'],0)
            self.assertTrue(done['source_free_deployment_state']);self.assertEqual(done['per_record_source_rows_retained'],0)
            deploy=json.loads((out/'deployment/complete.json').read_text());self.assertEqual(set(deploy),m.DEPLOYMENT_FIELDS)
            tensors=torch.load(out/'deployment/model.pt',map_location='cpu',weights_only=True)
            self.assertEqual(set(tensors),set(m.MODEL_FIELDS));self.assertEqual(sum(t.numel() for t in tensors.values()),20704)
            self.assertTrue(all(t.dtype==torch.float32 for t in tensors.values()))
            with np.load(out/'deployment/metric.npz') as z:self.assertEqual(set(z.files),set(m.METRIC_FIELDS));self.assertEqual(sum(z[k].size for k in z.files),52320)
            with np.load(out/'deployment/thresholds.npz') as z:self.assertEqual(set(z.files),{'conditional'});self.assertEqual(z['conditional'].shape,(2,6))
            with np.load(out/'known_predictions.npz') as z,np.load(Path(v[0]['parent_prediction_root'])/'known_predictions.npz') as p:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS)
                for key in m.base.KNOWN_FIELDS-{'raw_scores','scores'}:self.assertTrue(np.array_equal(z[key],p[key]))
                self.assertTrue(np.allclose(z['metric_unit_features'],p['unit_features'],rtol=0,atol=2e-6))
            self.assertFalse((out/'source_bank.npz').exists());self.assertFalse((out/'thresholds.npz').exists())

    def test_real_small_training_records_full_budget_only_metric_gradients_and_unique_coverage(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td),'compact_oe_structure');before={k:t.clone() for k,t in v[2]['reader'].state_dict().items()}
            done,events=self.execute(m,v);out=Path(v[0]['output_root']);trace=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
            self.assertEqual(len(trace),2);self.assertEqual([r['step'] for r in trace],[1,2]);self.assertEqual(done['g_training_steps'],2)
            self.assertEqual(done['g_training_L_exposures'],240);self.assertEqual(done['g_training_OE_exposures'],256);self.assertEqual(done['g_forward_rows'],826)
            self.assertEqual(len(set(i for r in trace for i in r['L_indices'])),120);self.assertEqual(len(set(i for r in trace for i in r['OE_indices'])),128)
            for row in trace:
                self.assertEqual(set(row),m.TRACE_FIELDS);self.assertGreater(row['g_grad_norm'],0)
                self.assertEqual(len(set(row['L_indices'])),120);self.assertEqual(len(set(row['OE_indices'])),128)
                self.assertTrue(np.isfinite([row[k] for k in ('total','known_contrast','OE_repulsion','OE_structure')]).all())
            self.assertTrue(all(torch.equal(t,before[k]) for k,t in v[2]['reader'].state_dict().items()))
            self.assertTrue(all(p.grad is None for p in v[2]['reader'].parameters()))
            train=json.loads((out/'training_complete.json').read_text());self.assertEqual(train['g_training_L_unique'],120);self.assertEqual(train['g_training_OE_unique'],128)

    def test_cpu_initialization_and_sample_trajectory_match_between_learned_arms(self):
        m=self.module();traces=[]
        with tempfile.TemporaryDirectory() as td:
            for arm in ('compact_oe','compact_oe_structure'):
                root=Path(td)/arm;root.mkdir();v=self.fixture(root,arm)
                model=m._new_model('cpu');initial={k:t.clone() for k,t in model.state_dict().items()}
                if traces:self.assertTrue(all(torch.equal(initial[k],traces[0][0][k]) for k in initial))
                trace_path=root/'trace.jsonl'
                with patch.object(m,'TRAIN_STEPS',2):summary=m._train(model,v[3],arm,'cpu',trace_path,v[0])
                rows=[json.loads(line) for line in trace_path.read_text().splitlines()];traces.append((initial,rows))
                self.assertEqual(summary['g_training_steps'],2)
            for left,right in zip(traces[0][1],traces[1][1]):
                self.assertEqual(left['L_indices'],right['L_indices']);self.assertEqual(left['OE_indices'],right['OE_indices'])

    def test_aggregate_change_during_V_blocks_novel_and_preserves_partial(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with self.assertRaises(ValueError):self.execute(m,v,metric_change='aggregate')
            out=Path(v[0]['output_root']);self.assertTrue((out/'training_complete.json').is_file())
            self.assertFalse((out/'novel_predictions.npz').exists());self.assertFalse((out/'complete.json').exists())

    def test_nonfinite_training_loss_stops_without_false_training_completion(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td),'compact_oe');model=m._new_model('cpu')
            def bad(*args):
                nan=torch.tensor(float('nan'),requires_grad=True)
                return {key:nan for key in ('total','known_contrast','OE_repulsion','OE_structure')}
            with patch.object(m,'TRAIN_STEPS',2),patch.object(m,'_losses',side_effect=bad),self.assertRaises(ValueError):
                m._train(model,v[3],'compact_oe','cpu',Path(td)/'trace.jsonl',v[0])

    def test_failed_smoke_precedes_all_parent_feature_reads(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with patch.object(m,'_parent',side_effect=AssertionError('source')),self.assertRaises(ValueError):self.execute(m,v,smoke='FAIL')
            self.assertFalse(Path(v[0]['output_root']).exists())

    def test_illegal_arm_source_private_key_overlap_and_existing_output_fail_before_backend(self):
        m=self.module()
        for kind in ('arm','truth','row','overlap','exists'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                c,*_=self.fixture(Path(td))
                if kind=='arm':c['arm']='future'
                elif kind=='truth':c['novel_truth']='forbidden'
                elif kind=='row':c['source_config']['row_id']='other'
                elif kind=='overlap':c['output_root']=c['parent_prediction_root']+'/new';c['source_config']['output_root']=c['output_root']
                else:Path(c['output_root']).mkdir()
                with patch.object(m,'_backend',side_effect=AssertionError('backend')),self.assertRaises(FileExistsError if kind=='exists' else ValueError):m.run(c,'cpu')


if __name__=='__main__':unittest.main()
