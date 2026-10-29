"""Synthetic original-F0 teacher-bank and truth-blind source orchestration."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class NoIQ:
    def __init__(self,n):self.shape=(n,2,256);self.dtype=np.dtype('float32')
    def __getitem__(self,key):raise AssertionError('L/OE or P0 IQ values must not be consumed')


class LocalSupportRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_local_support_run')

    def fixture(self,root):
        m=self.module();output=str(root/'out');source=dict(run_id=m.RUN_ID,row_id='star-local-support-s2026092701',model_seed=2026092701,
            base_release='release-F0',frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},source_output='F0/source',source_contract='contract.json',
            source_cache='old-clean-cache-NOT-OPEN',registered_classes=m.base.native.CLASSES.copy(),code_commit='a'*40,batch_size=128,output_root=output)
        config=dict(run_id=m.RUN_ID,code_commit='a'*40,source_config=source,cache_root=str(root/'cache'),teacher_root=str(root/'teacher'),output_root=output,batch_size=128)
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
            source_output='F0/source',source_contract='contract.json',base_release='release-F0')
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        counts=dict(L=40,OE=40,V=180,novel=30);roles={};features={}
        for role,n in counts.items():
            rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES] if role=='V' else None
            rx=np.array([x[1] for x in rows],np.int64) if rows else np.repeat(np.asarray(m.RECEIVERS,np.int64),n//5)
            day=np.arange(n,dtype=np.int64)%3+1;y=np.array([x[0] for x in rows],np.int64) if rows else np.arange(n,dtype=np.int64)%6
            meta=dict(rx_i=rx,day_i=day,domain=np.searchsorted(m.RECEIVERS,rx)*3+day-1)
            if role in ('L','V'):meta['y']=y
            if role=='V':meta['scenario']=np.asarray([x[2] for x in rows])
            restored=NoIQ(n)
            if role in ('V','novel'):
                restored=np.zeros((n,2,256),np.float32);restored[:,0,0]=y
            manifest=dict(role=role,count=n,source_only=True,received_input_path='P0',restored_input_path='P1',parent_F0_identity=identity,dura_identity=dura)
            roles[role]=dict(ids=np.asarray([role+str(i) for i in range(n)]),metadata=meta,manifest=manifest,received=NoIQ(n),restored=restored)
            if role=='V':roles[role]['baseline_closed']=np.column_stack((y,(y+1)%6)).astype(np.int64)
            f=np.zeros((n,160),np.float32);f[np.arange(n),np.arange(n)%160]=1;features[role]=f
        teacher=Path(config['teacher_root']);teacher.mkdir()
        tm=dict(schema=m.TEACHER_SCHEMA,status=m.TEACHER_STATUS,run_id=m.TEACHER_RUN,code_commit=m.TEACHER_COMMIT,
            parent_F0_identity=identity,dura_identity=dura,model_parameters=232993,input_path='cached_P1_only',feature='unit_fused160',
            source_L_count=40,source_OE_count=40,source_only=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,OE_truth_reads=0,
            encoder_updates=0,identity_head_updates=0,G_updates=0,G_forward_rows=0,cache_handles={k:roles[k]['manifest'] for k in ('L','OE')})
        (teacher/'complete.json').write_text(json.dumps(tm),encoding='utf-8')
        for role in ('L','OE'):np.savez(teacher/(role+'.npz'),ids=roles[role]['ids'],unit_features=features[role])
        model=torch.nn.Linear(1,1).eval().requires_grad_(False);backend=dict(reader=model,dura=None,reader_identity=identity)
        return config,roles,features,backend,counts

    def execute(self,m,v,*,smoke='PASS',mutate_bank=False,mutate_model=False,wrong=False):
        config,roles,features,backend,counts=v;events=[]
        def load(root,role,identity,dura=None):
            events.append('load_'+role)
            if role=='V':self.assertIn('fit_bank',events)
            if role=='novel':self.assertTrue(all((Path(config['output_root'])/f).is_file() for f in ('known_predictions.npz','thresholds.npz','calibration/complete.json')))
            return roles[role]
        def fit(L,L_rx,OE,OE_rx):
            events.append('fit_bank');self.assertTrue(np.array_equal(L,features['L']));self.assertTrue(np.array_equal(OE,features['OE']))
            return dict(L=L.astype(np.float64),OE=OE.astype(np.float64),L_rx=L_rx.copy(),OE_rx=OE_rx.copy())
        def encode(reader,x,device):
            events.append('encode');y=x[:,0,0].astype(int);z=np.zeros((len(x),160),np.float32);z[np.arange(len(x)),y]=3
            logits=np.full((len(x),6),-1.,np.float32);logits[np.arange(len(x)),(y+int(wrong))%6]=2
            if mutate_model:reader.weight.add_(1)
            return dict(fused160=z),logits
        def score(u,state):
            if mutate_bank:state['L'][0,0]+=.1
            return np.tile(u[:,0,None],(1,3)).astype(np.float64)
        with patch.dict(m.COUNTS,counts,clear=True),patch.object(m,'_backend',return_value=(backend,{})),patch.object(m,'_smoke',side_effect=lambda *a:events.append('smoke') or dict(status=smoke)), \
             patch.object(m,'_load_role',side_effect=load),patch.object(m,'close_role'),patch.object(m,'_encode',side_effect=encode), \
             patch.object(m,'_fit',side_effect=fit),patch.object(m,'_score',side_effect=score):done=m.run(config,'cpu')
        return done,events

    def test_teacher_only_bank_no_L_OE_or_P0_forward_then_tau_before_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v);out=Path(v[0]['output_root'])
            self.assertEqual(events[:4],['smoke','load_L','load_OE','fit_bank']);self.assertEqual([x for x in events if x.startswith('load_')],['load_L','load_OE','load_V','load_novel'])
            self.assertEqual(done['F_forward_rows'],212);self.assertEqual(done['source_L_forward_rows'],0);self.assertEqual(done['source_OE_forward_rows'],0)
            self.assertEqual(done['cpu_readout_rows'],210);self.assertGreaterEqual(done['cpu_readout_seconds'],0.)
            self.assertEqual(done['fresh_closed_vs_cached_baseline_changed_count'],180);self.assertFalse(done['default_head_replaced'])
            with np.load(out/'known_predictions.npz') as z:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS);self.assertTrue(np.array_equal(z['baseline_closed'],v[1]['V']['baseline_closed'][:,1]));self.assertTrue(np.allclose(np.linalg.norm(z['unit_features'],axis=1),1))
            with np.load(out/'source_bank.npz') as z:self.assertEqual(set(z.files),m.BANK_FIELDS);self.assertTrue(np.array_equal(z['L_unit_features'],v[2]['L']))
            with np.load(out/'novel_predictions.npz') as z:self.assertEqual(set(z.files),m.NOVEL_FIELDS)

    def test_invalid_teacher_identity_role_ID_order_or_features_cannot_complete(self):
        m=self.module()
        for kind in ('model','commit','role','ids','unit'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));p=Path(v[0]['teacher_root'])
                if kind in ('model','commit','role'):
                    t=json.loads((p/'complete.json').read_text())
                    if kind=='model':t['parent_F0_identity']['model']='r02_candidate'
                    elif kind=='commit':t['code_commit']='c'*40
                    else:t['cache_handles']['OE']['role']='L'
                    (p/'complete.json').write_text(json.dumps(t))
                else:
                    d=dict(np.load(p/'OE.npz'))
                    if kind=='ids':d['ids']=d['ids'][::-1]
                    else:d['unit_features'][0,0]=float('nan')
                    np.savez(p/'OE.npz',**d)
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_config_private_wrong_row_seed_and_input_output_overlap_rejected_before_backend(self):
        m=self.module()
        for kind in ('truth','row','seed','output'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                c,*_=self.fixture(Path(td))
                if kind=='truth':c['novel_truth']='forbidden'
                elif kind=='row':c['source_config']['row_id']='star-consensus-s2026092701'
                elif kind=='seed':c['source_config']['model_seed']=7
                else:c['output_root']=c['teacher_root']+'/new';c['source_config']['output_root']=c['output_root']
                with patch.object(m,'_backend',side_effect=AssertionError('backend')),self.assertRaises(ValueError):m.run(c,'cpu')

    def test_source_bank_or_model_change_blocks_novel(self):
        m=self.module()
        for kind in ('bank','model'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td))
                with self.assertRaises(ValueError):self.execute(m,v,mutate_bank=kind=='bank',mutate_model=kind=='model')
                self.assertFalse((Path(v[0]['output_root'])/'novel_predictions.npz').exists());self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_V_or_novel_cache_requires_exact_original_F0_identity(self):
        m=self.module()
        for role in ('V','novel'):
            with self.subTest(role=role),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));bad=dict(v[1][role]['manifest']['parent_F0_identity'],source_contract='different-contract')
                v[1][role]['manifest']['parent_F0_identity']=bad
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_infeasible_original_baseline_protection_still_completes_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));v[1]['V']['baseline_closed'][:,1]=v[1]['V']['metadata']['y'];done,events=self.execute(m,v,wrong=True)
            self.assertIn('load_novel',events);cal=json.loads((Path(v[0]['output_root'])/'calibration/complete.json').read_text())
            self.assertTrue(all(not x['constraints_passed'] for x in cal['methods'].values()));self.assertEqual(done['status'],m.STATUS)

    def test_failed_smoke_and_existing_output_do_not_consume_source(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with self.assertRaises(ValueError):self.execute(m,v,smoke='FAIL')
            out=Path(v[0]['output_root']);self.assertFalse((out/'complete.json').exists());out.mkdir(exist_ok=True)
            with patch.object(m,'_backend',side_effect=AssertionError('backend')),self.assertRaises(FileExistsError):m.run(v[0],'cpu')


if __name__=='__main__':unittest.main()
