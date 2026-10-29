"""Synthetic parent-feature reuse with truth-blind conditional calibration."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class ConditionalSupportRunTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_conditional_support_run')

    def fixture(self, root):
        m = self.module(); out = str(root / 'out')
        source = dict(run_id=m.RUN_ID, row_id='star-conditional-support-s2026092701', model_seed=2026092701,
            base_release='release-F0', frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')}, source_output='F0/source',
            source_contract='contract.json', source_cache='old-clean-cache-NOT-OPEN', registered_classes=m.base.base.native.CLASSES.copy(),
            code_commit='a'*40, batch_size=128, output_root=out)
        config = dict(run_id=m.RUN_ID, code_commit='a'*40, source_config=source, parent_prediction_root=str(root/'parent'),
            cache_root=str(root/'cache'), output_root=out, batch_size=128)
        identity = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100, model_seed=2026092701,
            source_output='F0/source', source_contract='contract.json', base_release='release-F0')
        dura = dict(deployment_id='dura_v1', parameters=266951, epoch=20, updates=4940)
        counts = dict(L=48, OE=40, V=180, novel=30)
        parent = Path(config['parent_prediction_root']); parent.mkdir()
        cache = Path(config['cache_root']); (cache/'L').mkdir(parents=True)
        bank = {}
        for role in ('L','OE'):
            n = counts[role]; unit = np.zeros((n,160),np.float32); unit[np.arange(n),np.arange(n)%6] = 1
            bank.update({role+'_ids':np.asarray([role+str(i) for i in range(n)]), role+'_unit_features':unit,
                role+'_rx_i':np.resize(np.asarray(m.RECEIVERS,np.int64),n)})
        np.savez(parent/'source_bank.npz',**bank)
        ly = np.arange(counts['L'],dtype=np.int64)%6
        np.save(cache/'L/ids.npy',bank['L_ids'])
        np.savez(cache/'L/metadata.npz',y=ly,rx_i=bank['L_rx_i'],day_i=np.arange(counts['L'],dtype=np.int64)%3+1,domain=np.zeros(counts['L'],np.int64))
        handles = {r:dict(role=r,count=n,parent_F0_identity=identity,dura_identity=dura,source_only=True) for r,n in counts.items()}
        (cache/'L/manifest.json').write_text(json.dumps(handles['L']),encoding='utf-8')
        marker = dict(schema=m.PARENT_SCHEMA,status=m.PARENT_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,
            model_identity=identity,dura_identity=dura,cache_root=config['cache_root'],cache_identities=handles,
            teacher_run_id=m.base.TEACHER_RUN,teacher_code_commit=m.base.TEACHER_COMMIT,teacher_root='original-F0-teacher',
            known_count=counts['V'],novel_count=counts['novel'],bank_L_count=counts['L'],bank_OE_count=counts['OE'],
            classification_path='P1',source_only=True,target_reads=0,source_novel_truth_reads=0,neural_updates=0,G_updates=0,
            G_forward_rows=0,source_U_reads=0,default_head_replaced=False,source_bank_frozen_before_V=True,source_bank_unchanged=True,
            model_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True,
            fresh_closed_vs_cached_baseline_changed_count=0)
        (parent/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        tau = np.full((3,6),-1.,np.float64)
        np.savez(parent/'thresholds.npz',conditional=tau)
        (parent/'calibration').mkdir()
        control = dict(constraints_passed=True,conditional_thresholds=tau[1].tolist(),groups={'fixture':{'constraints_passed':True}})
        calibration = dict(schema=m.PARENT_SCHEMA,status=m.base.CALIBRATION_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,
            method_names=list(m.base.METHODS),conditional_thresholds=tau.tolist(),methods={'pooled_relative':control},group_count=42,
            source_only=True,target_reads=0)
        (parent/'calibration/complete.json').write_text(json.dumps(calibration),encoding='utf-8')
        rows = [(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES]
        for role,n in (('known',counts['V']),('novel',counts['novel'])):
            y = np.asarray([r[0] for r in rows],np.int64) if role=='known' else np.arange(n,dtype=np.int64)%6
            rx = np.asarray([r[1] for r in rows],np.int64) if role=='known' else np.resize(np.asarray(m.RECEIVERS,np.int64),n)
            unit = np.zeros((n,160),np.float32); unit[np.arange(n),y] = 1
            logits = np.full((n,6),-2.,np.float32); logits[np.arange(n),y] = 2
            raw = np.column_stack((unit[:,0],unit[:,0],unit[:,1])).astype(np.float64)
            data = dict(ids=np.asarray([role+str(i) for i in range(n)]),rx_i=rx,day_i=np.arange(n,dtype=np.int64)%3+1,
                unit_features=unit,default_logits=logits,closed=y,raw_scores=raw,scores=raw-tau[:,y].T)
            if role=='known':data.update(y=y.copy(),baseline_closed=y.copy(),scenario=np.asarray([r[2] for r in rows]))
            np.savez(parent/(role+'_predictions.npz'),**data)
        model = torch.nn.Linear(1,1).eval().requires_grad_(False)
        backend = dict(reader=model,dura=None,reader_identity=identity)
        return config,counts,backend

    def execute(self,m,v,*,smoke='PASS',mutate_bank=False,mutate_model=False):
        config,counts,backend = v; events=[]; original_np_load=np.load
        def load(path,*args,**kwargs):
            name=Path(path).name; events.append('load:'+name)
            if name=='novel_predictions.npz':
                out=Path(config['output_root'])
                self.assertTrue(all((out/p).is_file() for p in ('known_predictions.npz','thresholds.npz','calibration/complete.json')))
            if name in ('received.npy','restored.npy') or str(path).startswith('old-clean-cache'):
                raise AssertionError('Real IQ must not be opened')
            self.assertIn('smoke',events)
            return original_np_load(path,*args,**kwargs)
        def fit(L,y,OE):
            events.append('fit_bank'); return dict(L=L.astype(np.float64),L_y=y.copy(),OE=OE.astype(np.float64))
        def score(features,closed,state):
            events.append('score')
            if mutate_bank:state['L'][0,0]+=.1
            if mutate_model:backend['reader'].weight.add_(1)
            return np.column_stack((features[:,0],features[:,1])).astype(np.float64)
        with patch.dict(m.COUNTS,counts,clear=True),patch.object(m,'_backend',return_value=(backend,{})), \
             patch.object(m,'_smoke',side_effect=lambda *a:events.append('smoke') or dict(status=smoke)), \
             patch.object(m,'_fit',side_effect=fit),patch.object(m,'_score',side_effect=score),patch.object(m.np,'load',side_effect=load):
            done=m.run(config,'cpu')
        return done,events

    def test_smoke_before_parent_values_bank_before_V_and_tau_before_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td));done,events=self.execute(m,v);out=Path(v[0]['output_root']);parent=Path(v[0]['parent_prediction_root'])
            self.assertEqual(events[0],'smoke');self.assertLess(events.index('fit_bank'),events.index('load:known_predictions.npz'))
            self.assertEqual(done['F_forward_rows'],2);self.assertEqual(done['real_IQ_values_read'],0)
            self.assertEqual(done['source_L_metadata_labels_read'],48);self.assertEqual(done['new_V_forward_rows'],0)
            self.assertEqual(done['cpu_readout_rows'],210);self.assertGreaterEqual(done['cpu_readout_seconds'],0)
            self.assertEqual(done['fresh_closed_vs_cached_baseline_changed_count'],0);self.assertTrue(done['control_scores_thresholds_acceptance_reproduced'])
            with np.load(out/'source_bank.npz') as z:self.assertEqual(set(z.files),m.BANK_FIELDS)
            with np.load(out/'known_predictions.npz') as z,np.load(parent/'known_predictions.npz') as p:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS)
                for field in m.KNOWN_FIELDS-{'raw_scores','scores'}:self.assertTrue(np.array_equal(z[field],p[field]))
                self.assertTrue(np.array_equal(z['raw_scores'][:,0],p['raw_scores'][:,1]));self.assertTrue(np.array_equal(z['scores'][:,0],p['scores'][:,1]))
            with np.load(out/'novel_predictions.npz') as z:self.assertEqual(set(z.files),m.NOVEL_FIELDS)
            cal=json.loads((out/'calibration/complete.json').read_text());old=json.loads((parent/'calibration/complete.json').read_text())
            self.assertEqual(cal['methods']['pooled_relative'],old['methods']['pooled_relative'])

    def test_parent_model_status_commit_or_source_permissions_rejected(self):
        m=self.module()
        for kind in ('model','status','commit','target','teacher'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));path=Path(v[0]['parent_prediction_root'])/'complete.json';marker=json.loads(path.read_text())
                if kind=='model':marker['model_identity']['model']='candidate'
                elif kind=='status':marker['status']='RUNNING'
                elif kind=='commit':marker['code_commit']='b'*40
                elif kind=='target':marker['target_reads']=1
                else:marker['teacher_code_commit']='c'*40
                path.write_text(json.dumps(marker))
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_source_L_label_ID_RX_or_private_field_mismatch_rejected(self):
        m=self.module()
        for kind in ('ids','labels','RX','private'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));cache=Path(v[0]['cache_root'])/'L'
                if kind=='ids':np.save(cache/'ids.npy',np.load(cache/'ids.npy')[::-1])
                else:
                    d=dict(np.load(cache/'metadata.npz'))
                    if kind=='labels':d['y'][0]=6
                    elif kind=='RX':d['rx_i'][0]=99
                    else:d['unknown_TX']=np.full(len(d['y']),'forbidden')
                    np.savez(cache/'metadata.npz',**d)
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'novel_predictions.npz').exists())

    def test_parent_default_identity_baseline_or_control_change_cannot_complete(self):
        m=self.module()
        for kind in ('closed','baseline','raw','margin','threshold'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td));root=Path(v[0]['parent_prediction_root'])
                if kind=='threshold':
                    d=dict(np.load(root/'thresholds.npz'));d['conditional'][1,0]=.5;np.savez(root/'thresholds.npz',**d)
                else:
                    d=dict(np.load(root/'known_predictions.npz'))
                    if kind=='closed':d['closed'][0]=1
                    elif kind=='baseline':d['baseline_closed'][0]=1
                    elif kind=='raw':d['raw_scores'][0,1]+=.1
                    else:d['scores'][0,1]+=.1
                    np.savez(root/'known_predictions.npz',**d)
                with self.assertRaises(ValueError):self.execute(m,v)
                self.assertFalse((Path(v[0]['output_root'])/'complete.json').exists())

    def test_bank_or_model_change_blocks_novel_and_retains_partial(self):
        m=self.module()
        for kind in ('bank','model'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                v=self.fixture(Path(td))
                with self.assertRaises(ValueError):self.execute(m,v,mutate_bank=kind=='bank',mutate_model=kind=='model')
                out=Path(v[0]['output_root']);self.assertTrue((out/'smoke.json').is_file())
                self.assertFalse((out/'novel_predictions.npz').exists());self.assertFalse((out/'complete.json').exists())

    def test_illegal_config_and_overwrite_fail_before_backend(self):
        m=self.module()
        for kind in ('truth','row','seed','output','exists'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                c,*_=self.fixture(Path(td))
                if kind=='truth':c['novel_truth']='forbidden'
                elif kind=='row':c['source_config']['row_id']='other'
                elif kind=='seed':c['source_config']['model_seed']=7
                elif kind=='output':c['output_root']=c['parent_prediction_root']+'/new';c['source_config']['output_root']=c['output_root']
                else:Path(c['output_root']).mkdir()
                with patch.object(m,'_backend',side_effect=AssertionError('backend')),self.assertRaises(FileExistsError if kind=='exists' else ValueError):m.run(c,'cpu')

    def test_failed_smoke_consumes_no_parent_values(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            v=self.fixture(Path(td))
            with self.assertRaises(ValueError):self.execute(m,v,smoke='FAIL')
            self.assertFalse(Path(v[0]['output_root']).exists())


if __name__=='__main__':unittest.main()
