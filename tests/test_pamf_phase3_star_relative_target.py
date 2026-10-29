import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch


class TargetTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_relative_target')
    def config(self,r):
        return dict(run_id='20261029-phase3-star-relative-s2026092701-r01',code_commit='a'*40,condition='P0',view_family='practical_cross_receiver',
            source_root=str(r/'source'),U_public_root=str(r/'discovery/public'),U_capsule_id='u',U_split_id='us',U_expected_count=384,
            Q_public_root=str(r/'evaluation/public'),Q_capsule_id='q',Q_split_id='qs',Q_expected_count=864,output_root=str(r/'predict'),
            batch_size=128,dura_checkpoint=None,dura_release=None)
    def test_no_truth_quota_clean_or_role_swap(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            c=self.config(Path(d));self.assertEqual(c,m.validate_config(c))
            for k in ('U_truth','Q_truth','unknown_count','new_class_quota'):
                with self.subTest(field=k),self.assertRaises(ValueError):m.validate_config(dict(c,**{k:'forbidden'}))
            with self.assertRaises(ValueError):m.validate_config(dict(c,condition='C0',view_family='clean_cross_receiver'))
            with self.assertRaises(ValueError):m.validate_config(dict(c,U_public_root=c['Q_public_root']))
    def test_source_before_model_and_public(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with patch.object(m,'_source',side_effect=ValueError('not frozen')),patch.object(m,'_metadata') as public,patch.object(m,'_backend') as model:
                with self.assertRaisesRegex(ValueError,'not frozen'):m.predict(self.config(Path(d)),'cpu')
            public.assert_not_called();model.assert_not_called()
    def test_four_states_frozen_before_q_with_no_core_or_clusters(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);c=self.config(r);c.update(U_expected_count=3,Q_expected_count=3)
            ids={x:np.asarray([x+str(i) for i in range(3)]) for x in ('U','Q')};maps={}
            for x in ids:
                p=r/(x+'.npy');np.save(p,np.zeros((3,2,256),np.float32));maps[x]=np.load(p,mmap_mode='r')
            class Reader(torch.nn.Module):
                def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False);self.eval()
            source=dict(geometry={},metric={},thresholds=np.full(7,.5),marker={},statistics=dict(sigma=np.ones(7),eps=np.asarray(.1),radius=np.asarray(.1),
                anchor_units=np.eye(1,160,dtype=np.float32),anchor_scores=np.full((1,7),2.)))
            def opened(cfg,role):
                if role=='Q':self.assertTrue((r/'predict/adaptation/complete.json').is_file())
                events.append(role);return {},ids[role],maps[role]
            def encoded(reader,x):
                n=len(x);return {k:np.zeros((n,v),np.float32) for k,v in m.DIMENSIONS.items()},np.zeros((n,6),np.float32),np.tile(np.eye(1,160,dtype=np.float32),(n,1))
            biases={k:np.zeros(v) for k,v in m.DIMENSIONS.items()}
            with patch.object(m,'validate_config',return_value=c),patch.object(m,'_source',return_value=source),patch.object(m,'_backend',return_value=(Reader(),None)),patch.object(m,'_smoke'),\
                 patch.object(m,'_metadata',side_effect=lambda cfg,role:({},ids[role])),patch.object(m,'_open',side_effect=opened),patch.object(m,'_encode',side_effect=encoded),\
                 patch.object(m,'_scores',side_effect=lambda b,l,s:np.full((len(l),7),.8)),patch.object(m.core,'fit_bias_single',return_value=(biases,dict(applied=False))),\
                 patch.object(m.reg,'embedding',side_effect=lambda b,g:np.zeros((len(b['fused160']),688),np.float32)):
                done=m.predict(c,'cpu');loaded,p,diag,state=m._predictions(r/'predict')
            self.assertEqual(events,['U','Q']);self.assertEqual(p['prediction'].shape,(4,3));self.assertEqual(done['prototype_counts'],{'DA0':0,'DA1':0})
            self.assertFalse(done['bias_applied']);self.assertEqual(done['primary_method'],'relative_mahalanobis')
            self.assertTrue(np.array_equal(p['closed_prediction'],p['baseline_closed_prediction']));self.assertTrue(np.array_equal(p['prediction'][0],p['prediction'][3]))
            self.assertEqual(done['target_labels_read'],0);self.assertFalse(done['query_fit']);self.assertEqual(done['encoder_models'],1)
    def test_incomplete_prediction_gate(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):self.module()._predictions(d)


if __name__=='__main__':unittest.main()
