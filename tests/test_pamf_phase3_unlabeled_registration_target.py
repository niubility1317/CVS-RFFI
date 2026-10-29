import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class RegistrationTargetTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.unlabeled_registration_target')
    def config(self,root):
        return dict(run_id='20261029-phase3-unlabeled-registration-s2026092701-r01',code_commit='a'*40,
            condition='C0',view_family='clean_cross_receiver',source_root=str(root/'source'),
            U_public_root=str(root/'data/discovery/public'),U_capsule_id='u',U_split_id='us',U_expected_count=384,
            Q_public_root=str(root/'data/evaluation/public'),Q_capsule_id='q',Q_split_id='qs',Q_expected_count=864,
            output_root=str(root/'predict'),batch_size=128,dura_checkpoint=None,dura_release=None)

    def test_config_excludes_truth_unknown_number_and_role_swap(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            c=self.config(Path(temp));self.assertEqual(module.validate_config(c),c)
            for k in ('U_truth','Q_truth','unknown_count','new_class_quota'):
                with self.subTest(key=k),self.assertRaises(ValueError):module.validate_config(dict(c,**{k:'forbidden'}))
            with self.assertRaises(ValueError):module.validate_config(dict(c,U_public_root=c['Q_public_root']))

    def test_source_incomplete_never_opens_target_or_model(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            c=self.config(Path(temp))
            with patch.object(module,'_source',side_effect=ValueError('not frozen')),patch.object(module,'_open') as opened,patch.object(module,'_backend') as backend:
                with self.assertRaisesRegex(ValueError,'not frozen'):module.predict(c,'cpu')
            opened.assert_not_called();backend.assert_not_called()

    def test_u_state_freezes_before_future_q_and_zero_clusters_keep_four_states(self):
        module=self.module();events=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=self.config(root);c.update(U_expected_count=3,Q_expected_count=3)
            ids={r:np.asarray([r+str(j) for j in range(3)]) for r in ('U','Q')}
            maps={}
            for r in ids:
                p=root/(r+'.npy');np.save(p,np.zeros((3,2,256),np.float32));maps[r]=np.load(p,mmap_mode='r')
            class Reader(torch.nn.Module):
                def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False);self.eval()
            model,original=Reader(),Reader()
            source=dict(geometry={},state={},thresholds=np.full(7,.5),marker={},statistics=dict(
                sigma=np.ones((2,7)),eps=np.full(2,.1),radius=np.full(2,.1),
                anchor_units=np.tile(np.eye(1,160,dtype=np.float32),(2,1,1)),anchor_scores=np.full((2,1,7),2.)))
            def metadata(config,role):events.append('metadata'+role);return {},ids[role]
            def opened(config,role):
                if role=='Q':self.assertTrue((root/'predict/adaptation/complete.json').is_file())
                events.append('open'+role);return {},ids[role],maps[role]
            def encoded(reader,identity,x):
                n=len(x);z={k:np.zeros((n,d),np.float32) for k,d in module.DIMENSIONS.items()}
                return z,np.zeros((n,6),np.float32),np.zeros((n,6),np.float32),np.tile(np.eye(1,160,dtype=np.float32),(n,1))
            bias={k:np.zeros(d) for k,d in module.DIMENSIONS.items()}
            with patch.object(module,'validate_config',return_value=c),patch.object(module,'_source',return_value=source),\
                 patch.object(module,'_backend',return_value=(model,original,None)),patch.object(module,'_smoke'),\
                 patch.object(module,'_metadata',side_effect=metadata),patch.object(module,'_open',side_effect=opened),\
                 patch.object(module,'_encode',side_effect=encoded),patch.object(module,'_scores',side_effect=lambda b,l,s:np.full((len(l),7),.8)),\
                 patch.object(module.reg,'fit_bias',return_value=(bias,dict(applied=False,core_mask=np.zeros(3,bool),pseudo_labels=np.zeros(3,np.int64),support_counts=np.zeros(6,np.int64),qualified_classes=[]))),\
                 patch.object(module.reg,'embedding',side_effect=lambda b,g:np.zeros((len(b['fused160']),688),np.float32)):
                done=module.predict(c,'cpu');loaded,p,diag,state=module._predictions(root/'predict')
            self.assertLess(events.index('openU'),events.index('openQ'));self.assertFalse(done['bias_applied'])
            self.assertEqual(done['prototype_counts'],{'DA0':0,'DA1':0});self.assertEqual(p['prediction'].shape,(4,3))
            self.assertTrue(np.array_equal(p['prediction'][0],p['prediction'][3]))
            self.assertTrue(np.array_equal(p['closed_prediction'],p['baseline_closed_prediction']))
            self.assertEqual(done['target_labels_read'],0);self.assertFalse(done['query_fit'])

    def test_incomplete_predictions_never_pass_truth_gate(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):module._predictions(Path(temp))


if __name__=='__main__':unittest.main()
