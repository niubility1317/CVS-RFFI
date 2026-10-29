import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class PracticalDevelopmentTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_development')

    def config(self, root):
        return dict(run_id='20261029-phase3-source-practical-readout-s2026092701-r01',code_commit='a'*40,
            condition='C0',view_family='clean_cross_receiver',Q_public_root=str(root/'data/evaluation/public'),
            Q_capsule_id='p3dev-clean-test',Q_split_id='clean-split',Q_expected_count=864,
            source_row_root=str(root/'20261029-phase3-source-embedding-readout-s2026092701-r01/rows/uniform_oe'),
            risk_prediction_root=str(root/'20261029-phase3-source-baseline-risk-s2026092701-r01/rows/uniform_oe/predict'),
            d0_prediction_root=str(root/'20261029-phase3-development-clean-transfer-s2026092701-r01/predict'),
            output_root=str(root/'new/predict'),batch_size=128,dura_checkpoint=None,dura_release=None,
            fit_root=str(root/'fit'),calibration_root=str(root/'calibration'))

    def test_config_rejects_target_fit_labels_counts_and_wrong_condition(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp));self.assertEqual(module.validate_config(cfg),cfg)
            for key in ('U_public_root','truth_path','unknown_count','target_TX_labels'):
                with self.subTest(key=key),self.assertRaises(ValueError):module.validate_config(dict(cfg,**{key:'forbidden'}))
            with self.assertRaises(ValueError):module.validate_config(dict(cfg,condition='P1'))

    def test_source_completion_failure_keeps_target_closed(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp))
            with patch.object(module,'_sources',side_effect=ValueError('not frozen')),patch.object(module,'_public') as opened,patch.object(module,'_backend') as backend:
                with self.assertRaisesRegex(ValueError,'not frozen'):module.predict(cfg,'cpu')
            opened.assert_not_called();backend.assert_not_called()

    def test_two_current_arms_share_forward_and_head_without_fitting(self):
        module=self.module();events=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=self.config(root);cfg['Q_expected_count']=3
            ids=np.asarray(['p3dev-'+('%024x'%j) for j in range(3)])
            iqpath=root/'iq.npy';np.save(iqpath,np.zeros((3,2,256),np.float32));iq=np.load(iqpath,mmap_mode='r')
            class Reader(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False);self.eval()
            reader,original=Reader(),Reader()
            source=dict(marker={'encoder_state_identity':{'arm':'uniform_oe'}},geometry={},state={},thresholds=np.full(7,.5))
            bundles={arm:dict(state={'bias':np.asarray(bias,np.float32)},thresholds=np.full(7,.5),identities={})
                for arm,bias in [('calibration_only',-1),('augmented_readout',1)]}
            def sources(c):events.append('frozen');return source,bundles
            def public(c):events.append('Q');return {'count':3},ids,iq
            def reference(c,s,values):
                return {'run_id':'old'},dict(closed_prediction=np.zeros(3,np.int64),baseline_closed_prediction=np.zeros(3,np.int64),
                    knownness=np.zeros((3,7)),accepted=np.zeros((3,7),bool),prediction=np.full((3,7),-1,np.int64))
            def compact(r,x,s):
                events.append('forward');return np.zeros((len(x),100),np.float32),np.zeros((len(x),5)),np.zeros(len(x),np.int64)
            def scored(d,static,state):return np.column_stack((static,np.full((len(d),2),state['bias'])))
            with patch.object(module,'validate_config',return_value=cfg),patch.object(module,'_sources',side_effect=sources),\
                 patch.object(module,'_backend',return_value=(reader,original,None)),patch.object(module,'_smoke',side_effect=lambda *a:events.append('smoke')),\
                 patch.object(module,'_public',side_effect=public),patch.object(module,'_reference',side_effect=reference),\
                 patch.object(module,'_compact',side_effect=compact),patch.object(module,'_score_compact',side_effect=scored):
                result=module.predict(cfg,'cpu')
            self.assertLess(events.index('frozen'),events.index('Q'));self.assertLess(events.index('smoke'),events.index('Q'))
            self.assertEqual(events.count('forward'),1)
            with np.load(Path(cfg['output_root'])/'predictions.npz',allow_pickle=False) as a:
                self.assertEqual(a['state_names'].tolist(),['original','calibration_only','augmented_readout'])
                self.assertTrue(np.array_equal(a['closed_prediction'][1],a['closed_prediction'][2]))
                self.assertFalse(a['accepted'][1,:,6].any());self.assertTrue(a['accepted'][2,:,6].all())
            self.assertEqual(result['target_U_reads'],0);self.assertFalse(result['query_fit'])
            self.assertTrue(torch.equal(reader.weight,torch.ones(1)))

    def test_incomplete_predictions_never_open_truth(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(module.transfer,'_truth',side_effect=AssertionError('truth opened')):
                with self.assertRaises(ValueError):module.score(root,root/'truth.npz',root/'score',root/'targets.json')

    def test_compact_uses_five_static_scores_without_old_readout_or_ids(self):
        module=self.module()
        from experiments.pamf_phase3_open_world.source_open_export import DIMENSIONS
        from experiments.pamf_phase3_open_world import source_consensus
        data={'features__'+k:np.zeros((2,d),np.float32) for k,d in DIMENSIONS.items()}
        data['logits']=np.zeros((2,6),np.float32)
        source={'geometry':{'branches':{'fused160':{'prototypes':np.ones((6,160),np.float32)}}}}
        with patch.object(module.transfer,'_encode_new',return_value=data),patch.object(module.transfer,'_scores',side_effect=AssertionError('old readout must not run')),\
             patch.object(source_consensus,'descriptor',return_value=np.zeros((2,100),np.float32)):
            desc,static,closed=module._compact(None,None,source)
        self.assertEqual(desc.shape,(2,100));self.assertEqual(static.shape,(2,5));self.assertEqual(closed.tolist(),[0,0])

    def test_actual_truth_dictionary_scores_all_states_and_validates_receiver_and_mapping(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ids=np.asarray(['sample'+str(j) for j in range(9)]);y=np.arange(9,dtype=np.int64)
            done=dict(condition='C0',view_family='clean_cross_receiver')
            closed=np.tile(np.minimum(y,5),(3,1));values=np.broadcast_to((y<6)[None,:,None],(3,9,7)).astype(np.float64).copy()
            pred=dict(ids=ids,closed_prediction=closed,baseline_closed_prediction=closed.copy(),knownness=values,accepted=values>.5)
            truth=root/'truth.npz';scenes=np.full(9,'clean')
            np.savez(truth,ids=ids,y=y,receiver=np.full(9,26,np.int64),scenario=scenes)
            targets=Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'
            with patch.object(module,'_predictions',return_value=(done,pred)):
                result=module.score(root/'predict',truth,root/'score',targets)
                self.assertTrue(result['truth_last']);self.assertTrue(result['primary_pass'])
                self.assertEqual(set(result['states']),set(module.STATES));self.assertEqual(result['target_U_reads'],0)
                np.savez(truth,ids=ids,y=y,receiver=np.full(9,27,np.int64),scenario=scenes)
                with self.assertRaisesRegex(ValueError,'receiver|RX24-1'):module.score(root/'predict',truth,root/'bad_rx',targets)
                np.savez(truth,ids=ids,y=y+1,receiver=np.full(9,26,np.int64),scenario=scenes)
                with self.assertRaisesRegex(ValueError,'mapping|TX'):module.score(root/'predict',truth,root/'bad_tx',targets)


if __name__=='__main__':unittest.main()
