import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class PracticalEncoderTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_encoder')

    def config(self,root):
        run='20261029-phase3-source-practical-encoder-s2026092701-r01'
        return dict(run_id=run,code_commit='a'*40,arm='practical_increment',source_row_root=str(root/'old/rows/uniform_oe'),
            risk_prediction_root=str(root/'risk/rows/uniform_oe/predict'),cache_root=str(root/run/'cache'),
            output_root=str(root/run/'rows/practical_increment/train'),steps=222,lr=.0001,L_seed=2026092701,OE_seed=2026102907,scene_seed=2026102932)

    def test_config_blocks_target_truth_validation_budget_and_replacement(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            c=self.config(Path(temp));self.assertEqual(module.validate_config(c),c)
            for key,value in [('truth','private'),('V_root','forbidden'),('target_U','forbidden'),('unknown_count',3)]:
                with self.subTest(key=key),self.assertRaises(ValueError):module.validate_config(dict(c,**{key:value}))
            for key,value in [('steps',223),('arm','head_only'),('scene_seed',1)]:
                with self.subTest(key=key),self.assertRaises(ValueError):module.validate_config(dict(c,**{key:value}))

    def test_view_selection_preserves_same_rows_and_control_clean(self):
        module=self.module();views={v:np.full((4,2,256),j,np.float32)+np.arange(4,dtype=np.float32)[:,None,None]*10 for j,v in enumerate(module.VIEWS)}
        ids=np.asarray([3,0,2]);scene=np.asarray([0,1,2])
        x=module.select_view(views,ids,scene,'practical_increment')
        self.assertEqual(x[:,0,0].tolist(),[31,2,23])
        x=module.select_view({'clean':views['clean']},ids,scene,'clean_increment')
        self.assertEqual(x[:,0,0].tolist(),[30,0,20])
        with self.assertRaises(ValueError):module.select_view(views,ids,np.asarray([0,1,3]),'practical_increment')

    def test_extra_ce_preserves_one_native_hook_and_updates_shared_head(self):
        module=self.module();calls=[]
        model=torch.nn.Linear(512,6,bias=False)
        raw=torch.ones((2,2,256));y=torch.tensor([0,1]);meta={'cache_index':torch.tensor([1,0])}
        arrays={v:np.ones((2,2,256),np.float32)*(j+1) for j,v in enumerate(module.VIEWS)}
        def native_loss(m,*args):calls.append('native');return m.weight.sum()*0,{'T1_L_packets':768,'CRIC_supcon_weight':.1}
        def logits(m,x):return m(x.flatten(1))
        with patch.object(module.native,'supervised_loss',side_effect=native_loss),patch.object(module.native,'_reader_logits',side_effect=logits):
            combined=module.labeled_loss(arrays,'practical_increment',torch.Generator().manual_seed(1),{})
            loss,metrics=combined(model,raw,y,None,meta,None,None,1);loss.backward()
        self.assertEqual(calls,['native']);self.assertGreater(float(model.weight.grad.abs().sum()),0)
        self.assertAlmostEqual(metrics['extra_CE_weight'],.68);self.assertEqual(metrics['extra_L_packet_views'],2)

    def test_incomplete_checkpoint_never_constructs_model(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(module,'initial_model',side_effect=AssertionError('constructed')):
                with self.assertRaises(ValueError):module.load_trained(Path(temp),'cpu')

    def test_existing_training_output_is_preserved_before_backend(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            c=self.config(Path(temp));out=Path(c['output_root']);out.mkdir(parents=True);(out/'keep').write_text('yes')
            with patch.object(module,'initial_model',side_effect=AssertionError('constructed')):
                with self.assertRaises(FileExistsError):module.train(c,'cpu')
            self.assertEqual((out/'keep').read_text(),'yes')

    def test_cached_l_keeps_native_label_permission_and_cache_index(self):
        module=self.module()
        meta={k:np.asarray([4,9],np.int64) for k in ('y','domain','rx_i','day_i','eq_i','sig_i','base_index')}
        data=module.CachedL({'clean':np.ones((2,2,256),np.float32)},meta)
        x,y,domain,extra=data[1]
        self.assertTrue(extra['tx_label_visible']);self.assertEqual(int(extra['cache_index']),1)
        self.assertEqual(int(extra['base_index']),9);self.assertEqual(int(y),9)


if __name__=='__main__':unittest.main()
