import unittest
import torch
from experiments.cric_dura_comparison.plan import make_plan
from experiments.cric_dura_comparison.continuation_setup import final_t1_rows
from experiments.cric_dura_comparison.t1_model import pair


class FinalT1(unittest.TestCase):
    def test_final_matrix_reuses_five_source_and_preserves_trajectory(self):
        p=make_plan();h={pid:dict(classes=list(range(10 if v['dataset']=='ManyRx' else 6)),source_cache='/cache/'+pid,source_contract='/contract/'+pid) for pid,v in p['protocols'].items()}
        rows=final_t1_rows(dict(project='/project',code_commit='fixed',frozen_sources={}),p,h)
        self.assertEqual(len(rows),213);self.assertEqual(sum(bool(r['ablation']) for r in rows),63)
        self.assertFalse(any(r['protocol_id']=='F0' and r['ablation'] is None and r['regime']=='native_cric' for r in rows))
        self.assertFalse(any(r['protocol_id'].startswith('m5-') for r in rows))
        for row in p['rows']:
            self.assertEqual(row['method_id'],'pamf_t1_e50');self.assertEqual(row['trajectory'],[20,10,20])
            if row['regime']=='native_cric':self.assertEqual(row['U_start_epoch'],14)
    def test_As_KL_zero_keeps_two_true_label_CE_for_six_or_ten_TX(self):
        for classes in (6,10):
            a=torch.randn(64,classes,requires_grad=True);b=torch.randn(64,classes,requires_grad=True);y=torch.arange(64)%classes
            value,ce,kl=pair(a,b,y,kl_weight=0.)
            expected=.5*(torch.nn.functional.cross_entropy(a,y,label_smoothing=.01)+torch.nn.functional.cross_entropy(b,y,label_smoothing=.01))
            self.assertTrue(torch.equal(value,ce));self.assertTrue(torch.allclose(value,expected))
            value.backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(b.grad.abs().sum()),0.)

if __name__=='__main__':unittest.main()
