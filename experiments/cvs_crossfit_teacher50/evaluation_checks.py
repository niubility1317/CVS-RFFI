import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d,gate,evaluate
from .model import build
from experiments.cvs_phase1_stack.runtime import installed


class Checks(unittest.TestCase):
    def test_real_native_strong_train_dropout_matches_clone_without_changing_frozen_state(self):
        torch.set_num_threads(2);model=build(d.config(d.rows()[0]),'cpu').eval().requires_grad_(False)
        clone=copy.deepcopy(model).train();before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(12,2,256);domains=torch.arange(12)%15;modes={m:m.training for m in model.modules()}
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(173);expected=clone(x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=domains)['tx_logits']
            torch.manual_seed(173);actual=gate.strong_forward(model,x,domains)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=1e-5))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(m.training==mode for m,mode in modes.items()));self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_actual_native_gate_masks_replay_and_forbid_TX_metadata(self):
        args=d.parent.make_args(d.operational_parent(),'cpu');weak=torch.randn(6,6)*4;strong=torch.randn(6,6)
        meta={k:torch.tensor(v) for k,v in dict(rx_i=[1,1,1,3,3,3],day_i=[1]*6,eq_i=[1]*6,
            sig_i=[1,2,8,1,2,8],base_index=[1,2,8,101,102,108]).items()}
        with installed(d.operational_parent()) as native:
            actual=gate.batch_masks(native,args,weak,strong,meta);conf,pseudo=weak.softmax(1).max(1);domain=meta['rx_i']*10+meta['day_i']
            expected=native._threshold_mask(conf,domain,args)&native._temporal_mask_tensor(pseudo,conf,(domain,meta),args,'cpu')&(strong.argmax(1)==pseudo)
            self.assertTrue(torch.equal(actual['accepted'],expected))
            with self.assertRaises(ValueError):gate.batch_masks(native,args,weak,strong,dict(meta,tx_i=torch.arange(6)))
            # Numeric domain labels are irrelevant; the RX/day partition is fixed.
            self.assertTrue(torch.equal(native._threshold_mask(conf,domain,args),native._threshold_mask(conf,domain*17+5,args)))

    def test_source_corrections_damage_true_probability_and_empty_region(self):
        y=torch.tensor([0,1,2,3]);a=torch.full((4,6),-3.);b=a.clone()
        a[0,1]=8;a[1,1]=8;a[2,2]=8;a[3,3]=8
        b[0,0]=8;b[1,0]=8;b[2,2]=8;b[3,3]=8
        result=evaluate.summarize(a,b,y,torch.tensor([True,True,False,False]))
        self.assertEqual((result['repairs'],result['new_errors'],result['net']),(1,1,0))
        self.assertEqual(result['sample_accuracy'],.5);self.assertEqual(result['excluded_accuracy'],.5)
        self.assertEqual(result['sample_highconfidence_wrong'],.5)
        self.assertEqual(evaluate.summarize(a,b,y,torch.zeros(4,dtype=torch.bool))['count'],0)

    def test_existing_source_predictions_reject_before_loading_model(self):
        c=d.config(d.rows()[0])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'source_evaluation').mkdir();c=dict(c,output_root=str(root/'source'))
            with patch.object(d,'config',return_value=c),patch.object(evaluate,'own_model',side_effect=AssertionError('Must not load')):
                with self.assertRaises(FileExistsError):evaluate.predict(d.rows()[0]['row_id'])


if __name__=='__main__':unittest.main()
