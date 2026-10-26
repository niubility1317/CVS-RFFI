import copy
import unittest
import torch
from experiments.standard_ce_baselines.model import build
from experiments.standard_ce_baselines.source import validate_config


class RemovedSIGTimeTest(unittest.TestCase):
    def test_only_signature_parameters_removed_and_default_head_kept(self):
        torch.set_num_threads(2)
        seed=2026092701
        torch.manual_seed(seed);full=build('sig_time',seed).eval()
        torch.manual_seed(seed);control=build('sig_time_removed',seed).eval()
        a,b=full.state_dict(),control.state_dict()
        self.assertEqual(set(b),{k for k in a if not k.startswith('sequence_pool.')})
        self.assertTrue(all(torch.equal(v,a[k]) for k,v in b.items()))
        self.assertEqual(sum(p.numel() for p in full.parameters()),196001)
        self.assertEqual(sum(p.numel() for p in control.parameters()),177025)
        x=torch.randn(3,2,256)
        from experiments.cvs_reference_identity.model import ReferenceCVS
        with torch.no_grad():
            expected=ReferenceCVS.features(full,x)
            self.assertTrue(torch.equal(control.features(x),expected))
            self.assertTrue(torch.equal(control(x),full.classify_features(expected)))
        control.train();loss=torch.nn.functional.cross_entropy(control(x),torch.tensor([0,1,2]));loss.backward()
        self.assertIsNotNone(control.core.id_backbone.cls_head.weight.grad)
        self.assertFalse(any('sequence_pool' in n for n,_ in control.named_parameters()))

    def test_new_variant_uses_unchanged_pure_ce_contract(self):
        c=dict(method='standard_ce_baselines',variant='sig_time_removed',epochs=200,batch_size=128,
            lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,
            domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
        validate_config(c)
        for key,value in [('epochs',20),('augmentation',True),('teacher','old.pt'),('extra_losses',['KL']),('target_truth','truth.json')]:
            wrong=copy.deepcopy(c);wrong[key]=value
            with self.assertRaises(ValueError):validate_config(wrong)


if __name__=='__main__':unittest.main()
