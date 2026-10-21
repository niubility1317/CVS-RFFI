import importlib.util
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).parent

def module(name, path=None):
    path = path or HERE/(name+'.py')
    spec = importlib.util.spec_from_file_location('mix_test_'+name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

class MixTests(unittest.TestCase):
    def test_balanced_rotating_membership_has_no_rng_side_effect(self):
        policy = module('policy')
        state = torch.random.get_rng_state()
        a = policy.restored_mask(64, 1, 'cpu')
        b = policy.restored_mask(64, 2, 'cpu')
        self.assertEqual(int(a.sum()), 32)
        self.assertTrue(torch.equal(a, ~b))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for size, step in ((32, 1), (64, 0), (128, 2)):
            with self.assertRaises(ValueError):
                policy.restored_mask(size, step, 'cpu')

    def test_mixing_preserves_every_original_slot_and_detaches_G(self):
        policy = module('policy')
        class Engine:
            def __init__(self): self.sizes=[]
            @torch.inference_mode()
            def __call__(self, x):
                self.sizes.append(len(x))
                return (x*.25,)
        engine = Engine()
        x = torch.randn(64,2,256)
        before=x.clone()
        y, mask = policy.mix(engine, x, 1)
        self.assertEqual(engine.sizes, [32])
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(y[~mask], x[~mask]))
        self.assertTrue(torch.equal(y[mask], x[mask]*.25))
        self.assertFalse(torch.is_inference(y))
        self.assertFalse(y.requires_grad)
        head = torch.nn.Linear(256,6)
        head(y).square().mean().backward()
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_bad_G_fails_without_fallback(self):
        policy = module('policy')
        for engine in (lambda x:(x*float('nan'),), lambda x:(x[:1],)):
            with self.assertRaises((ValueError,FloatingPointError)):
                policy.mix(engine, torch.ones(64,2,256),1)

    def test_restored_maps_receive_only_cross_RX_same_TX_positive_geometry(self):
        relations=module('relations', HERE.parent/'cvs_crossview_supcon41/relations.py')
        # Layout ensures all TXs occur across different source receivers.
        y=torch.arange(128)%6
        rx=torch.tensor([1,3,4,6,8])[(torch.arange(128)//6)%5]
        mask=torch.arange(128)%2==0
        a={k:torch.randn(128,4,8,requires_grad=True) for k in relations.LEVELS}
        b={k:torch.randn(128,4,8,requires_grad=True) for k in relations.LEVELS}
        value,records=relations.penalty(a,b,y,rx,mask,'SAT192')
        value.backward()
        for k in relations.LEVELS:
            self.assertGreater(float(b[k].grad[mask].abs().sum()),0)
            self.assertEqual(float(b[k].grad[~mask].abs().sum()),0)
        for r in records:
            self.assertEqual(r['total_anchors'],192)
            self.assertEqual(r['same_RX_positive_pairs'],0)
            self.assertGreater(r['cross_view_positive_pairs'],0)
        _, records=relations.penalty(a,b,y,rx,mask,'ORD128')
        self.assertTrue(all(r['additional_anchors']==0 for r in records))

if __name__=='__main__': unittest.main()
