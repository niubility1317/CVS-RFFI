"""Focused tests for functional-state cache correctness and packet exclusion."""
import unittest
import torch
from .held_credit import assess, fresh_parameter_caches


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(960).reshape(6, 160).float() / 960)
        self.feature = torch.nn.Parameter(torch.ones(160) * .2)
        self._norm_weight_cache_key = None
        self._norm_weight_cache = None
    def forward(self, x):
        key = self.weight._version
        if self._norm_weight_cache_key != key:
            self._norm_weight_cache = self.weight.detach().clone()
            self._norm_weight_cache_key = key
        z = x[:, 0, :160] * self.feature
        return z @ self._norm_weight_cache.T


class Checks(unittest.TestCase):
    def test_native_cosface_versions_are_not_value_identity(self):
        import importlib.util
        import sys
        from pathlib import Path
        path = Path(__file__).resolve().parents[1]/'adv3b02_xuc/code/model.py'
        spec = importlib.util.spec_from_file_location('heldcredit_native_model_check', path)
        native = importlib.util.module_from_spec(spec)
        sys.path.insert(0,str(path.parent))
        try: spec.loader.exec_module(native)
        finally: sys.path.pop(0)
        head = native.CosFaceHead(160,6).eval()
        x = torch.eye(160)[:6]
        a = torch.eye(160)[:6].clone(); b = -a.clone()
        self.assertEqual(a._version,b._version)
        with torch.no_grad():
            first = torch.func.functional_call(head,{'weight':a},(x,),strict=True)
            stale = torch.func.functional_call(head,{'weight':b},(x,),strict=True)
            with fresh_parameter_caches(head):
                fresh = torch.func.functional_call(head,{'weight':b},(x,),strict=True)
        self.assertTrue(torch.equal(first,stale))
        self.assertTrue(torch.equal(fresh,-first))

    def test_fresh_cache_and_restore(self):
        m = Model().eval()
        x = torch.ones(2, 2, 256)
        a = {'weight': torch.zeros_like(m.weight), 'feature': m.feature.detach().clone()}
        b = {'weight': torch.ones_like(m.weight), 'feature': m.feature.detach().clone()}
        self.assertEqual(a['weight']._version, b['weight']._version)
        sentinel = object(); m._norm_weight_cache = sentinel; m._norm_weight_cache_key = -1
        with fresh_parameter_caches(m) as count, torch.no_grad():
            first = torch.func.functional_call(m, a, (x,), strict=True)
            second = torch.func.functional_call(m, b, (x,), strict=True)
        self.assertEqual(count[0], 2)
        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.all(second > 0))
        self.assertIs(m._norm_weight_cache, sentinel)
        self.assertEqual(m._norm_weight_cache_key, -1)

    def test_exclusion_and_no_live_change(self):
        torch.manual_seed(7)
        m = Model().eval(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randn(10, 2, 256); y = torch.arange(10) % 6
        rx = torch.tensor([1, 3, 4, 6, 8] * 2)
        current = [(1, int(rx[i]), 1, i, i) for i in range(10)]
        held = [(1, int(rx[i]), 1, i+20, i+20) for i in range(10)]
        base = tuple(torch.ones_like(p)*.1 for p in m.parameters())
        aux = (None, torch.ones_like(m.feature)*.02)
        before = {k: v.clone() for k,v in m.state_dict().items()}
        kwargs = dict(default_weight_name='weight', evaluation_ids=held,
                      current_ids=current, mode='HELD_L')
        r = assess(m,opt,base,aux,x,y,rx,1,**kwargs)['summary']
        self.assertEqual(r['fresh_parameter_cache_forwards'],2)
        self.assertTrue(r['excluded_current_packets'])
        self.assertFalse(opt.state)
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        for changed in (dict(evaluation_ids=current),dict(role='U_s'),dict(mode='V'),
                        dict(evaluation_ids=held[:-1]),dict(tolerance=1e-5)):
            with self.assertRaises(ValueError):
                assess(m,opt,base,aux,x,y,rx,1,**dict(kwargs,**changed))


if __name__ == '__main__': unittest.main()
