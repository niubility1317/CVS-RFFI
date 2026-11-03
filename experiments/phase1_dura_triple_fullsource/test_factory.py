import importlib
import unittest
import torch
from types import SimpleNamespace
from unittest.mock import patch
import experiments


class FrozenTripleFactoryTest(unittest.TestCase):
    def factory(self):
        try:return importlib.import_module('experiments.phase1_dura_triple_fullsource.replay').model_for_schema
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_triple_fullsource.replay':self.fail('triple paired replay factory is not implemented')
            raise

    def test_factory_selects_correct_frozen_architecture(self):
        f=self.factory();kw=dict(edge_factory=lambda:SimpleNamespace(kind='edge'),triple_factory=lambda:SimpleNamespace(kind='triple'))
        self.assertEqual(f('phase1_dura_edgeblend_v1',**kw).kind,'edge')
        self.assertEqual(f('phase1_dura_tripleblend_v1',**kw).kind,'triple')
        with self.assertRaises(ValueError):f('unregistered',**kw)

    def test_factory_survives_runtime_package_path_reset(self):
        f=self.factory();kw=dict(edge_factory=lambda:SimpleNamespace(kind='edge'),triple_factory=lambda:SimpleNamespace(kind='triple'))
        with patch.object(experiments,'__path__',[]):
            self.assertEqual(f('phase1_dura_tripleblend_v1',**kw).kind,'triple')


    def test_pair_uses_same_window_context_only_for_triple(self):
        replay=importlib.import_module('experiments.phase1_dura_triple_fullsource.replay')
        received=torch.zeros(2,2,256);anchor=torch.ones_like(received);lv=torch.zeros(2)
        class Reader:
            def features(self, x):
                self.assert_input=x
                return torch.ones(2,160)
            def classify_features(self, x):return torch.full((2,6),2.)
        def edge(x, a, variance):
            self.assertEqual(a.shape,(2,2,256));self.assertTrue(torch.equal(a,anchor));return a
        def triple(x, packed, variance):
            self.assertEqual(packed.shape,(2,3,256))
            self.assertTrue(torch.equal(packed[:,:2],anchor))
            self.assertTrue(torch.equal(packed[:,2,:160],torch.ones(2,160)))
            self.assertTrue(torch.equal(packed[:,2,160:166],torch.full((2,6),2.)))
            return packed[:,:2]
        result=replay.fused_active_outputs([edge,triple],received,anchor,lv,Reader())
        self.assertTrue(all(torch.equal(x,anchor) for x in result))

if __name__=='__main__':unittest.main()
