import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import experiments


class FrozenFusionFactoryTest(unittest.TestCase):
    def factory(self):
        try:return importlib.import_module('experiments.phase1_dura_edge_fullsource.replay').model_for_schema
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_edge_fullsource.replay':self.fail('paired fusion replay factory is not implemented')
            raise

    def test_factory_preserves_registered_boundary_behaviors(self):
        f=self.factory()
        model=f('phase1_dura_blendgate_v1',factory=lambda:SimpleNamespace(preserve_edges=False))
        self.assertFalse(model.preserve_edges)
        model=f('phase1_dura_edgeblend_v1',factory=lambda:SimpleNamespace(preserve_edges=False))
        self.assertTrue(model.preserve_edges)
        with self.assertRaises(ValueError):f('unregistered',factory=lambda:SimpleNamespace())

    def test_factory_works_after_runtime_replaces_package_path(self):
        f=self.factory()
        with patch.object(experiments,'__path__',[]):
            model=f('phase1_dura_edgeblend_v1',factory=lambda:SimpleNamespace(preserve_edges=False))
        self.assertTrue(model.preserve_edges)


if __name__=='__main__':unittest.main()
