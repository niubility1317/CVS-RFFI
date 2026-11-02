import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import experiments


class FrozenComplexFactoryTest(unittest.TestCase):
    def factory(self):
        try:return importlib.import_module('experiments.phase1_dura_complex_fullsource.replay').model_for_schema
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_complex_fullsource.replay':self.fail('complex paired replay factory is not implemented')
            raise

    def test_factory_selects_correct_frozen_architecture(self):
        f=self.factory();kw=dict(edge_factory=lambda:SimpleNamespace(kind='edge'),complex_factory=lambda:SimpleNamespace(kind='complex'))
        self.assertEqual(f('phase1_dura_edgeblend_v1',**kw).kind,'edge')
        self.assertEqual(f('phase1_dura_complexblend_v1',**kw).kind,'complex')
        with self.assertRaises(ValueError):f('unregistered',**kw)

    def test_factory_survives_runtime_package_path_reset(self):
        f=self.factory();kw=dict(edge_factory=lambda:SimpleNamespace(kind='edge'),complex_factory=lambda:SimpleNamespace(kind='complex'))
        with patch.object(experiments,'__path__',[]):
            self.assertEqual(f('phase1_dura_complexblend_v1',**kw).kind,'complex')


if __name__=='__main__':unittest.main()
