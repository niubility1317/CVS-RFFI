import importlib
import unittest
from unittest.mock import patch
import experiments
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_flow.model import DuraFlow


class FrozenFlowFactoryTest(unittest.TestCase):
    def factory(self):
        try:return importlib.import_module('experiments.phase1_dura_flow_fullsource.replay').model_for_schema
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_flow_fullsource.replay':self.fail('flow paired replay factory is not implemented')
            raise

    def test_factory_selects_registered_architectures(self):
        f=self.factory()
        self.assertIsInstance(f('phase1_dura_context_v1'),ContextRecovery)
        self.assertIsInstance(f('phase1_dura_flow_v1'),DuraFlow)
        with self.assertRaises(ValueError):f('unregistered')

    def test_factory_survives_runtime_path_reset(self):
        f=self.factory()
        with patch.object(experiments,'__path__',[]):
            model=f('phase1_dura_flow_v1')
        self.assertIsInstance(model,DuraFlow)


if __name__=='__main__':unittest.main()
