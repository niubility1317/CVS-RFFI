import unittest
import experiments
from unittest.mock import patch
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_robust_fullsource.replay import model_for_schema

class ModelFactoryTest(unittest.TestCase):
    def test_only_registered_architecture_schemas(self):
        for schema in ('phase1_dura_context_v1', 'phase1_dura_robust_margin_v1'):
            self.assertIsInstance(model_for_schema(schema), ContextRecovery)
        with self.assertRaises(ValueError): model_for_schema('unregistered')

    def test_factory_survives_bootstrap_replacing_package_path(self):
        with patch.object(experiments, '__path__', []):
            model = model_for_schema('phase1_dura_robust_margin_v1')
        self.assertIsInstance(model, ContextRecovery)

if __name__ == '__main__': unittest.main()
