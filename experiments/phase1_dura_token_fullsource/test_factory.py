import unittest
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_spectral_token.model import SpectralTokenRecovery
from experiments.phase1_dura_token_fullsource.replay import model_for_schema

class ModelFactoryTest(unittest.TestCase):
    def test_exact_registered_architectures(self):
        self.assertIsInstance(model_for_schema('phase1_dura_context_v1'),ContextRecovery)
        self.assertIsInstance(model_for_schema('phase1_dura_spectral_token_v1'),SpectralTokenRecovery)
        with self.assertRaises(ValueError):model_for_schema('unregistered')

if __name__=='__main__':unittest.main()
