import unittest
from unittest.mock import patch

from experiments.phase1_spectral_margin import train_source, ground, score_source
from experiments.phase1_spectral_coupling.model import SpectralCoupling


class AdapterTest(unittest.TestCase):
    def test_trainer_uses_margin_objective_and_restores_shared_trainer(self):
        base = train_source.base; old = base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss
        with patch.object(base, 'main') as main:
            main.side_effect = lambda: self.assertEqual((base.DynamicVolterra, base.SCHEMA, base.source_loss),
                                                         (SpectralCoupling, 'phase1_spectral_margin_v1', train_source.source_loss))
            train_source.main()
        self.assertEqual((base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss), old)

    def test_ground_uses_matching_model_and_source(self):
        base = ground.base; old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
        with patch.object(base, 'main') as main:
            main.side_effect = lambda: self.assertEqual((base.DynamicVolterra, base.SOURCE, base.SCHEMA),
                                                         (SpectralCoupling, ground.SOURCE, 'phase1_spectral_margin_v1'))
            ground.main()
        self.assertEqual((base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX), old)

    def test_source_scorer_restores_schema(self):
        base = score_source.base; old = base.SCHEMA
        with patch.object(base, 'main') as main:
            main.side_effect = lambda: self.assertEqual(base.SCHEMA, 'phase1_spectral_margin_v1')
            score_source.main()
        self.assertEqual(base.SCHEMA, old)


if __name__ == '__main__': unittest.main()
