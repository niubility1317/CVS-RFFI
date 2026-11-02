import unittest
from unittest.mock import patch
from experiments.phase1_dura_volterra import train_source, ground, score_source


class AdapterTest(unittest.TestCase):
    def test_training_restores_all_shared_settings(self):
        base = train_source.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        old = tuple(getattr(base, name) for name in names)
        with patch.object(base, 'main') as main:
            main.side_effect = lambda: self.assertTrue(base.USES_DURA and base.source_loss is train_source.source_loss
                                                       and base.predict_batch is train_source.predict_batch)
            train_source.main()
        self.assertEqual(tuple(getattr(base, name) for name in names), old)

    def test_ground_restores_shared_settings(self):
        base = ground.base
        names = ('DynamicVolterra', 'SOURCE', 'SCHEMA', 'LOG_PREFIX', 'USES_DURA', 'load_components', 'predict_batch')
        old = tuple(getattr(base, name) for name in names)
        with patch.object(base, 'main') as main:
            main.side_effect = lambda: self.assertTrue(base.USES_DURA and base.SOURCE == ground.SOURCE
                                                       and base.predict_batch is ground.predict_batch)
            ground.main()
        self.assertEqual(tuple(getattr(base, name) for name in names), old)

    def test_source_scorer_restores_shared_settings(self):
        base = score_source.base; old = base.SCHEMA, base.USES_DURA
        with patch.object(base, 'main') as main:
            main.side_effect = lambda: self.assertEqual((base.SCHEMA, base.USES_DURA), ('phase1_dura_volterra_v1', True))
            score_source.main()
        self.assertEqual((base.SCHEMA, base.USES_DURA), old)


if __name__ == '__main__': unittest.main()
