import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from experiments.phase1_dura_correlation import ground, train_source
from experiments.phase1_dura_correlation.integrated import CorrelationPosteriorInDura


class AdapterTest(unittest.TestCase):
    def test_training_uses_correlation_corrections_and_restores_base_module(self):
        original = train_source.base.PosteriorInDura
        with patch.object(sys, 'argv', ['train_source', '--posterior-kind', 'mean']), \
             patch.object(train_source.base, 'main') as target:
            target.side_effect = lambda: self.assertIs(train_source.base.PosteriorInDura, CorrelationPosteriorInDura)
            train_source.main()
        self.assertIs(train_source.base.PosteriorInDura, original)
        target.assert_called_once()

    def test_training_rejects_labeled_nonlocal_parent(self):
        for args in (['--posterior-kind', 'nonlocal'], ['--posterior-kind=nonlocal'], ['--posterior-k', 'nonlocal'], ['--p', 'nonlocal']):
            with patch.object(sys, 'argv', ['train_source', *args]), \
                 patch.object(train_source.base, 'main') as target:
                with self.assertRaises(ValueError):
                    train_source.main()
                target.assert_not_called()

    def test_ground_uses_same_architecture_and_source(self):
        fake = SimpleNamespace(PosteriorInDura=object(), SOURCE='old', NONLOCAL_SOURCE='other', main=Mock())
        original = (fake.PosteriorInDura, fake.SOURCE, fake.NONLOCAL_SOURCE)
        def check():
            self.assertIs(fake.PosteriorInDura, CorrelationPosteriorInDura)
            self.assertEqual(fake.SOURCE, ground.SOURCE)
            self.assertEqual(fake.NONLOCAL_SOURCE, ground.SOURCE)
        fake.main.side_effect = check
        with patch.object(ground, 'import_module', return_value=fake):
            ground.main()
        self.assertEqual((fake.PosteriorInDura, fake.SOURCE, fake.NONLOCAL_SOURCE), original)
        fake.main.assert_called_once()


if __name__ == '__main__':
    unittest.main()
