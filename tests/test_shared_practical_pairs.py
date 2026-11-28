import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from experiments.pamf_lora_main.shared_practical_pairs_run import complex_packet, practical_pool


class SourcePracticalPairTest(unittest.TestCase):
    def test_two_iq_representations_are_identical(self):
        iq = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        np.testing.assert_array_equal(complex_packet(iq, 3), complex_packet(iq[0] + 1j*iq[1], 3))

    def test_wrong_shape_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):
            complex_packet(np.zeros((3, 2), dtype=np.float32), 3)
        with self.assertRaises(ValueError):
            complex_packet(np.array([complex(float('nan'), 0)]), 1)

    def test_source_pool_covers_scenes_and_smokes_before_remaining_pairs(self):
        channel = ModuleType('leo_practical.channel')
        channel.normalize_input = lambda x: (x, 1.)
        restoration = ModuleType('experiments.pamf_restoration.data')
        restoration.SCENES = tuple('abcdef')
        seen = []
        def corrupt(item):
            index, clean, scene, *_ = item
            seen.append(index)
            return index, clean * 2, scene
        restoration.corrupt_one = corrupt
        native = ModuleType('experiments.pamf_lora_main.w24_native_full')
        native.native_full_view = lambda *args: None
        fake_modules = {'leo_practical.channel': channel, 'experiments.pamf_restoration.data': restoration,
                        'experiments.pamf_lora_main.w24_native_full': native}
        source = np.ones((1024, 2, 256), dtype=np.float32)
        snapshots = []
        with patch.dict('sys.modules', fake_modules):
            clean, observed, counts = practical_pool(source, np.arange(1024), 'wisig',
                                                     lambda a, b: snapshots.append((len(a), len(seen))))
        self.assertEqual(snapshots, [(16, 14)])
        self.assertEqual(counts['clean'], 128)
        self.assertEqual(sum(counts.values()), 1024)
        self.assertLessEqual(max(counts[x] for x in restoration.SCENES) -
                             min(counts[x] for x in restoration.SCENES), 1)
        self.assertEqual(clean.shape, observed.shape)


if __name__ == '__main__':
    unittest.main()
