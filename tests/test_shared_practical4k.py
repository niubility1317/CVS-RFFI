import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from experiments.pamf_lora_main.shared_practical4k_run import practical_pool


class SourcePracticalFourKTest(unittest.TestCase):
    def test_source_pool_size_balance_and_early_smoke(self):
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
        fake = {'leo_practical.channel': channel, 'experiments.pamf_restoration.data': restoration,
                'experiments.pamf_lora_main.w24_native_full': native}
        source = np.ones((4096, 2, 256), dtype=np.float32)
        smoke = []
        with patch.dict('sys.modules', fake):
            clean, observed, counts = practical_pool(source, np.arange(4096), 'wisig',
                                                     lambda a, b: smoke.append((len(a), len(seen))))
        self.assertEqual(smoke, [(16, 14)])
        self.assertEqual(len(clean), 4096)
        self.assertEqual(clean.shape, observed.shape)
        self.assertEqual(counts['clean'], 512)
        self.assertEqual(sum(counts.values()), 4096)
        self.assertLessEqual(max(counts[x] for x in restoration.SCENES) -
                             min(counts[x] for x in restoration.SCENES), 1)


if __name__ == '__main__':
    unittest.main()
