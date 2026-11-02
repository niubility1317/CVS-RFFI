import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from experiments.phase1_dura_posterior.train_source import prepare_source, fixed_source_predictions, VIEWS


class Gate(nn.Module):
    def forward(self, x): return x[:, 0, 0]


class Physical(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, x):
        self.count += len(x)
        return x+1, torch.zeros(len(x))


class Reader(nn.Module):
    def features(self, x): return x.flatten(1)
    def classify_features(self, x): return torch.stack((x[:, 0], -x[:, 0]), dim=1)


class Dura(nn.Module):
    def forward(self, x): return x*0.5


class Source:
    def __init__(self, x, scene): self.x, self.scene = x, scene
    def epoch_data(self, cache, epoch): return self.x, self.scene


class Pipe(nn.Module):
    def __init__(self):
        super().__init__()
        self.integrated = nn.Module()
        self.integrated.dura = Dura()

    def forward(self, x):
        return self.integrated.dura(x), torch.zeros(len(x), dtype=torch.bool)


class SourceTrainingTest(unittest.TestCase):
    def test_preparation_only_computes_posterior_for_open_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh = root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
            (fresh/'train').mkdir(parents=True)
            (fresh/'train/physical_ids.npy').touch()
            for epoch in (21, 22, 23):
                path = fresh/'epochs'/f'{epoch:02d}'
                path.mkdir(parents=True)
                info = dict(count=63000, namespace=f'learned-action/train/epoch{epoch:02d}',
                            TX_labels_read=0, F_outputs_read=0, target_reads=0)
                (path/'complete.json').write_text(json.dumps(info), encoding='utf8')
            x = np.zeros((6, 2, 256), dtype=np.float32)
            x[:, 0, 0] = [-3, 2, -3, 3, -3, -3]
            clean = np.ones_like(x)
            scene = np.array(['clean', 'clean', 'practical_low_urban', 'practical_low_urban', 'clean', 'clean'])
            physical = Physical()
            data, counts = prepare_source(root, fresh, np.arange(6), clean, np.array([4, 6, 4, 6, 4, 6]),
                                          Gate(), physical, Dura(), Reader(), 'cpu', Source(x, scene))
            self.assertEqual(physical.count, 6)
            self.assertEqual(counts, {'clean': 3, 'practical_low_urban': 3})
            self.assertEqual(len(data[0]), 6)
            self.assertTrue(torch.equal(data[-1], torch.full((6,), 6)))
            torch.testing.assert_close(data[2], data[0]+1)

    def test_predictions_keep_all_receiver_scene_strata(self):
        scene = np.asarray(list(VIEWS)*4)
        x = np.ones((28, 2, 256), dtype=np.float32)
        clean = x.copy()
        rx = np.asarray([4]*14+[8]*14)
        ids, split, actual_rx, actual_scene, values = fixed_source_predictions(
            Path('.'), Path('.'), clean, rx, np.arange(14), np.arange(14, 28),
            np.array([], dtype=np.int64), Pipe(), Reader(), 'cpu', True, Source(x, scene))
        self.assertEqual(values.shape, (28, 9))
        self.assertEqual(len(np.unique(ids)), 28)
        self.assertEqual(np.count_nonzero(split == 'RX4_6'), 14)
        self.assertEqual(np.count_nonzero(split == 'RX8'), 14)
        self.assertEqual(np.count_nonzero(values[:, 2]), 0)
        self.assertTrue(np.array_equal(rx[ids], actual_rx))
        self.assertTrue(np.array_equal(scene[ids], actual_scene))


if __name__ == '__main__':
    unittest.main()
