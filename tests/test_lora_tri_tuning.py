from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main import tri_tuning
from experiments.pamf_lora_main.packet_physics import chirp_samples
from experiments.pamf_ssl_research.hard_pl_objective import mechanism_loss
from experiments.pamf_ssl_research.source_receiver_extrapolation import extrapolate


class TriTuningTests(TestCase):
    def test_rejects_target_for_unselected_row_before_opening_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'selection.json', {'names': ['control']})
            with self.assertRaisesRegex(ValueError, 'source-frozen eligible'):
                tri_tuning.predict({'run_root':temp}, 'fusion192')

    def test_physical_ce_can_change_without_enabling_as_kl(self):
        a = torch.zeros(2, 10)
        args = (a, a, a, a+torch.arange(10), a, a, torch.tensor([0,1]),
                torch.ones(2,dtype=torch.bool), torch.tensor([True, False]), torch.tensor(0.))
        base, info = mechanism_loss(*args)
        modified, other = mechanism_loss(*args, physical_weight=2.)
        self.assertGreater(modified.item(), base.item())
        self.assertGreater(info['As_KL'].item(), 0.)
        self.assertEqual(other['As_KL_weight'], 0.)

    def test_extrapolation_preserves_registered_width_and_rx_constraint(self):
        z = torch.stack([torch.zeros(192), torch.ones(192)])
        ids = torch.tensor([0,1])
        same = torch.zeros(2,dtype=torch.long)
        mask = torch.ones(2,dtype=torch.bool)
        partner = ids.flip(0)
        result = extrapolate(z, same, mask, ids, partner, mask, .25, width=192)
        self.assertTrue(torch.equal(result[0], torch.full((192,), -.25)))
        with self.assertRaisesRegex(ValueError, 'different-sourceRX'):
            extrapolate(z, same, mask, same, partner, mask, .25, width=192)
        with self.assertRaises(ValueError):
            extrapolate(z, same, mask, ids, partner, mask, .25)

    def test_long_grid_retains_native_band_unavailable_to_256_points(self):
        timeline = np.arange(8192)/1_000_000
        tone = np.exp(2j*np.pi*170_000*timeline)
        low = chirp_samples(tone,samples=256)
        high = chirp_samples(tone,samples=512)
        self.assertEqual(low.shape,(6,256))
        self.assertEqual(high.shape,(6,512))
        self.assertLess(np.mean(np.abs(low)),.05)
        self.assertGreater(np.mean(np.abs(high)),.95)
