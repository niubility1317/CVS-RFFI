from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import tri_tuning
from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, padded_crop
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.full_tri_repl_lora import clean_source_ce
from experiments.pamf_lora_main.packet_physics import chirp_samples
from experiments.pamf_lora_main.tri_branch_shapes import retain_projection
from experiments.pamf_lora_main import tri_clean95
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.repair_features import packet_phase_curvature
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

    def test_wider_branch_pool_preserves_the_projected_mean_at_initialization(self):
        layer = torch.nn.Linear(3,5)
        values = torch.arange(48,dtype=torch.float32).reshape(2,3,8)
        old = layer(values.mean(dim=2))
        new = retain_projection(layer)(torch.nn.functional.adaptive_avg_pool1d(values,4).flatten(1))
        torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-5)

    def test_vectorized_clean_grid_matches_the_registered_packet_readout(self):
        rng = np.random.default_rng(26)
        x = (rng.normal(size=(3,8192))+1j*rng.normal(size=(3,8192))).astype(np.complex64)
        for n in (192,256,384,512):
            actual = tri_clean95.clean_iq(x,n)
            expected = np.stack([chirp_windows(raw,samples=n) for raw in x])
            np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)

    def test_clean_predictor_blocks_unselected_rows_without_target_handle(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'selection.json', {'names':['control']})
            with self.assertRaisesRegex(ValueError, 'source-frozen'):
                tri_clean95.predict({'run_root':temp},'iq512')

    def test_robust_study_only_predicts_for_the_source_champion_at_98_percent(self):
        for n, correct, expected in ((12000,11759,[]), (11999,11760,[]),
                                     (12000,11760,['joint_rx'])):
            with self.subTest(n=n, correct=correct), TemporaryDirectory() as temp:
                root=Path(temp)/'run'
                rt={'run_root':str(root),'scratch':True,'robust_study':True,
                    'fixed_clean_cache':'source-only'}
                predicted=[]

                def schedule(_,jobs,launch):
                    for kind,name in jobs:
                        if kind=='predict':
                            predicted.append(name)
                            continue
                        record={'n':n,'correct':correct if name=='joint_rx' else 11478,'nll':.1}
                        write(root/'F'/name/'complete.json',{'frozen':record})

                with patch.object(tri_clean95,'start',return_value=Mock(wait=lambda:0)), \
                     patch.object(tri_clean95,'schedule',side_effect=schedule):
                    tri_clean95.dispatch(rt)
                self.assertEqual(predicted,expected)
                self.assertEqual(read(root/'selection.json')['names'],expected)

    def test_packet_phase_feature_removes_common_phase_and_linear_drift(self):
        t=torch.arange(8,dtype=torch.float64)
        base=torch.polar(1+.2*t,torch.sin(t)).unsqueeze(0)
        rotated=base*torch.polar(torch.ones_like(t),.4+.2*t)
        actual=packet_phase_curvature(rotated)
        self.assertEqual(actual.shape,(1,12))
        torch.testing.assert_close(actual,packet_phase_curvature(base))

    def test_timing_views_use_neighbor_samples_within_every_physical_chirp(self):
        center=padded_crop(ALL_CHIRPS)[:,125:1125]
        for offset in (-2,2):
            shifted=padded_crop(ALL_CHIRPS,offset)[:,125:1125]
            np.testing.assert_array_equal(shifted,center+offset)
            self.assertTrue((shifted[:,0]>=np.arange(8)*1024).all())
            self.assertTrue((shifted[:,-1]<(np.arange(8)+1)*1024).all())

    def test_rtl_weight_applies_only_to_the_labeled_source_receiver(self):
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'rtl_study':True})
        self.assertEqual((epochs,u_batch),(112,16))
        self.assertEqual(rows['phase_rtl']['rtl_weight'],2.)
        t=torch.zeros(3,10)
        t[1,1]=5.
        labels=torch.zeros(3,dtype=torch.long)
        rx=torch.tensor([0,SOURCE.index('rtl_1'),2])
        ce=torch.nn.functional.cross_entropy(t,labels,label_smoothing=.01,reduction='none')
        torch.testing.assert_close(clean_source_ce(t,labels,rx),ce.mean())
        torch.testing.assert_close(clean_source_ce(t,labels,rx,2.),(ce.sum()+ce[1])/4)
