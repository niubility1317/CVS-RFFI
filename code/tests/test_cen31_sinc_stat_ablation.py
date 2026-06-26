import unittest

import torch


class CEN31SincStatisticAblationTest(unittest.TestCase):
    def test_sinc_frequency_and_pa_lowrank_forward(self):
        from model import build_model

        model = build_model(
            num_classes=6,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            freq_feature_source="sinc_energy",
            pa_feature_source="sinc_lowrank",
            pa_orders=(1, 5),
        ).eval()
        x = torch.randn(2, 2, 128)
        with torch.no_grad():
            aux = model(x, return_aux=True)
            logits = model(x, return_aux=False)

        self.assertEqual(aux["logits"].shape, (2, 6))
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(model.freq_feature_source, "sinc_energy")
        self.assertEqual(model.pa_feature_source, "sinc_lowrank")
        self.assertEqual(model.pa_lift.orders, (1, 5))

    def test_statistic_ablation_switches_build_and_forward(self):
        from model import build_model

        model = build_model(
            num_classes=6,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            use_circularity=False,
            use_freq_stats=False,
            use_pa_stats=False,
            use_freq_band_gate=False,
        ).eval()
        x = torch.randn(2, 2, 128)
        with torch.no_grad():
            aux = model(x, return_aux=True)
            logits = model(x, return_aux=False)

        self.assertEqual(aux["logits"].shape, (2, 6))
        self.assertEqual(logits.shape, (2, 6))
        self.assertIsNone(model.freq_stats_proj)
        self.assertIsNone(model.pa_stats_proj)
        self.assertFalse(model.use_circularity)

    def test_dual_model_accepts_minimal_rcn_stats(self):
        from model_dual_cvsincnet import build_dual_model

        model = build_dual_model(
            num_classes=6,
            num_domains=4,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            domain_branch_ablation="no_stats",
            domain_enhancer="rcn_minimal_6stats",
            freq_feature_source="sinc_energy",
            pa_feature_source="sinc_lowrank",
            pa_orders=(1, 5),
        ).eval()
        x = torch.randn(2, 2, 128)
        y = torch.tensor([0, 1])
        with torch.no_grad():
            aux = model(x, y_tx=y, return_aux=True)
            logits = model(x, y_tx=y, return_aux=False)

        self.assertEqual(aux["tx_logits"].shape, (2, 6))
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(model.dom_enhancer.stat_encoder.stat_dim, 6)


if __name__ == "__main__":
    unittest.main()
