import unittest

import torch


class ArchitectureFamilyBackboneTest(unittest.TestCase):
    def test_resnet18_1d_uses_dual_training_interface(self):
        from model_dual_cvsincnet import build_dual_model

        model = build_dual_model(
            num_classes=6,
            num_domains=3,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            arch_family="resnet18_1d",
            fast_infer_when_no_aux=True,
        )
        model.eval()

        x = torch.randn(2, 2, 128)
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 2])
        with torch.no_grad():
            out = model(x, y_tx=y, domain_labels=d, return_aux=True)
            logits = model(x, return_aux=False)

        self.assertEqual(out["tx_logits"].shape, (2, 6))
        self.assertEqual(out["dom_logits"].shape, (2, 3))
        self.assertEqual(out["adv_dom_logits"].shape, (2, 3))
        self.assertEqual(out["z_id"].shape[0], 2)
        self.assertEqual(out["z_dom"].shape, out["z_id"].shape)
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(model.arch_family, "resnet18_1d")

    def test_cvcnn_uses_dual_training_interface(self):
        from model_dual_cvsincnet import build_dual_model

        model = build_dual_model(
            num_classes=6,
            num_domains=3,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            arch_family="cvcnn",
            fast_infer_when_no_aux=False,
        )
        model.eval()

        x = torch.randn(2, 2, 128)
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 2])
        with torch.no_grad():
            out = model(x, y_tx=y, domain_labels=d, return_aux=True)

        self.assertEqual(out["tx_logits"].shape, (2, 6))
        self.assertEqual(out["dom_logits"].shape, (2, 3))
        self.assertEqual(out["adv_dom_logits"].shape, (2, 3))
        self.assertEqual(out["z_id"].shape, (2, 128))
        self.assertEqual(out["z_dom"].shape, (2, 128))
        self.assertEqual(model.arch_family, "cvcnn")


if __name__ == "__main__":
    unittest.main()
