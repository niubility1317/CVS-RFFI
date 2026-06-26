import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _count_unique_params(model) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        ident = id(p)
        if ident in seen:
            continue
        seen.add(ident)
        total += int(p.numel())
    return total


class Cen31StudentModelTest(unittest.TestCase):
    def _build(self, variant: str, branch_ablation: str = "no_dac"):
        from model_dual_cvsincnet import build_dual_model

        model = build_dual_model(
            num_classes=6,
            num_domains=4,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant=variant,
            branch_ablation=branch_ablation,
            domain_branch_ablation="no_stats",
            domain_enhancer="rcn_stats",
            domain_enhancer_strength=0.20,
            fast_infer_when_no_aux=True,
        )
        model.eval()
        return model

    def test_cen31_student_variants_are_smaller_than_lite_d_teacher_shape(self):
        teacher_shape = self._build("lite_d", "no_dac")
        candidates = {
            "lite_f": self._build("lite_f", "no_dac,no_stats"),
            "lite_g": self._build("lite_g", "no_dac,no_pa,no_stats"),
            "lite_h": self._build("lite_h", "time_only"),
        }

        teacher_params = _count_unique_params(teacher_shape)
        for name, model in candidates.items():
            with self.subTest(name=name):
                self.assertLess(_count_unique_params(model), teacher_params)
                x = torch.randn(2, 2, 128)
                y = torch.tensor([0, 1])
                d = torch.tensor([0, 2])
                with torch.no_grad():
                    out = model(x, y_tx=y, domain_labels=d, return_aux=True)
                    logits = model(x, return_aux=False)
                self.assertEqual(out["tx_logits"].shape, (2, 6))
                self.assertEqual(out["dom_logits"].shape, (2, 4))
                self.assertEqual(out["adv_dom_logits"].shape, (2, 4))
                self.assertEqual(out["z_id"].shape[0], 2)
                self.assertEqual(logits.shape, (2, 6))

    def test_lite_h_time_only_removes_frequency_and_pa_latency_paths(self):
        model = self._build("lite_h", "time_only")
        self.assertTrue(model.id_backbone.use_time_path)
        self.assertFalse(model.id_backbone.use_freq_path)
        self.assertFalse(model.id_backbone.use_pa_path)
        self.assertFalse(model.id_backbone.use_dac_path)

    def test_deploy_forward_skips_student_domain_backbone(self):
        model = self._build("lite_g", "no_dac,no_pa,no_stats")
        x = torch.randn(2, 2, 128)
        with patch.object(model.dom_backbone, "forward", side_effect=AssertionError("domain path should be skipped")):
            with torch.no_grad():
                logits = model(x, return_aux=False)
        self.assertEqual(logits.shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
