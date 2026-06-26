import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RaiseOnForward(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = str(name)

    def forward(self, *args, **kwargs):
        raise AssertionError(f"{self.name} should be skipped in deploy fast path")


class CVSLowLatencyFastPathTest(unittest.TestCase):
    def _build_model(self):
        from model import build_model

        torch.manual_seed(20260603)
        model = build_model(
            num_classes=8,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
        )
        model.eval()
        return model

    def test_deploy_logits_skip_auxiliary_projection_heads(self):
        model = self._build_model()
        x = torch.randn(2, 2, 128)
        with torch.no_grad():
            reference = model(x, return_aux=True)["logits"]

        model.con_proj = _RaiseOnForward("con_proj")
        model.cls_head.pa_head = _RaiseOnForward("pa_head")
        with torch.no_grad():
            fast = model(x, return_aux=False)

        self.assertTrue(torch.allclose(fast, reference, atol=1e-5, rtol=1e-5))

    def test_sinc_iq_frontend_uses_single_filterbank_call(self):
        model = self._build_model()
        x = torch.randn(3, 2, 128)
        with torch.no_grad():
            reference = torch.cat(
                [model.sinc(x[:, 0:1, :]), model.sinc(x[:, 1:2, :])],
                dim=1,
            )

        with patch.object(model.sinc, "forward", wraps=model.sinc.forward) as wrapped:
            with torch.no_grad():
                fused = model._sinc_on_iq(x)

        self.assertTrue(torch.allclose(fused, reference, atol=1e-5, rtol=1e-5))
        self.assertLessEqual(wrapped.call_count, 1)

    def test_frequency_mirror_features_pool_once(self):
        model = self._build_model()
        x = torch.randn(2, 2, 128)
        with patch("model.F.adaptive_avg_pool1d", wraps=__import__("torch").nn.functional.adaptive_avg_pool1d) as wrapped:
            with torch.no_grad():
                feat_f, rho, dac_stats, pa_stats = model._mirror_compressed_features(x)

        self.assertEqual(feat_f.shape, (2, 4, model.freq_bands))
        self.assertEqual(rho.shape, (2, 1))
        self.assertEqual(dac_stats.shape, (2, 3))
        self.assertEqual(pa_stats.shape, (2, 3))
        self.assertLessEqual(wrapped.call_count, 1)

    def test_memory_polynomial_lift_avoids_generic_pow(self):
        from model import MemoryPolynomialLift

        lift = MemoryPolynomialLift(memory_depth=3, orders=(1, 3, 5), clip=2.0)
        x = torch.randn(2, 2, 64)
        with patch("torch.pow", side_effect=AssertionError("generic torch.pow should be avoided")):
            with torch.no_grad():
                out = lift(x)

        self.assertEqual(out.shape, (2, 18, 64))

    def test_sinc_eval_filter_cache_reuses_and_invalidates(self):
        model = self._build_model()
        model.sinc.eval()
        with torch.no_grad():
            first = model.sinc._filters(device=torch.device("cpu"), dtype=torch.float32)
            second = model.sinc._filters(device=torch.device("cpu"), dtype=torch.float32)

        self.assertEqual(first.data_ptr(), second.data_ptr())

        with torch.no_grad():
            model.sinc.low_hz_.add_(1.0)
            third = model.sinc._filters(device=torch.device("cpu"), dtype=torch.float32)

        self.assertNotEqual(second.data_ptr(), third.data_ptr())

    def test_cosface_eval_weight_cache_matches_uncached_logits(self):
        from model import CosFaceHead

        torch.manual_seed(20260603)
        head = CosFaceHead(16, 5).eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            reference = head(x)
            cached = head(x)

        self.assertTrue(torch.allclose(cached, reference, atol=1e-6, rtol=1e-6))
        self.assertIsNotNone(head._norm_weight_cache)

    def test_memory_polynomial_lift_vectorized_matches_reference_and_grad(self):
        from model import MemoryPolynomialLift

        def reference_lift(x, memory_depth, orders, clip):
            xr = torch.clamp(x[:, 0:1, :], -clip, clip)
            xi = torch.clamp(x[:, 1:2, :], -clip, clip)
            outs = []
            for m in range(memory_depth):
                if m <= 0:
                    ar = xr
                    ai = xi
                else:
                    ar = torch.cat([xr.new_zeros(xr.size(0), xr.size(1), m), xr[..., :-m]], dim=-1)
                    ai = torch.cat([xi.new_zeros(xi.size(0), xi.size(1), m), xi[..., :-m]], dim=-1)
                mag2 = ar * ar + ai * ai
                mag2_safe = torch.clamp(mag2, min=1e-8)
                for p in orders:
                    if p == 1:
                        scale = torch.ones_like(mag2)
                    else:
                        scale = mag2_safe
                        for _ in range(1, (p - 1) // 2):
                            scale = scale * mag2_safe
                    outs.append(ar * scale)
                    outs.append(ai * scale)
            return torch.cat(outs, dim=1)

        for batch_size in (1, 2):
            for memory_depth, orders in ((1, (1,)), (2, (1, 3)), (4, (1, 3, 5))):
                torch.manual_seed(100 + memory_depth + batch_size)
                x_ref = torch.randn(batch_size, 2, 33, requires_grad=True)
                x_new = x_ref.detach().clone().requires_grad_(True)
                lift = MemoryPolynomialLift(memory_depth=memory_depth, orders=orders, clip=2.0)

                y_ref = reference_lift(x_ref, memory_depth, orders, 2.0)
                y_new = lift(x_new)
                self.assertTrue(torch.allclose(y_new, y_ref, atol=1e-6, rtol=1e-6))

                y_ref.square().mean().backward()
                y_new.square().mean().backward()
                self.assertTrue(torch.allclose(x_new.grad, x_ref.grad, atol=1e-6, rtol=1e-6))

    def test_frequency_mirror_features_match_full_spectrum_reference(self):
        model = self._build_model()
        x = torch.randn(2, 2, 128)

        def reference_features(m, iq):
            B, _, L = iq.shape
            eps = m.eps
            spec = m._fft_full_via_rfft_pair(iq)
            power = spec.real * spec.real + spec.imag * spec.imag
            power = torch.nan_to_num(power, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
            half = L // 2
            pos = power[:, 1:half]
            neg = torch.flip(power[:, half + 1:], dims=[-1])
            mm = min(pos.size(-1), neg.size(-1))
            pos = pos[:, :mm]
            neg = neg[:, :mm]
            pos = torch.nan_to_num(pos, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
            neg = torch.nan_to_num(neg, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
            logP_pos = torch.log1p(pos.clamp_max(1e6))
            logP_neg = torch.log1p(neg.clamp_max(1e6))
            logR = (torch.log(pos + eps) - torch.log(neg + eps)).clamp(-20.0, 20.0)
            asym = torch.abs(pos - neg) / (pos + neg + eps)
            K = max(4, m.freq_bands)
            pooled = F.adaptive_avg_pool1d(
                torch.stack([logP_pos, logP_neg, logR, asym, pos, neg], dim=1),
                K,
            )
            feat_f = torch.stack([pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :], pooled[:, 3, :]], dim=1)
            pos_lin = pooled[:, 4, :]
            neg_lin = pooled[:, 5, :]
            tot_lin = torch.nan_to_num(pos_lin + neg_lin, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(eps).clamp_max(1e6)
            hf_start = max(0, int(0.75 * K))
            hf_ratio = tot_lin[:, hf_start:].sum(dim=1, keepdim=True) / (tot_lin.sum(dim=1, keepdim=True) + eps)
            asym_hf_mean = feat_f[:, 3, hf_start:].mean(dim=1, keepdim=True)
            flatness = torch.exp(torch.mean(torch.log(tot_lin.clamp_min(eps)), dim=1, keepdim=True).clamp(-20.0, 20.0)) / (torch.mean(tot_lin, dim=1, keepdim=True) + eps)
            dac_stats = torch.cat([hf_ratio, asym_hf_mean, flatness], dim=1)
            edge_bins = max(1, int(0.20 * K))
            center_l = max(0, int(0.30 * K))
            center_r = max(center_l + 1, int(0.70 * K))
            edge_energy = tot_lin[:, -edge_bins:].sum(dim=1, keepdim=True)
            center_energy = tot_lin[:, center_l:center_r].sum(dim=1, keepdim=True) + eps
            edge_ratio = edge_energy / (tot_lin.sum(dim=1, keepdim=True) + eps)
            regrowth_ratio = edge_energy / center_energy
            mu = torch.mean(tot_lin, dim=1, keepdim=True)
            var = torch.mean((tot_lin - mu) ** 2, dim=1, keepdim=True) + eps
            spec_kurtosis = (torch.mean((tot_lin - mu).clamp(-1e3, 1e3) ** 4, dim=1, keepdim=True) / (var * var).clamp_min(eps)).clamp(0.0, 1e4)
            pa_stats = torch.cat([edge_ratio, regrowth_ratio, spec_kurtosis], dim=1)
            z = m._to_complex(iq)
            Ez2 = torch.mean(z * z, dim=-1)
            Eabs2 = torch.mean(z.real * z.real + z.imag * z.imag, dim=-1)
            rho = (torch.abs(Ez2) / (Eabs2 + eps)).unsqueeze(1)
            return feat_f, rho, dac_stats, pa_stats

        with torch.no_grad():
            actual = model._mirror_compressed_features(x)
            expected = reference_features(model, x)

        for got, want in zip(actual, expected):
            self.assertTrue(torch.allclose(got, want, atol=1e-5, rtol=1e-5))

    def test_sinc_shared_candidate_switches_build_and_forward(self):
        from model import build_model

        torch.manual_seed(20260605)
        model = build_model(
            num_classes=8,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            freq_feature_source="sinc_phase_asym",
            pa_feature_source="sinc_lowrank",
            pa_orders=(1, 5),
            use_circularity=False,
            use_freq_stats=False,
            use_pa_stats=False,
            use_aux_spectral_stats=False,
            use_freq_band_gate=False,
            channel_trim_scale=0.75,
        ).eval()
        x = torch.randn(2, 2, 128)
        with torch.no_grad():
            aux = model(x, return_aux=True)
            logits = model(x, return_aux=False)

        self.assertEqual(logits.shape, (2, 8))
        self.assertEqual(aux["logits"].shape, (2, 8))
        self.assertEqual(model.freq_feature_source, "sinc_phase_asym")
        self.assertEqual(model.pa_feature_source, "sinc_lowrank")
        self.assertEqual(tuple(model.pa_lift.orders), (1, 5))
        self.assertIsNone(aux["rho"])


if __name__ == "__main__":
    unittest.main()
