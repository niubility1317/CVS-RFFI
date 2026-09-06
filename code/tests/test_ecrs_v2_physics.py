import math
import unittest
import torch
import torch.nn.functional as F
from cvsrffi.ecrs_v2 import (schur_ridge, ECRSV2PhysicalEstimator, ECRSV2Branch,
    FixedOperatorCache, make_anchor_states, tangent_fusion, build_response_basis)


class ECRSV2PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)

    def test_schur_matches_joint_with_real_nuisance_and_masks(self):
        y, n, p = torch.randn(3, 42, dtype=torch.float64), torch.randn(3, 42, 3, dtype=torch.float64), torch.randn(3, 42, 16, dtype=torch.float64)
        mask = torch.arange(42).remainder(3).ne(0).expand(3, -1)
        fit = schur_ridge(y, n, p, fit_mask=mask, diagnostics=True)
        d = torch.cat((n, p), -1)
        scale = ((d.square()*mask[..., None]).sum(1)/mask.sum(1)[:, None]).sqrt()
        dn = d/scale[:, None]
        gram = dn.transpose(1, 2)@(dn*mask[..., None]) + .01*torch.eye(19, dtype=y.dtype)
        joint = torch.linalg.solve(gram, dn.transpose(1, 2)@(y*mask)[..., None]).squeeze(-1)/scale
        torch.testing.assert_close(fit['eta'], joint[:, :3], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(fit['theta'], joint[:, 3:], atol=1e-10, rtol=1e-10)
        self.assertEqual(fit['uncertainty_kind'], 'ridge_sensitivity')

    def test_fixed_cache_dynamic_parity_and_all_invalidation(self):
        y = torch.randn(2, 20, dtype=torch.float64)
        n, p = torch.randn(1, 20, 3, dtype=y.dtype), torch.randn(1, 20, 6, dtype=y.dtype)
        cache = FixedOperatorCache()
        def compare(version='v1', **kwargs):
            actual = cache.solve(y, n, p, reference_version=version, **kwargs)
            expected = schur_ridge(y, n.expand(2,-1,-1), p.expand(2,-1,-1), **kwargs)
            torch.testing.assert_close(actual['theta'], expected['theta'], atol=1e-10, rtol=1e-10)
        compare(); compare()
        self.assertEqual((cache.misses, cache.hits), (1, 1))
        p.add_(.01); compare()
        n.mul_(.99); compare()
        compare('v2'); compare('v2', alpha_theta=.02)
        compare('v2', weights=torch.ones(1,20, dtype=y.dtype)*.5)
        self.assertEqual(cache.misses, 6)

    def test_crossfit_no_evaluation_contamination(self):
        x = torch.randn(2, 2, 64)
        physical = ECRSV2PhysicalEstimator()
        original = physical.cross_fit(x)[0]
        modified = x.clone(); modified[:, :, 34:] = torch.randn_like(modified[:, :, 34:])*100
        other = physical.cross_fit(modified)[0]
        torch.testing.assert_close(original['fit']['resp_coef'], other['fit']['resp_coef'], atol=0, rtol=0)
        torch.testing.assert_close(original['fit']['diagnostics']['reference'], other['fit']['diagnostics']['reference'], atol=0, rtol=0)
        self.assertFalse(torch.equal(original['nmse_full'], other['nmse_full']))
        self.assertEqual(original['effective_points'], 30)
        with self.assertRaises(ValueError): physical.cross_fit(x, guard=0)

    def test_conjugate_blindness_and_explicit_history(self):
        s, h = make_anchor_states('real8')
        torch.testing.assert_close(s, s.conj())
        c, history = make_anchor_states('complex24')
        self.assertGreater(float((c-c.conj()).abs().max()), 1.)
        torch.testing.assert_close(history[:12], c[:12])
        self.assertEqual(float(history[12:].abs().sum()), 0.)
        order = torch.randperm(24)
        torch.testing.assert_close(build_response_basis(c, history)[order], build_response_basis(c[order], history[order]))

    def test_zero_and_nonfinite_are_finite_invalid(self):
        branch = ECRSV2Branch()
        x = torch.zeros(3, 2, 32); x[1,0,4] = float('nan'); x[2,1,7] = float('inf')
        out = branch(x, torch.zeros(3,160), return_diagnostics=True)
        for key in ['z_resp','z_id_fused','resp_coef','resp_anchor','fusion_quality_features']:
            self.assertTrue(torch.isfinite(out[key]).all(), key)
        for value in out['resp_quality'].values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(out['quality_valid'].any())
        self.assertEqual(float(out['z_resp'].detach().abs().sum()), 0.)

    def test_tangent_bound_zero_projection_and_encoder_gradient(self):
        raw = torch.randn(4, 160, requires_grad=True)
        resp = torch.randn(4, 160)*100
        fused = tangent_fusion(raw, resp, .05)
        angle = torch.acos((F.normalize(raw, dim=-1)*fused).sum(-1).clamp(-1,1))
        self.assertLessEqual(float(angle.detach().max()), math.atan(.05)+1e-5)
        branch = ECRSV2Branch(fusion_mode='fixed')
        iq = torch.randn(4,2,64, requires_grad=True)
        out = branch(iq, raw)
        torch.testing.assert_close(out['z_id_fused'], F.normalize(raw,dim=-1))
        out['z_id_fused'][:, 0].sum().backward()
        self.assertGreater(float(branch.response_projection.weight.grad.norm()), 0)
        self.assertIsNone(raw.grad)
        self.assertIsNone(iq.grad)
        branch.zero_grad()
        out = branch(iq)
        out['z_resp'][:,0].sum().backward()
        self.assertGreater(float(branch.encoder[0].weight.grad.norm()), 0)
        self.assertIsNone(iq.grad)

    def test_public_reference_cache_and_bundle_roundtrip(self):
        ref = torch.randn(32, dtype=torch.complex64)
        branch = ECRSV2Branch(reference_mode='public_reference', public_reference=ref, fusion_mode='fixed')
        x = torch.randn(2,2,32)
        a = branch(x)['resp_coef']; b = branch.physical(x, use_cache=False)['resp_coef']
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
        branch.set_active_fusion(0)
        clone = ECRSV2Branch(reference_mode='public_reference', public_reference=ref, fusion_mode='fixed')
        clone.load_state_dict(branch.export_bundle()['state_dict'])
        self.assertEqual(float(clone.active_rho), 0.)
        torch.testing.assert_close(clone.physical.public_reference, ref)

    def test_overlap_pruning_and_sensitivity_semantics(self):
        ref = torch.ones(32, dtype=torch.complex64)
        physical = ECRSV2PhysicalEstimator(reference_mode='public_reference', public_reference=ref)
        out = physical(torch.randn(2,2,32), return_diagnostics=True)
        # Constant excitation cannot identify current cubic versus real gain/phase.
        self.assertFalse(out['diagnostics']['response_active_real_columns'][:,0].any())
        self.assertTrue(torch.isfinite(out['diagnostics']['anchor_sensitivity_diagonal']).all())
        y, n, p = torch.randn(2,20), torch.randn(2,20,3), torch.randn(2,20,4)
        fit = schur_ridge(y,n,p,diagnostics=True,noise_variance=.3)
        torch.testing.assert_close(fit['response_posterior_covariance'], .3*fit['response_sensitivity'])
        self.assertEqual(fit['uncertainty_kind'], 'iid_gaussian_conditional_posterior')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_keeps_physics_fp32_and_encoder_trainable(self):
        branch = ECRSV2Branch(fusion_mode='fixed').cuda()
        iq, raw = torch.randn(4,2,64,device='cuda'), torch.randn(4,160,device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            out = branch(iq,raw)
            loss = out['z_resp'][:,0].float().sum() + out['z_id_fused'][:,0].float().sum()
        loss.backward()
        self.assertEqual(out['resp_coef'].dtype, torch.complex64)
        self.assertTrue(torch.isfinite(branch.encoder[0].weight.grad).all())
        self.assertGreater(float(branch.encoder[0].weight.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()
