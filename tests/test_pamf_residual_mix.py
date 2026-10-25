import unittest

import torch

from experiments.pamf_channel_pl.residual_mix import mix_iq,paired_views


class ResidualMixTests(unittest.TestCase):
    def test_exact_original_and_deployed_restoration_endpoints(self):
        torch.manual_seed(131)
        raw=torch.randn(5,2,256);restored=torch.randn_like(raw)
        self.assertTrue(torch.equal(mix_iq(raw,restored,torch.zeros(5)),raw))
        self.assertTrue(torch.equal(mix_iq(raw,restored,torch.ones(5)),restored))
        alpha=torch.tensor([0.,.25,.5,.75,1.])
        actual=mix_iq(raw,restored,alpha)
        torch.testing.assert_close(actual,(1-alpha[:,None,None])*raw+alpha[:,None,None]*restored)
        self.assertTrue(torch.isfinite(actual).all())

    def test_local_randomness_reproduces_dose_pairs_without_touching_native_rng(self):
        torch.manual_seed(132);raw=torch.zeros(8,2,256);restored=torch.ones_like(raw)
        state=torch.random.get_rng_state().clone()
        a=paired_views(raw,restored,'MixMix',torch.Generator().manual_seed(271))
        b=paired_views(raw,restored,'MixMix',torch.Generator().manual_seed(271))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        for x,y in zip(a[0]+a[1],b[0]+b[1]):self.assertTrue(torch.equal(x,y))
        self.assertFalse(torch.equal(a[1][0],a[1][1]))
        c=paired_views(raw,restored,'MixG',torch.Generator().manual_seed(271))
        self.assertTrue(torch.equal(c[0][0],a[0][0]));self.assertTrue(torch.equal(c[0][1],restored))
        self.assertEqual(len(c[1]),1)
        for view,alpha in zip(a[0],a[1]):
            self.assertTrue(bool(((alpha>=0)&(alpha<=1)).all()))
            torch.testing.assert_close(view,alpha[:,None,None].expand_as(view))
        full=torch.Generator().manual_seed(272);pair=torch.Generator().manual_seed(272)
        for _ in range(3):
            m=paired_views(raw,restored,'MixMix',full)
            p=paired_views(raw,restored,'MixG',pair)
            self.assertTrue(torch.equal(m[0][0],p[0][0]))

    def test_rejects_malformed_nonfinite_or_trainable_restoration(self):
        raw=torch.zeros(4,2,256);restored=torch.ones_like(raw);alpha=torch.ones(4)*.5
        for r,g,a in [(raw,restored[:3],alpha),(raw,restored,alpha[:3]),(raw,restored,alpha+1),
                      (raw,restored*float('nan'),alpha),(raw,restored.clone().requires_grad_(),alpha)]:
            with self.assertRaises(ValueError):mix_iq(r,g,a)
        with self.assertRaises(ValueError):paired_views(raw,restored.detach(),'other',torch.Generator())


if __name__=='__main__':unittest.main()
