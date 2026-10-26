import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from experiments.pamf_channel_pl import sig1_role_views as v


class RoleViewsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_U_selection_is_RX_balanced_deterministic_and_rejects_TX(self):
        m=dict(rx_i=np.repeat([1,3,4,6,8],1400));a=v.select_U(m,[1,3,4,6,8]);b=v.select_U(m,[1,3,4,6,8])
        np.testing.assert_array_equal(a,b);self.assertEqual(len(set(a)),6300)
        self.assertEqual([int((m['rx_i'][a]==r).sum()) for r in [1,3,4,6,8]],[1260]*5)
        with self.assertRaises(ValueError):v.select_U(dict(m,y=np.zeros(7000)),[1,3,4,6,8])

    def test_U_hidden_and_L_visible_roles_are_enforced(self):
        hidden=(None,torch.full((3,),-1),None,dict(tx_label_visible=torch.zeros(3,dtype=torch.bool)))
        v.hidden(hidden,'U_s')
        with self.assertRaises(ValueError):v.hidden((None,torch.zeros(3,dtype=torch.long),None,hidden[3]),'U_s')
        with self.assertRaises(ValueError):v.hidden(hidden,'V')
        with self.assertRaises(ValueError):v.hidden(hidden,'L_s')

    def test_known_RX_does_not_establish_same_TX(self):
        pred=np.array([0,0,0,1]);rx=np.array([1,3,4,6]);true=np.array([0,0,1,1]);mask=np.ones(4,bool)
        r=v.cross_rx_pairs(pred,mask,rx,true);self.assertEqual(r['ordered_same_pseudo_cross_RX_pairs'],6)
        self.assertEqual((r['true_same_TX_pairs'],r['false_positive_pairs']),(2,4))
        unknown=v.cross_rx_pairs(pred,mask,rx);self.assertFalse(unknown['true_pair_quality_known']);self.assertNotIn('positive_pair_precision',unknown)

    def test_relative_noise_is_40dB_deterministic_and_does_not_rescale_IQ(self):
        x=torch.randn(3,2,256);meta={k:torch.arange(3) for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
        with patch.dict('sys.modules',{'leo_practical.channel':SimpleNamespace(stable_seed=lambda *args:173)}):
            a=v.noise40(x,meta,'post');b=v.noise40(x,meta,'post')
        self.assertTrue(torch.equal(a,b));power=x.square().sum(1).mean(1);noise=(a-x).square().sum(1).mean(1)
        self.assertTrue(torch.allclose(10*torch.log10(power/noise),torch.full((3,),40.),atol=1e-4))
        with self.assertRaises(ValueError):v.noise40(torch.full_like(x,float('nan')),meta,'post')

    def test_feature_gradients_do_not_fit_or_update_classifier(self):
        head=torch.nn.Linear(160,6,bias=False).requires_grad_(False);model=SimpleNamespace(classify_features=head)
        h=torch.randn(8,160);y=torch.tensor([0,1,2,3,4,5,0,1]);before=head.weight.detach().clone()
        a,b,norm=v.feature_gradients(model,h,torch.ones(8,6)/6,y,y)
        self.assertTrue(torch.allclose(a,torch.ones_like(a),atol=1e-6));self.assertTrue(torch.isfinite(b).all())
        self.assertIsNone(head.weight.grad);self.assertTrue(torch.equal(before,head.weight));self.assertFalse(h.requires_grad)

    def test_source_runtime_rejects_target_or_foreign_output(self):
        rt=dict(run_id=v.RUN,project='/project',python='python',run_root='/project/runs/'+v.RUN,release_root='/project/releases/'+v.RELEASE,
            context='/project/releases/'+v.RELEASE+'/context.json',runtime_file='/project/releases/'+v.RELEASE+'/runtime.json',code_commit='commit',frozen_sources={},allowed_gpus=list(range(8)),launch_owner='codex-pamf-sig1-role-views-20261026')
        v.validate_runtime(rt)
        for change in (dict(tests={}),dict(truth='truth'),dict(run_root='/other'),dict(allowed_gpus=[0])):
            with self.assertRaises(ValueError):v.validate_runtime(dict(rt,**change))


if __name__=='__main__':unittest.main()
