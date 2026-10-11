import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import metric_audit as m


def fixture():
    rng=torch.Generator().manual_seed(52)
    roles=torch.tensor([(y,d,r) for y in range(6) for d in (1,2,3) for r in (1,3,4) for _ in range(4)])
    y,day,rx=roles.T;centers=torch.randn(6,160,generator=rng,dtype=torch.double)
    z=F.normalize(centers[y]+.1*torch.randn(len(y),160,generator=rng,dtype=torch.double),dim=1)
    j=.01*torch.randn(len(y),160,8,generator=rng,dtype=torch.double)
    cache=dict(y=y,day=day,rx=rx,parent_feature=z,ids=[f'L{i}' for i in range(len(y))])
    response=dict(jacobian=j,validation=dict(y=y.clone(),day=day.clone(),rx=rx.clone(),feature=z.clone()))
    return cache,response,centers


class Checks(unittest.TestCase):
    def test_spd_and_full_dimensional(self):
        a,b,_=fixture();mu,c,meta=m.fit_components(a['parent_feature'],a['y'],a['rx'],a['day'],b['jacobian'])
        for name in m.METHODS:
            p,s=m.make_metric(c,name);self.assertEqual(int(torch.linalg.matrix_rank(p)),160)
            self.assertGreater(s['min_eigenvalue'],0);self.assertTrue(torch.isfinite(m.logits(a['parent_feature'],mu,p)).all())
        self.assertEqual(meta['groups'],54)

    def test_rotation_invariance(self):
        a,b,_=fixture();q=torch.linalg.qr(torch.randn(160,160,dtype=torch.double,generator=torch.Generator().manual_seed(72))).Q
        z=a['parent_feature'];j=b['jacobian'];mu,c,_=m.fit_components(z,a['y'],a['rx'],a['day'],j)
        mr,cr,_=m.fit_components(z@q,a['y'],a['rx'],a['day'],torch.einsum('ndk,de->nek',j,q))
        for name in m.METHODS:
            p,_=m.make_metric(c,name);pr,_=m.make_metric(cr,name)
            self.assertTrue(torch.allclose(m.logits(z,mu,p),m.logits(z@q,mr,pr),atol=2e-8,rtol=2e-8))

    def test_held_condition_excluded_and_validation_no_fit(self):
        a,b,h=fixture();before={};real=m.fit_components
        def traced(z,y,rx,day,j):
            self.assertFalse(bool((rx==3).any()));before['z']=z.clone();return real(z,y,rx,day,j)
        with patch.object(m,'fit_components',side_effect=traced):result,params=m.evaluate_fold(a,b,h,'rx',3)
        changed=dict(b,validation={**b['validation'],'feature':-b['validation']['feature'],'y':(b['validation']['y']+1)%6})
        result2,params2=m.evaluate_fold(a,changed,h,'rx',3)
        self.assertEqual(result['fit_ids'],result2['fit_ids'])
        for name in m.METHODS:
            self.assertTrue(torch.equal(params[name]['transform'],params2[name]['transform']))
            self.assertTrue(torch.equal(params[name]['centers'],params2[name]['centers']))
        self.assertEqual(result['fit_count'],144);self.assertEqual(result['validation_count'],72)

    def test_nested_rx_prediction_uses_other_tx_and_day(self):
        a,b,_=fixture();_,c,info=m.fit_components(a['parent_feature'],a['y'],a['rx'],a['day'],b['jacobian'])
        for fold in info['folds']:
            self.assertFalse(set(fold['held_tx'])&set(fold['fit_tx']));self.assertNotIn(fold['held_day'],fold['fit_days'])
            self.assertGreaterEqual(fold['c'],0);self.assertLessEqual(fold['c'],1)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(c['R']).min()),-1e-10)

    def test_zero_actions_leave_metric_unchanged(self):
        a,b,_=fixture();z=a['parent_feature'];z=F.normalize(F.one_hot(a['y'],160).double(),dim=1)
        mu,c,info=m.fit_components(z,a['y'],a['rx'],a['day'],torch.zeros_like(b['jacobian']))
        self.assertTrue(torch.equal(c['R'],torch.zeros_like(c['R'])))
        p,_=m.make_metric(c,'packet');p2,_=m.make_metric(c,'packet_LTR');self.assertTrue(torch.equal(p,p2))
        self.assertTrue(torch.equal(m.logits(z,mu,p).argmax(1),a['y']))

    def test_inference_independent_of_batch(self):
        a,b,_=fixture();z=a['parent_feature'];mu,c,_=m.fit_components(z,a['y'],a['rx'],a['day'],b['jacobian']);p,_=m.make_metric(c,'packet_LTR')
        self.assertTrue(torch.allclose(m.logits(z,mu,p)[0],m.logits(z[:1],mu,p)[0],atol=1e-10,rtol=1e-10))

    def test_missing_cells_rejected(self):
        a,b,_=fixture();mask=~((a['y']==0)&(a['day']==1)&(a['rx']==1))
        with self.assertRaisesRegex(ValueError,'Missing source cell'):m.fit_components(a['parent_feature'][mask],a['y'][mask],a['rx'][mask],a['day'][mask],b['jacobian'][mask])


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
