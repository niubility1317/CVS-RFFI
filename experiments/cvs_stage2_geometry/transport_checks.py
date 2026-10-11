import copy,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import transport_audit as m


def fixture():
    rng=torch.Generator().manual_seed(71);observations={};layout=dict(txs=list(range(6)),days=[1,2,3],rxs=[1,3,4,6,8]);head=torch.randn(6,160,generator=rng,dtype=torch.double)
    rx={r:.1*torch.randn(160,generator=rng,dtype=torch.double) for r in layout['rxs']}
    for y in layout['txs']:
        for day in layout['days']:
            for r in layout['rxs']:
                z=F.normalize(head[y]+rx[r],dim=0);s=torch.randn(40,generator=rng,dtype=torch.double)
                observations[y,day,r]=dict(jointA=z+.001*torch.randn(160,generator=rng,dtype=torch.double),jointB=z+.001*torch.randn(160,generator=rng,dtype=torch.double),stateA=s,stateB=s+.01*torch.randn(40,generator=rng,dtype=torch.double))
    return observations,layout,head


class Checks(unittest.TestCase):
    def test_dual_matches_primal(self):
        g=torch.Generator().manual_seed(12);x=torch.randn(16,40,generator=g,dtype=torch.double);y=torch.randn(16,160,generator=g,dtype=torch.double);fit=m.fit_affine(x,y)
        xc=(x-x.mean(0))/fit['scale'];yc=y-y.mean(0)
        primal=torch.linalg.solve(xc.T@xc/len(x)+m.RIDGE*torch.eye(40,dtype=torch.double),xc.T@yc/len(x))
        self.assertTrue(torch.allclose(primal,fit['weight'],atol=1e-10,rtol=1e-10))

    def test_constant_input_is_shared(self):
        x=torch.ones(16,160,dtype=torch.double);y=torch.arange(16*160,dtype=torch.double).reshape(16,160)/1000
        fit=m.fit_affine(x,y);self.assertTrue(torch.equal(fit['weight'],torch.zeros_like(fit['weight'])))
        self.assertTrue(torch.equal(m.predict_affine(fit,x[0]),m.SHRINK*y.mean(0)))

    def test_cross_halves_fit_target(self):
        obs,layout,_=fixture();fit={k:v for k,v in obs.items() if k[0] not in (0,1) and k[1]!=1};models=m.fit_transport(fit,layout['rxs'])
        target=[];x=[]
        for y,day in sorted({(a,b) for a,b,r in fit}):
            for side,other in [('A','B'),('B','A')]:
                x.append(fit[y,day,1]['joint'+side]);target.append(fit[y,day,3]['joint'+other]-fit[y,day,1]['joint'+other])
        direct=m.fit_affine(torch.stack(x),torch.stack(target))
        self.assertTrue(torch.equal(models[1,3]['conditional_joint']['weight'],direct['weight']))
        self.assertTrue(torch.equal(models[1,3]['shared_rx'],m.SHRINK*torch.stack(target).mean(0)))

    def test_no_held_destination_input_or_fit(self):
        obs,layout,head=fixture();records,_,folds=m.frozen_predictions(obs,layout,head)
        changed=copy.deepcopy(obs);changed[0,1,3]['jointA']+=100;changed[0,1,3]['jointB']-=200;changed[0,1,3]['stateA']+=100
        other,_,_=m.frozen_predictions(changed,layout,head)
        one=next(r for r in records if r['fold']==0 and r['tx']==0 and r['anchor']==1);two=next(r for r in other if r['fold']==0 and r['tx']==0 and r['anchor']==1)
        for method in m.METHODS:self.assertTrue(torch.equal(one['predictions'][method],two['predictions'][method]))
        for f in folds:
            self.assertFalse(set(f['fit_tx'])&set(f['held_tx']));self.assertNotIn(f['held_day'],f['fit_days'])

    def test_star_cycles(self):
        obs,layout,head=fixture();records,_,_=m.frozen_predictions(obs,layout,head)
        for r in records:
            for p in r['predictions'].values():
                self.assertTrue(torch.equal(p[layout['rxs'].index(r['anchor'])],torch.zeros(160,dtype=torch.double)))
                self.assertLess(float(((p[0]-p[1])+(p[1]-p[2])+(p[2]-p[0])).abs().max()),1e-12)

    def test_joint_rotation(self):
        obs,layout,head=fixture();q=torch.linalg.qr(torch.randn(160,160,dtype=torch.double,generator=torch.Generator().manual_seed(28))).Q
        fit={k:v for k,v in obs.items() if k[0] not in (0,1) and k[1]!=1};rot={k:{name:(v@q if name.startswith('joint') else v) for name,v in item.items()} for k,item in fit.items()}
        a=m.fit_transport(fit,layout['rxs']);b=m.fit_transport(rot,layout['rxs']);x=obs[0,1,1]
        p=m.star_predictions(a,1,x['jointA'],x['stateA'],layout['rxs']);pr=m.star_predictions(b,1,x['jointA']@q,x['stateA'],layout['rxs'])
        for method in m.METHODS:self.assertTrue(torch.allclose(p[method]@q,pr[method],rtol=1e-9,atol=1e-10))

    def test_exact_star_has_zero_scoring_error(self):
        obs,layout,head=fixture();records,_,_=m.frozen_predictions(obs,layout,head);out={k:v['jointB'] for k,v in obs.items()}
        for r in records:
            vals=torch.stack([out[r['tx'],r['day'],rx]-out[r['tx'],r['day'],r['anchor']] for rx in layout['rxs']]);r['predictions']={name:vals for name in m.METHODS}
        result=m.score_predictions(records,out,layout['rxs'],head,'synthetic')
        for method in m.METHODS:
            self.assertEqual(result['metrics'][method]['relations'],900)
            for space in ('full','logit','margin'):self.assertLess(result['metrics'][method][space]['mse'],1e-26)

    def test_source_split_complete_disjoint_and_fixed(self):
        g=torch.Generator().manual_seed(92);roles=torch.tensor([(y,day,rx) for y in range(6) for day in (1,2,3) for rx in (1,3,4,6,8) for i in range(70)])
        y,day,rx=roles.T;cache=dict(y=y,day=day,rx=rx,ids=[f'packet-{i}' for i in range(6300)],parent_feature=F.normalize(torch.randn(6300,160,generator=g),dim=1),state=torch.randn(6300,40,generator=g))
        obs,layout=m.source_observations(cache);self.assertEqual(len(obs),90);seen=[]
        for group in layout['splits']:
            self.assertEqual(len(group['A_indices']),35);self.assertFalse(set(group['A_indices'])&set(group['B_indices']));seen+=group['A_indices']+group['B_indices']
        self.assertEqual(sorted(seen),list(range(6300)))
        order=torch.randperm(6300,generator=g);other={k:([v[i] for i in order] if k=='ids' else v[order]) for k,v in cache.items()};obs2,_=m.source_observations(other)
        for key in obs:self.assertTrue(torch.equal(obs[key]['jointA'],obs2[key]['jointA']))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
