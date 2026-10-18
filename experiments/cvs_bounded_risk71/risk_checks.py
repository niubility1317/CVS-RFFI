"""Pure bounded-risk and Reader-routing checks; no actual model/channel/data."""
import argparse
from copy import deepcopy
from collections import Counter
import json
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F


def risk_checks():
    from . import risk
    passed = []
    nominal = [.1, .2, .3, .4]
    target = [.15, .40, .25, .20]
    pi = risk.mixture(nominal, .25, target)
    assert all(abs(a - (.75 * p + .25 * v)) < 1e-14 for a, p, v in zip(pi, nominal, target))
    assert nominal == [.1, .2, .3, .4] and target == [.15, .40, .25, .20]
    passed.append('fixed_convex_mixture_preserves_inputs')
    for p, gamma, v in [([float('nan'), .2, .3, .4], .25, target),
                         ([-.1, .3, .4, .4], .25, target),
                         ([.1, .2, .3, .5], .25, target),
                         (nominal, -.1, target), (nominal, 1.1, target),
                         (nominal, float('inf'), target),
                         (nominal, .25, [.15, .4, -.2, .65]),
                         (nominal, .25, [.15, .4, .25, .3])]:
        try: risk.mixture(p, gamma, v)
        except ValueError: pass
        else: raise AssertionError('Malformed mixture accepted')
    passed.append('nonfinite_negative_sum_and_gamma_rejected')
    try: risk.mixture([.5, .5, 0., 0.], .25, target)
    except ValueError: pass
    else: raise AssertionError('Positive mixture group without nominal observations accepted')
    assert risk.mixture([.5, .5, 0., 0.], 0., target) == [.5, .5, 0., 0.]
    passed.append('unsupported_positive_pi_group_rejected_before_consumption')

    from experiments.cvs_quality_risk70.model import weighted_satellite_terms
    values = torch.arange(30, dtype=torch.float64).reshape(5, 6)
    first = values.sin().requires_grad_(); second = (values * .7).cos().requires_grad_()
    labels = torch.tensor([0, 1, 2, 3, 4]); groups = torch.tensor([0, 0, 1, 2, 3])
    same = risk.mixture(nominal, 0., target)
    a, kl_a, _ = weighted_satellite_terms(first, second, labels, groups, nominal, 'QUALITY_STRAT')
    b, kl_b, _ = weighted_satellite_terms(first, second, labels, groups, same, 'QUALITY_STRAT')
    grad_a = torch.autograd.grad(a, (first, second), retain_graph=True)
    grad_b = torch.autograd.grad(b, (first, second), retain_graph=True)
    log_p, log_q = first.log_softmax(1), second.log_softmax(1)
    expected_kl = .5 * ((log_p.exp() - log_q.exp()) * (log_p - log_q)).sum(1).mean()
    assert torch.equal(a, b) and all(torch.equal(x, y) for x, y in zip(grad_a, grad_b))
    assert torch.equal(kl_a, kl_b) and torch.allclose(kl_b, expected_kl)
    passed.append('gamma_zero_CE_gradient_identity_and_unweighted_KL')

    class Reader:
        method = 'QUALITY_STRAT'
        def __init__(self, p_g=nominal):
            self.p_g = list(p_g); self.calls = self.packets = 0
            self.cache = object(); self.schedule = object()
            self.payload = (torch.zeros(64, 2, 256), torch.zeros(64, dtype=torch.long),
                            torch.zeros(64, dtype=torch.long), [dict(scene='toy')] * 64,
                            dict(method='QUALITY_STRAT', p_g=self.p_g, used_pool_indices=list(range(64)), groups=[i % 4 for i in range(64)]))
        def consume(self, step, epoch, device):
            self.calls += 1; self.packets += 64
            return self.payload
        def summary(self): return dict(method=self.method, p_g=self.p_g, calls=self.calls, packets=self.packets)
        def audit(self): return dict(method=self.method, p_g=self.p_g, fixed_pool=True)
        def __deepcopy__(self, memo):
            result = type(self)(self.p_g); result.cache = self.cache; result.schedule = self.schedule
            memo[id(self)] = result
            return result

    reader = Reader(); original_record = deepcopy(reader.payload[4])
    proxy = risk.ReadProxy(reader, .25, target)
    got = proxy.consume(1, 1, 'cpu')
    assert all(got[i] is reader.payload[i] for i in range(4)) and got[4] is not reader.payload[4]
    assert got[4]['used_pool_indices'] is reader.payload[4]['used_pool_indices']
    assert got[4]['p_g'] == proxy.pi_g and got[4]['nominal_p_g'] == nominal
    assert reader.payload[4] == original_record and reader.p_g == nominal
    assert proxy.calls == reader.calls == 1 and proxy.packets == reader.packets == 64
    assert proxy.summary()['p_g'] == proxy.audit()['p_g'] == nominal
    assert proxy.summary()['method'] == 'BOUNDED_RISK' and proxy.audit()['allocation_method'] == 'QUALITY_STRAT'
    passed.append('proxy_exact_pack_indices_nominal_record_and_counter_routing')
    clone = deepcopy(proxy)
    assert clone.reader is not reader and clone.reader.cache is reader.cache and clone.reader.schedule is reader.schedule
    assert clone.calls == clone.packets == 0
    clone.consume(1, 1, 'cpu')
    assert proxy.calls == 1 and clone.calls == 1 and proxy.packets == clone.packets == 64
    passed.append('deepcopy_fresh_counter_shared_readonly_cache_schedule')
    from experiments.cvs_quality_risk70 import pool
    calls = []
    def loaded(directory, method, *, seed):
        calls.append((directory, method, seed)); return Reader()
    with patch.object(pool, 'load', loaded), patch.object(pool, 'prepare', side_effect=AssertionError('New channel preparation forbidden')):
        loaded_proxy = risk.load('synthetic-existing-pool', seed=71, gamma=.25, v=target)
        assert isinstance(loaded_proxy, risk.ReadProxy) and calls == [('synthetic-existing-pool', 'QUALITY_STRAT', 71)]
        assert loaded_proxy.calls == loaded_proxy.packets == 0
    passed.append('load_existing_quality_strat_pool_only_zero_channel')
    return passed


def hook_checks():
    """Use the actual P70/P71 hooks with tensor logits, never a CVS model."""
    from . import risk
    from .model import _labeled_hook
    def terms(a,b,y,alpha):
        c0,c1=(F.cross_entropy(z,y,label_smoothing=.01) for z in (a,b))
        p,q=a.log_softmax(1),b.log_softmax(1)
        kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
        return .5*(c0+c1)+alpha*kl,.5*(c0+c1),kl,c0,c1
    rx_by_domain=[r for r in (1,3,4,6,8) for _ in range(3)]
    supcon=SimpleNamespace(schedule=lambda epoch:0,clear=lambda m:setattr(m,'_supcon_pending',None),RX_BY_DOMAIN=rx_by_domain,LEVELS=())
    base=SimpleNamespace(terms=terms,capture_forward=lambda m,x,d,a:(m.forward_identity_only(x,domain_labels=d)['tx_logits'],{}))
    fake={}
    for name,attrs in [
        ('experiments.cvs_crossview_supcon41',dict(model=base,relations=SimpleNamespace())),
        ('experiments.cvs_fullsource_concat20',dict(supcon=supcon)),
        ('experiments.cvs_noise_shape_factorial43',dict(noise=SimpleNamespace(apply=lambda raw,step,mode:(raw.clone(),dict(applied=0)))) )]:
        module=ModuleType(name);module.__dict__.update(attrs);fake[name]=module
    raw=torch.arange(128*2*256,dtype=torch.float64).reshape(128,2,256)/100000
    ordinary=raw+.1;y=torch.arange(128)%6;domains=torch.arange(128)%15
    sat=raw[:64]+2;sat_y=(torch.arange(64)+2)%6;sat_d=(torch.arange(64)+4)%15
    groups=[0]*48+[1]*4+[2]*4+[3]*8;pg=[.1,.2,.3,.4];v=[.15,.40,.25,.20]
    observations=[dict(sample_id=[1,rx_by_domain[int(d)],1,i,i],scene='toy',state_end='LOS',receiver_lock_status='locked') for i,d in enumerate(sat_d)]
    allocation=dict(groups=groups,counts=[48,4,4,8],p_g=pg,used_pool_indices=list(range(64)),origin_steps=[1]*64,origin_slots=list(range(0,128,2)))
    class Reader:
        method='QUALITY_STRAT'
        def __init__(self):self.calls=self.packets=0
        def summary(self):return dict(p_g=pg)
        def consume(self,step,epoch,device):
            self.calls+=1;self.packets+=64
            return sat,sat_y,sat_d,observations,allocation
    def drive(gamma,internal='QUALITY_STRAT'):
        calls=[];logits=[]
        m=SimpleNamespace(training=True,_quality_pool_bound=True,_quality_method=internal,_feature_l_steps=1,_full_calls=0,
            _legacy_actual_L_input=ordinary.clone(),_supcon_pending=None,_full_L_packets=0,_supcon_calls=0,_legacy_calls=0,
            _legacy_extra_packets=0,_legacy_positive_KL_calls=0,_full_records=[],_full_coverage=Counter(),_supcon_applied=0,
            _shape_offset=4440,_shape_mode='NONE',_shape_NN=0,_shape_packets=0,_shape_records=[],_quality_records=[],_crossview_mode='ORD128')
        m._risk_records=m._quality_records;m._full_channel=risk.ReadProxy(Reader(),gamma,v)
        def forward(x,domain_labels):
            calls.append((x.detach().clone(),domain_labels.detach().clone()))
            axis=torch.arange(6,dtype=x.dtype)
            z=(x.mean((1,2))[:,None]*(axis+.3)+domain_labels[:,None]*axis*.001+len(calls)*axis*.01).requires_grad_()
            logits.append(z);return dict(tx_logits=z)
        m.forward_identity_only=forward
        first=forward(ordinary,domains)['tx_logits'];identity=F.cross_entropy(first,y,label_smoothing=.01)
        with patch.dict(sys.modules,fake):
            delta,_=_labeled_hook(m,model=m,labeled_output=dict(tx_logits=first),labels=y,
                extra=(domains,dict(rx_i=torch.tensor(rx_by_domain)[domains])),domains=domains,
                optimizer_step=1,identity_loss=identity,labeled_count=128,labeled_inputs=raw)
        return m,calls,logits,delta+identity
    for gamma in (0.,.25):
        m,calls,z,actual=drive(gamma)
        mask=(torch.arange(128)+1)%2!=0;mixed_y=y.clone();mixed_y[mask]=sat_y
        mixed_d=domains.clone();mixed_d[mask]=sat_d;mixed_x=ordinary.clone();mixed_x[mask]=sat
        assert len(calls)==6 and all(torch.equal(calls[i][0],calls[j][0]) for i,j in ((0,1),(2,3),(4,5)))
        assert torch.equal(calls[2][0],mixed_x) and torch.equal(calls[2][1],mixed_d)
        per=.5*sum(F.cross_entropy(t[mask],sat_y,reduction='none',label_smoothing=.01) for t in z[2:4])
        group_ce=[per[torch.tensor(groups)==g].mean() for g in range(4)]
        nominal=sum(pg[g]*group_ce[g] for g in range(4));pi=risk.mixture(pg,gamma,v)
        increment=.25*sum((pi[g]-pg[g])*group_ce[g] for g in range(4))
        m2=.5*terms(z[0],z[1],y,1.)[0]+.5*terms(z[2],z[3],mixed_y,1.)[0]+.25*terms(z[4],z[5],y,1.)[0]+.25*(nominal-per.mean())
        expected=m2+increment
        assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)
        ag=torch.autograd.grad(actual,z,retain_graph=True);eg=torch.autograd.grad(expected,z,retain_graph=True)
        assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(ag,eg))
        record=m._quality_records[0]
        assert record['p_g']==pg and record['pi_g']==pi and record['v_g']==v
        assert record['used_pool_indices']==list(range(64)) and record['As_KL_quality_weighted'] is False
        assert abs(record['nominal_M2_same_logits_total_L']-float(m2))<1e-10
        assert abs(record['risk_vs_nominal_total_correction']-float(increment))<1e-10
        assert m._risk_records is m._quality_records and m._full_L_packets+m._shape_NN==768
        if gamma==0:assert torch.allclose(actual,m2) and record['risk_increment_to_total']==0
    try:drive(.25,'POOL_RANDOM')
    except ValueError:pass
    else:raise AssertionError('Wrong internal allocation mode silently ignored the bounded CE weights')
    return ['actual_six_forward_risk_hook_pack_loss_gradients_gamma_zero_and_mode_guard']


def run(hook_only=False):
    started = time.perf_counter()
    passed = [] if hook_only else risk_checks()
    passed += hook_checks()
    return dict(status='PASS', checks=len(passed), passed=passed,
                wall_seconds=time.perf_counter() - started,
                real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0, actual_channel_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--hook-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.hook_only), allow_nan=False))
