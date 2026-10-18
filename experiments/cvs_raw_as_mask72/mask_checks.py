"""Pure Dropout/tensor checks; no actual CVS model, IQ, CP, or channel."""
import argparse
from copy import deepcopy
import json
import time

import torch
from torch import nn
from torch.nn import functional as F


def mask_checks():
    from . import masks
    passed = []

    class Toy(nn.Module):
        def __init__(self, p=.45):
            super().__init__()
            self.sentinel = nn.Parameter(torch.tensor(2.))
            owner = self
            parts = masks.TARGET.split('.')
            for name in parts[:-1]:
                child = nn.Module(); owner.add_module(name, child); owner = child
            owner.add_module(parts[-1], nn.Dropout(p))
        def forward(self, x): return self.get_submodule(masks.TARGET)(x)

    def inputs():
        return [(torch.arange(128 * 160, dtype=torch.float64).reshape(128, 160) / 30000 + 1 + i).requires_grad_() for i in range(5)]

    results = {}
    for step in (1, 2):
        mask = (torch.arange(128) + step).remainder(2) != 0
        native = Toy().train(); xs = inputs(); torch.manual_seed(734)
        expected = [native(x) for x in xs]; expected_rng = torch.get_rng_state().clone()
        for mode in ('INDEPENDENT_FUSE', 'SHARED_RAW_AS_FUSE'):
            model = Toy().train(); values = inputs(); torch.manual_seed(734)
            with masks.scope(model, step=step, cumulative_step=4440 + step, method=mode, smask=mask) as recorder:
                actual = [model(x) for x in values]
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert torch.equal(actual[0], expected[0])
            assert all(torch.equal(a[~mask], e[~mask]) for a, e in zip(actual[1:], expected[1:]))
            assert len(model.get_submodule(masks.TARGET)._forward_hooks) == 0
            assert recorder.summary()['calls'] == 5 and recorder.summary()['private_draws'] == 4
            results[(step, mode)] = (recorder, actual, values)
    passed.append('five_native_draws_same_global_rng_ordinary_and_other64_unchanged_both_parities')
    for step in (1, 2):
        ind, independent, _ = results[(step, 'INDEPENDENT_FUSE')]
        shared, coupled, _ = results[(step, 'SHARED_RAW_AS_FUSE')]
        mask = (torch.arange(128) + step).remainder(2) != 0
        assert all(torch.equal(ind._masks[key], shared._masks[key]) for key in ('As0', 'As1', 'raw0', 'raw1'))
        assert torch.equal(coupled[1][mask] != 0, coupled[3][mask] != 0)
        assert torch.equal(coupled[2][mask] != 0, coupled[4][mask] != 0)
        assert not torch.equal(shared._masks['As0'], shared._masks['As1'])
        assert not torch.equal(independent[1][mask] != 0, independent[3][mask] != 0)
        assert shared.summary()['As_raw_pair_hamming'] == [0, 0]
    passed.append('paired_As_raw_masks_shared_RD_distinct_unused_independent_raw_draws_identical')
    recorder, actual, values = results[(1, 'SHARED_RAW_AS_FUSE')]
    selected = (torch.arange(128) + 1).remainder(2) != 0
    sum(value.sum() for value in actual).backward()
    for index, key in ((1, 'As0'), (2, 'As1'), (3, 'As0'), (4, 'As1')):
        expected_grad = recorder._masks[key].to(torch.float64) / .55
        assert torch.allclose(values[index].grad[selected], expected_grad)
        assert torch.allclose(actual[index][selected], values[index][selected] * expected_grad)
    torch.manual_seed(734); baseline = Toy().train(); reference = [baseline(x.detach()) for x in values]
    assert bool(((actual[1][selected] != 0) & (reference[1][selected] == 0)).any())
    passed.append('selected_gradient_uses_pre_dropout_input_without_double_mask')

    def reject(name, model, count=5, shape=(128, 160), mask=None):
        model.train(); selected = (torch.arange(128) + 1).remainder(2) != 0 if mask is None else mask
        try:
            with masks.scope(model, step=1, cumulative_step=4441, method='INDEPENDENT_FUSE', smask=selected):
                for _ in range(count): model(torch.ones(shape, requires_grad=True))
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted invalid scope: ' + name)
        assert len(model.get_submodule(masks.TARGET)._forward_hooks) == 0

    reject('wrong probability', Toy(.4))
    reject('wrong shape', Toy(), shape=(64, 160))
    reject('wrong parity', Toy(), mask=(torch.arange(128) + 2).remainder(2) != 0)
    reject('wrong selected count', Toy(), mask=torch.ones(128, dtype=torch.bool))
    passed.append('wrong_p_shape_and_mask_rejected_without_lingering_hook')
    reject('too few calls', Toy(), count=4); reject('too many calls', Toy(), count=6)
    passed.append('five_call_budget_enforced_and_hooks_removed')
    model = Toy().train(); selected = (torch.arange(128) + 1).remainder(2) != 0
    try:
        with masks.scope(model, step=1, cumulative_step=4441, method='INDEPENDENT_FUSE', smask=selected):
            model(torch.ones(128, 160, requires_grad=True))
            raise RuntimeError('original-toy-failure')
    except RuntimeError as error:
        assert str(error) == 'original-toy-failure'
    else:
        raise AssertionError('Original failure hidden')
    assert not model.get_submodule(masks.TARGET)._forward_hooks
    passed.append('exception_cleanup_preserves_original_failure')
    for no_grad in (False, True):
        model = Toy().train() if no_grad else Toy().eval()
        before = torch.get_rng_state().clone()
        try:
            with torch.set_grad_enabled(not no_grad):
                with masks.scope(model, step=1, cumulative_step=4441, method='INDEPENDENT_FUSE', smask=selected): pass
        except ValueError: pass
        else: raise AssertionError('Eval/no-grad scope accepted')
        assert not model.get_submodule(masks.TARGET)._forward_hooks and torch.equal(before, torch.get_rng_state())
    passed.append('eval_and_no_grad_install_no_hook_or_private_draw')
    model = Toy().train(); before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with masks.scope(model, step=1, cumulative_step=4441, method='SHARED_RAW_AS_FUSE', smask=selected):
        for x in inputs(): model(x)
    assert list(before) == list(model.state_dict()) and all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    passed.append('parameters_and_persistent_state_unchanged')
    return passed


def hook_checks():
    """Exact cache origin boundaries and the actual six-forward hook glue."""
    return ordered_checks()+model_hook_checks()


def ordered_checks():
    from . import ordered_pool
    from experiments.cvs_quality_risk70 import pool
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import patch
    n=142080;steps=np.arange(n)//64+1;slots=2*(np.arange(n)%64)+(steps+1)%2
    rxs=np.array([1,3,4,6,8]);rx_index=slots%5;days=slots%3+1
    ids=np.stack([np.ones(n,dtype=np.int64),rxs[rx_index],days,slots,slots],1)
    arrays=dict(physical_ids=ids,labels=slots%6,domains=rx_index*3+days-1,
                origin_steps=steps,origin_slots=slots)
    shared=dict(summary=dict(N=n,updates=2220,steps=222),arrays=arrays)
    class Reader:
        def __init__(self,shared,schedule,method,seed):
            self._shared,self._schedule,self.method,self.seed=shared,schedule,method,seed
            self.calls=self.packets=0
        def __deepcopy__(self,memo):
            result=type(self)(self._shared,self._schedule,self.method,self.seed);memo[id(self)]=result;return result
        def consume(self,step,epoch,device):
            used=self._schedule[step-1];a=self._shared['arrays'];meta=[]
            for i in used:
                meta.append(dict(sample_id=a['physical_ids'][i].tolist(),physical_id=a['physical_ids'][i].tolist(),
                    label=int(a['labels'][i]),domain=int(a['domains'][i]),origin_step=int(a['origin_steps'][i]),origin_slot=int(a['origin_slots'][i])))
            self.calls+=1;self.packets+=64
            return (torch.zeros(64,2,256),torch.tensor(a['labels'][used].tolist()),torch.tensor(a['domains'][used].tolist()),meta,
                    dict(used_pool_indices=used.tolist()))
        def summary(self):return dict(calls=self.calls,packets=self.packets)
        def audit(self):return dict(new_channel_calls=0)
    original_order=np.arange(n-1,-1,-1).reshape(2220,64)
    template=Reader(shared,original_order,'POOL_RANDOM',71)
    with patch.object(pool,'load',return_value=template),patch.object(pool,'Reader',Reader),patch.object(pool,'prepare',side_effect=AssertionError('no regeneration')):
        reader=ordered_pool.load('toy-completed-pool',seed=71)
    assert np.array_equal(reader.reader._schedule.ravel(),np.arange(n))
    assert template._schedule is original_order and original_order[0,0]==n-1
    def expected(step):
        span=slice(64*(step-1),64*step)
        return dict(labels=arrays['labels'][span],domains=arrays['domains'][span],physical_ids=ids[span])
    def reject(fn):
        try:fn()
        except ValueError:pass
        else:raise AssertionError('Invalid ordered raw-As origin accepted')
    reject(lambda:reader.consume(1,1,'cpu'))
    reject(lambda:reader.validate_origin(0,1,**expected(1)))
    wrong=expected(1);wrong['physical_ids']=wrong['physical_ids'].copy();wrong['physical_ids'][0,4]+=1
    reject(lambda:reader.validate_origin(1,1,**wrong))
    wrong=expected(1);wrong['labels']=wrong['labels']+1
    reject(lambda:reader.validate_origin(1,1,**wrong))
    wrong=expected(1);wrong['domains']=wrong['domains']+1
    reject(lambda:reader.validate_origin(1,1,**wrong))
    wrong=expected(1);wrong['physical_ids']=wrong['physical_ids'].astype(float)
    reject(lambda:reader.validate_origin(1,1,**wrong))
    for step in (1,2):
        origin=reader.validate_origin(step,1,**expected(step));_,_,_,_,record=reader.consume(step,1,'cpu')
        assert record['used_pool_indices']==list(range(64*(step-1),64*step))
        assert origin['selected_raw_indices']==list(range(0 if step%2 else 1,128,2))
    reject(lambda:reader.validate_origin(2,1,**expected(2)))
    clone=deepcopy(reader);assert clone.calls==clone.packets==0 and clone.reader._shared is shared
    # A valid prefix state isolates the last-step boundary without replaying IQ.
    reader.reader.calls=2219;reader.reader.packets=2219*64
    reject(lambda:reader.validate_origin(2220,9,**expected(2220)))
    origin=reader.validate_origin(2220,10,**expected(2220));reader.consume(2220,10,'cpu')
    assert origin['used_pool_indices']==list(range(n-64,n)) and reader.summary()['complete']
    reject(lambda:reader.validate_origin(2221,11,**expected(2220)))
    return ['ordered_cache_identity_origin_parity_first_last_labels_domains_and_no_regeneration']


def model_hook_checks():
    from .model import _labeled_hook
    from . import masks
    from collections import Counter
    from types import ModuleType,SimpleNamespace
    from unittest.mock import patch
    import sys
    def terms(a,b,y,alpha):
        c0,c1=(F.cross_entropy(z,y,label_smoothing=.01) for z in (a,b))
        p,q=a.log_softmax(1),b.log_softmax(1);kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
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
    smask=(torch.arange(128)+1)%2!=0;selected=torch.nonzero(smask).flatten().tolist()
    meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor(rx_by_domain)[domains],
              day_i=domains%3+1,sig_i=torch.arange(128),base_index=torch.arange(128))
    ids=[[int(meta[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')] for i in selected]
    observed=[dict(sample_id=pid,scene='toy',state_end='LOS',receiver_lock_status='locked') for pid in ids]
    class Reader:
        def __init__(self):self.last_origin=None
        def validate_origin(self,step,epoch,*,labels,domains,physical_ids):
            assert physical_ids==ids and torch.equal(labels,y[smask]) and torch.equal(domains,globals_d[smask])
            self.last_origin=dict(physical_ids=ids,labels=labels.tolist(),domains=domains.tolist());return self.last_origin
        def consume(self,step,epoch,device):
            assert self.last_origin is not None
            return raw[smask]+2,y[smask],domains[smask],observed,dict(groups=[i%4 for i in range(64)],counts=[16]*4,
                p_g=[.1,.2,.3,.4],used_pool_indices=list(range(64)),origin_steps=[1]*64,origin_slots=selected)
    globals_d=domains
    states=[];records=[]
    for mode in masks.MODES:
        torch.manual_seed(117);target=torch.nn.Dropout(.45).train();calls=[];logits=[]
        m=SimpleNamespace(training=True,_quality_pool_bound=True,_quality_method='POOL_RANDOM',_feature_l_steps=1,_full_calls=0,
            _legacy_actual_L_input=ordinary.clone(),_supcon_pending=None,_full_L_packets=0,_supcon_calls=0,_legacy_calls=0,
            _legacy_extra_packets=0,_legacy_positive_KL_calls=0,_full_records=[],_full_coverage=Counter(),_supcon_applied=0,
            _shape_offset=4440,_shape_mode='NONE',_shape_NN=0,_shape_packets=0,_shape_records=[],_quality_records=[],_crossview_mode='ORD128',
            _mask_mode=mode,_mask_logical_method='MASK_SHARED' if mode==masks.MODES[1] else 'MASK_INDEPENDENT',_mask_records=[],_full_channel=Reader())
        m.get_submodule=lambda name:target if name==masks.TARGET else None
        def forward(x,domain_labels):
            calls.append((x.detach().clone(),domain_labels.detach().clone()))
            axis=torch.arange(160,dtype=x.dtype)
            features=(x.mean((1,2))[:,None]*(axis+.3)/100+domain_labels[:,None]*axis*.0001).requires_grad_()
            z=target(features)[:,:6];logits.append(z);return dict(tx_logits=z)
        m.forward_identity_only=forward
        first=forward(ordinary,domains)['tx_logits'];identity=F.cross_entropy(first,y,label_smoothing=.01)
        with patch.dict(sys.modules,fake):
            delta,_=_labeled_hook(m,model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,
                optimizer_step=1,identity_loss=identity,labeled_count=128,labeled_inputs=raw)
        actual=delta+identity
        expected=.5*terms(logits[0],logits[1],y,1.)[0]+.5*terms(logits[2],logits[3],y,1.)[0]+.25*terms(logits[4],logits[5],y,1.)[0]
        assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)
        ag=torch.autograd.grad(actual,logits,retain_graph=True);eg=torch.autograd.grad(expected,logits,retain_graph=True)
        assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(ag,eg))
        assert len(calls)==6 and all(torch.equal(calls[i][0],calls[j][0]) for i,j in ((0,1),(2,3),(4,5)))
        assert not target._forward_hooks and m._full_L_packets+m._shape_NN==768
        q=m._quality_records[0];r=m._mask_records[0]
        assert q['origin_verified'] and q['CE_correction_to_total']==0 and q['physical_ids']==ids
        assert r['calls']==5 and r['As_origin_verified'] and r['used_pool_indices']==list(range(64))
        states.append(torch.get_rng_state().clone());records.append(r)
    assert torch.equal(states[0],states[1])
    assert records[0]['mask_seeds']==records[1]['mask_seeds'] and records[0]['private_kept_counts']==records[1]['private_kept_counts']
    assert records[1]['As_raw_pair_hamming']==[0,0] and all(v>0 for v in records[0]['As_raw_pair_hamming'])
    assert all(r['RD_different'] for r in records)
    return ['actual_six_forward_mask_hook_origin_flat_loss_gradients_and_matched_private_streams']


def run(hook_only=False):
    started = time.perf_counter()
    passed = [] if hook_only else mask_checks()
    passed += hook_checks()
    return dict(status='PASS', checks=len(passed), passed=passed,
                wall_seconds=time.perf_counter() - started,
                real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0, actual_channel_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--hook-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.hook_only), allow_nan=False))
