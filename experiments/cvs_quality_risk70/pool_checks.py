"""Small synthetic pool and tensor-loss checks; no actual channel/model/data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import tempfile
import threading
import time
from types import SimpleNamespace, ModuleType
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def _toy_directory(pool):
    with tempfile.TemporaryDirectory(prefix='quality70-toy-') as temp:
        try:
            yield temp
        finally:
            # Windows cannot remove a temporary NPY while its shared mmap is open.
            for key in list(pool._CACHE):
                if Path(key).is_relative_to(Path(temp)):
                    shared = pool._CACHE.pop(key)
                    for array in shared['arrays'].values(): array._mmap.close()
            for key in list(pool._SCHEDULE_CACHE):
                if Path(key[0]).is_relative_to(Path(temp)): pool._SCHEDULE_CACHE.pop(key)


def loss_checks():
    from .model import weighted_satellite_terms
    values = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    first = values.sin().requires_grad_()
    second = (values * .7).cos().requires_grad_()
    y = torch.tensor([0, 1, 2, 3, 4, 5])
    groups = torch.tensor([0, 0, 0, 1, 2, 3])
    p_g = [.1, .2, .3, .4]
    ce, kl, stats = weighted_satellite_terms(first, second, y, groups, p_g, 'QUALITY_STRAT')
    per = .5 * (F.cross_entropy(first, y, reduction='none', label_smoothing=.01)
                  + F.cross_entropy(second, y, reduction='none', label_smoothing=.01))
    expected = sum(p_g[g] * per[groups == g].mean() for g in range(4))
    assert torch.allclose(ce, expected) and not torch.allclose(ce, per.mean())
    actual_grad = torch.autograd.grad(ce, (first, second), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (first, second), retain_graph=True)
    assert all(torch.allclose(a, b) for a, b in zip(actual_grad, expected_grad))
    plain, other_kl, _ = weighted_satellite_terms(first, second, y, groups, p_g, 'POOL_RANDOM')
    p, q = first.log_softmax(1), second.log_softmax(1)
    expected_kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1).mean()
    assert torch.allclose(plain, per.mean()) and torch.allclose(kl, expected_kl) and torch.allclose(other_kl, expected_kl)
    try:
        weighted_satellite_terms(first[:5], second[:5], y[:5], groups[:5], p_g, 'QUALITY_STRAT')
    except ValueError:
        pass
    else:
        raise AssertionError('Positive nominal group absent from stratified batch')
    return ['fixed_nominal_weights_and_gradients', 'KL_unweighted_in_both_methods', 'missing_positive_group_rejected']


def pool_checks():
    from . import pool
    passed = []
    groups = np.repeat(np.arange(4), [5, 6, 7, 238])
    random_state = np.random.get_state()
    a = pool._schedule(groups, 'POOL_RANDOM', seed=71, updates=4)
    b = pool._schedule(groups, 'POOL_RANDOM', seed=71, updates=4)
    assert np.array_equal(a, b) and sorted(a.ravel().tolist()) == list(range(256))
    assert all(np.array_equal(x, y) for x, y in zip(random_state, np.random.get_state()))
    passed.append('random_private_reproducible_one_use_schedule')
    strat = pool._schedule(groups, 'QUALITY_STRAT', seed=71, updates=4)
    counts = np.stack([np.bincount(groups[row], minlength=4) for row in strat])
    totals = np.bincount(groups, minlength=4)
    assert strat.shape == (4, 64) and sorted(strat.ravel().tolist()) == list(range(256))
    assert np.array_equal(counts.sum(0), totals)
    assert np.all((counts == totals // 4) | (counts == totals // 4 + 1)) and np.all(counts > 0)
    passed.append('stratified_balanced_rows_exact_columns_and_one_use')

    class Dataset:
        transform = None
        def __init__(self):
            self.selected = np.arange(128, dtype=np.int64)
            self.base = SimpleNamespace(index=[SimpleNamespace(eq_i=1, rx_i=1, day_i=1, sig_i=i) for i in range(128)], transform=None)
        def __getitem__(self, index):
            i = int(self.selected[index])
            return (torch.full((2, 256), i / 128.), i % 6, 101 + i % 2,
                    dict(eq_i=1, rx_i=1, day_i=1, sig_i=i, base_index=i, tx_i=i % 6, tx='toy', split_source='labeled'))

    class Channel:
        quality = 'valid'
        constructors = 0
        blocker = None
        def __init__(self):
            type(self).constructors += 1
            self.calls = self.packets = 0
        def apply(self, x, meta, epoch, step):
            if self.blocker is not None and self.calls == 0:
                self.blocker[0].set()
                assert self.blocker[1].wait(5)
            random.random(); np.random.rand(); torch.rand(())
            records = []
            for i in range(len(x)):
                quality = ([1., 4., 10., 18.][i % 4] if self.quality == 'valid'
                           else (4. if self.calls == 0 and i == 0 else 1.) if self.quality == 'rare'
                           else 1. if self.quality == 'one' else None)
                records.append(dict(signal_to_added_noise_db=quality, scene='toy', state_end='LOS', receiver_lock_status='locked',
                                    sample_id=[int(meta[k][i]) for k in pool.KEYS], keep_complete_field=dict(value=i), namespace=f'toy/{epoch}/{step}'))
            self.calls += 1; self.packets += len(x)
            return x + .001 * step, records
        def audit(self): return dict(toy=True)
        def summary(self): return dict(calls=self.calls, packets=self.packets)

    original = pool._new_channel
    pool._new_channel = Channel
    try:
        with _toy_directory(pool) as temp:
            root = Path(temp)
            trace = root / 'trace.gz'
            ids = [[1, 1, 1, i, i] for i in range(128)]
            with gzip.open(trace, 'wt', encoding='utf8') as stream:
                for _ in range(4):
                    stream.write(json.dumps(dict(role='L', ids=ids)) + '\n')
                    # This malformed U payload must never be JSON-decoded.
                    stream.write('{"role":"U","ids":DO_NOT_DECODE_U}\n')
            ctx = dict(train_loader=SimpleNamespace(dataset=Dataset()), domain_label_map={101: 7, 102: 8})
            py = random.getstate(); nr = np.random.get_state(); tr = torch.get_rng_state().clone()
            ready = pool.prepare(ctx, root / 'valid', trace, seed=71, updates=4, steps=2)
            assert random.getstate() == py and all(np.array_equal(x, y) for x, y in zip(nr, np.random.get_state())) and torch.equal(torch.get_rng_state(), tr)
            assert ready['N'] == 256 and ready['counts'] == [64] * 4 and ready['p_g'] == [.25] * 4
            assert ready['channel']['calls'] == 4 and ready['channel']['packets'] == 256
            first = pool.load(root / 'valid', 'POOL_RANDOM', seed=71)
            iq, y, domains, metadata, record = first.consume(1, 1, 'cpu')
            assert iq.shape == (64, 2, 256) and y.shape == domains.shape == (64,)
            assert set(domains.tolist()) <= {7, 8} and all(m['keep_complete_field'] for m in metadata)
            assert all(m['sample_id'] == m['physical_id'] for m in metadata)
            assert set(record) >= {'used_pool_indices', 'counts', 'groups', 'p_g', 'origin_steps', 'origin_slots'}
            passed.append('prepare_L_only_occurrences_domains_metadata_and_RNG_restored')
            before = Channel.constructors
            assert pool.prepare(ctx, root / 'valid', trace, seed=71, updates=4, steps=2) == ready and Channel.constructors == before
            passed.append('completed_pool_reuse_generates_no_extra_channel')
            for method in ('POOL_RANDOM', 'QUALITY_STRAT'):
                reader = pool.load(root / 'valid', method, seed=71)
                clone = deepcopy(reader)
                assert reader._shared is clone._shared and reader._schedule is clone._schedule and clone.calls == clone.packets == 0
                used = []; physical = []
                for step in range(1, 5):
                    _, _, _, meta, rec = reader.consume(step, (step - 1) // 2 + 1, 'cpu')
                    used.extend(rec['used_pool_indices']); physical.extend(tuple(m['sample_id']) for m in meta)
                    if method == 'QUALITY_STRAT': assert rec['counts'] == [16] * 4
                assert sorted(used) == list(range(256)) and len(set(physical)) == 128
                assert reader.calls == 4 and reader.packets == 256 and clone.calls == clone.packets == 0
                try: reader.consume(4, 2, 'cpu')
                except ValueError: pass
                else: raise AssertionError('Repeated consume accepted')
            passed.append('reader_one_use_occurrences_counters_deepcopy_and_order_guard')
            Channel.quality = 'none'
            try: pool.prepare(ctx, root / 'bad_quality', trace, seed=71, updates=4, steps=2)
            except pool.PoolPreparationError: pass
            else: raise AssertionError('None quality accepted')
            failed = json.loads((root / 'bad_quality' / 'failed.json').read_text(encoding='utf8'))
            assert failed['written'] == 64 and (root / 'bad_quality' / 'iq.npy').exists() and (root / 'bad_quality' / 'metadata.jsonl').stat().st_size > 0
            passed.append('invalid_quality_stops_preserving_generated_pool')
            Channel.quality = 'rare'
            try: pool.prepare(ctx, root / 'insufficient', trace, seed=71, updates=4, steps=2)
            except pool.PoolPreparationError: pass
            else: raise AssertionError('Insufficient group accepted')
            failed = json.loads((root / 'insufficient' / 'failed.json').read_text(encoding='utf8'))
            assert failed['written'] == 256 and failed['counts'] == [255, 1, 0, 0]
            passed.append('insufficient_group_stops_without_drop_or_resample')
            Channel.quality = 'valid'
            bad = root / 'bad_trace.gz'
            with gzip.open(bad, 'wt', encoding='utf8') as stream:
                wrong = [row[:] for row in ids]; wrong[-1][-1] = 999
                stream.write(json.dumps(dict(role='L', ids=wrong)) + '\n')
            try: pool.prepare(ctx, root / 'illegal', bad, seed=71, updates=4, steps=2)
            except pool.PoolPreparationError: pass
            else: raise AssertionError('Non-L selected source accepted')
            passed.append('trace_membership_rejected_before_IQ_channel')
            start, release = threading.Event(), threading.Event()
            Channel.blocker = (start, release)
            before = Channel.constructors
            with ThreadPoolExecutor(max_workers=2) as executor:
                one = executor.submit(pool.prepare, ctx, root / 'shared', trace, seed=71, updates=4, steps=2)
                assert start.wait(5)
                two = executor.submit(pool.prepare, ctx, root / 'shared', trace, seed=71, updates=4, steps=2)
                time.sleep(.05)
                assert Channel.constructors == before + 1
                release.set()
                assert one.result(timeout=10) == two.result(timeout=10)
            Channel.blocker = None
            passed.append('shared_directory_lock_one_generation_for_two_rows')
    finally:
        pool._new_channel = original
    return passed


def interface_checks():
    """Actual dataset tuple types, zero nominal groups, and native offset code."""
    from . import pool
    groups = np.zeros(256, dtype=np.int64)
    schedule = pool._schedule(groups, 'QUALITY_STRAT', seed=71, updates=4)
    assert schedule.shape == (4, 64) and sorted(schedule.ravel().tolist()) == list(range(256))
    try:
        pool._schedule(np.array([0] * 255 + [1]), 'QUALITY_STRAT', seed=71, updates=4)
    except ValueError:
        pass
    else:
        raise AssertionError('A positive group smaller than the update count was accepted')
    class Dataset:
        def __getitem__(self, index):
            return (torch.zeros(2, 256), torch.tensor(index % 6), torch.tensor(102),
                    dict(zip(pool.KEYS, (1, 1, 1, index, index))))
    ids = [(1, 1, 1, i, i) for i in range(128)]
    result = pool._batch(Dataset(), {pid:i for i,pid in enumerate(ids)}, {102:7}, ids, 1)
    assert len(result[1]) == 64 and set(result[2]) == {7}
    assert result[4] == list(range(0, 128, 2))
    # Compile the actual tiny adapter class, without importing or building a CVS.
    import ast, importlib.util
    spec = importlib.util.find_spec('experiments.cvs_clock_factorial30.model')
    source = Path(spec.origin).read_text(encoding='utf8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Shifted')
    ns = {'deepcopy':deepcopy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), spec.origin, 'exec'), ns)
    class Recorder:
        def apply(self, x, meta, epoch, step): return epoch, step
    shifted = ns['Shifted'](Recorder())
    assert shifted.apply(None, None, epoch=1, step=1) == (21, 4441)
    assert shifted.apply(None, None, epoch=10, step=2220) == (30, 6660)
    return ['zero_nominal_groups_and_rare_positive_group', 'native_scalar_label_domain_tuple', 'actual_Shifted_first_last_tail_boundaries']


def hook_checks():
    """Drive the actual hook with tensor logits, not a learned model."""
    from .model import _labeled_hook, weighted_satellite_terms
    def terms(a, b, y, alpha):
        ce0, ce1 = (F.cross_entropy(z, y, label_smoothing=.01) for z in (a, b))
        p, q = a.log_softmax(1), b.log_softmax(1)
        kl = .5 * ((p.exp()-q.exp())*(p-q)).sum(1).mean()
        return .5*(ce0+ce1)+alpha*kl, .5*(ce0+ce1), kl, ce0, ce1
    rx_by_domain = [r for r in (1,3,4,6,8) for _ in range(3)]
    supcon = SimpleNamespace(schedule=lambda epoch:0, clear=lambda m:setattr(m,'_supcon_pending',None), RX_BY_DOMAIN=rx_by_domain, LEVELS=())
    base = SimpleNamespace(terms=terms, capture_forward=lambda m,x,d,a:(m.forward_identity_only(x,domain_labels=d)['tx_logits'],{}))
    noise = SimpleNamespace(apply=lambda raw,step,mode:(raw.clone(),dict(applied=0)))
    fake = {}
    for name, attributes in [
        ('experiments.cvs_crossview_supcon41',dict(model=base,relations=SimpleNamespace())),
        ('experiments.cvs_fullsource_concat20',dict(supcon=supcon)),
        ('experiments.cvs_noise_shape_factorial43',dict(noise=noise))]:
        module=ModuleType(name);module.__dict__.update(attributes);fake[name]=module
    values=torch.arange(128*2*256,dtype=torch.float64).reshape(128,2,256)/100000
    raw=values.clone();ordinary=values+.1;y=torch.arange(128)%6;domains=torch.arange(128)%15
    sat=values[:64]+2;sat_y=(torch.arange(64)+2)%6;sat_d=(torch.arange(64)+4)%15
    group=[0]*48+[1]*4+[2]*4+[3]*8;pg=[.1,.2,.3,.4]
    observations=[dict(sample_id=[1,rx_by_domain[int(d)],1,i,i],scene='toy',state_end='LOS',receiver_lock_status='locked') for i,d in enumerate(sat_d)]
    allocation=dict(groups=group,counts=[48,4,4,8],p_g=pg,used_pool_indices=list(range(64)),origin_steps=[1]*64,origin_slots=list(range(0,128,2)))
    for method in ('POOL_RANDOM','QUALITY_STRAT'):
        calls=[];logits=[]
        m=SimpleNamespace(training=True,_quality_pool_bound=True,_quality_method=method,_feature_l_steps=1,_full_calls=0,
            _legacy_actual_L_input=ordinary.clone(),_supcon_pending=None,_full_L_packets=0,_supcon_calls=0,_legacy_calls=0,
            _legacy_extra_packets=0,_legacy_positive_KL_calls=0,_full_records=[],_full_coverage={},_supcon_applied=0,
            _shape_offset=4440,_shape_mode='NONE',_shape_NN=0,_shape_packets=0,_shape_records=[],_quality_records=[],_crossview_mode='ORD128')
        from collections import Counter
        m._full_coverage=Counter()
        def forward(x,domain_labels):
            calls.append((x.detach().clone(),domain_labels.detach().clone()))
            axis=torch.arange(6,dtype=x.dtype)
            z=(x.mean((1,2))[:,None]*(axis+.3)+domain_labels[:,None]*axis*.001+len(calls)*axis*.01).requires_grad_()
            logits.append(z);return dict(tx_logits=z)
        m.forward_identity_only=forward
        m._full_channel=SimpleNamespace(consume=lambda step,epoch,device:(sat,sat_y,sat_d,observations,allocation))
        first=forward(ordinary,domains)['tx_logits'];identity=F.cross_entropy(first,y,label_smoothing=.01)
        with patch.dict(sys.modules,fake):
            delta,_=_labeled_hook(m,model=m,labeled_output=dict(tx_logits=first),labels=y,
                extra=(domains,dict(rx_i=torch.tensor(rx_by_domain)[domains])),domains=domains,
                optimizer_step=1,identity_loss=identity,labeled_count=128,labeled_inputs=raw)
        mask=(torch.arange(128)+1)%2!=0;mixed_y=y.clone();mixed_y[mask]=sat_y
        mixed_d=domains.clone();mixed_d[mask]=sat_d;expected_view=ordinary.clone();expected_view[mask]=sat
        assert len(calls)==6 and all(torch.equal(calls[i][0],calls[j][0]) for i,j in ((0,1),(2,3),(4,5)))
        assert torch.equal(calls[2][0],expected_view) and torch.equal(calls[2][1],mixed_d)
        assert torch.equal(calls[4][0],raw) and torch.equal(calls[4][1],domains)
        common=terms(logits[0],logits[1],y,1.)[0];mixed=terms(logits[2],logits[3],mixed_y,1.)[0];raw_rd=terms(logits[4],logits[5],y,1.)[0]
        weighted,_,_=weighted_satellite_terms(logits[2][mask],logits[3][mask],sat_y,group,pg,method)
        plain=.5*sum(F.cross_entropy(z[mask],sat_y,label_smoothing=.01) for z in logits[2:4])
        expected=.5*common+.5*mixed+.25*raw_rd+.25*(weighted-plain)
        actual=delta+identity
        assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)
        ag=torch.autograd.grad(actual,logits,retain_graph=True);eg=torch.autograd.grad(expected,logits,retain_graph=True)
        assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(ag,eg))
        q=m._quality_records[0]
        assert abs(q['used_total_L']-q['original_total_L']-q['CE_correction_to_total'])<1e-10
        assert m._full_records[0]['L_forwards']==6 and m._shape_NN==256 and m._full_L_packets==512
        assert sum(m._full_coverage.values())==64 and q['counts']==[48,4,4,8]
        if method=='POOL_RANDOM':assert q['CE_correction_to_total']==0
    return ['actual_six_forward_hook_pack_labels_domains_loss_and_gradients']


def run(loss_only=False, pool_only=False, new_only=False):
    started = time.perf_counter()
    if new_only:
        passed=interface_checks()+hook_checks()
    else:
        passed = [] if pool_only else loss_checks()
        if not loss_only: passed += pool_checks()
    return dict(status='PASS', checks=len(passed), passed=passed,
                wall_seconds=time.perf_counter() - started,
                real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0, actual_channel_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-only', action='store_true')
    parser.add_argument('--pool-only', action='store_true')
    parser.add_argument('--new-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.loss_only, args.pool_only, args.new_only), allow_nan=False))
