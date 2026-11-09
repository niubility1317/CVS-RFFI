"""The acceleration must preserve every source sample and simulator record."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT=Path('E:/type10-7')
WORK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'local_artifacts/cric_dura_comparison_20261021/frozen_sources/G/code'),str(WORK/'code')]
from experiments.cric_dura_comparison.t1_practical import Practical as Original
from experiments.pamf_no_reference_completion.fast_practical import Practical as Fast


def inputs(n=24):
    p=ROOT/'local_artifacts/cric_dura_comparison_20261021/native_loop_e9cdde7e/cache'
    x=torch.from_numpy(np.load(p/'L_s_x.npy')[:n].copy())
    with np.load(p/'L_s_metadata.npz',allow_pickle=False) as a:
        meta={k:torch.from_numpy(a[k][:n].copy()) for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
    return x,meta


def test_bit_exact_cache_and_state(tmp_path):
    x,meta=inputs();c={'source_rxs':[1,3],'source_days':[1,2,3],'fast_as_cache':str(tmp_path/'cache'),'fast_as_min_free_bytes':0}
    original=Original(c);first=Fast(c);second=Fast(c)
    torch_state=torch.random.get_rng_state().clone();np_state=np.random.get_state()
    for epoch,step in ((1,1),(1,2),(14,3001),(50,11099)):
        a=original(x,meta,epoch,step);b=first(x,meta,epoch,step);d=second(x,meta,epoch,step)
        assert torch.equal(a,b) and torch.equal(a,d)
    assert first.cache_misses==4 and first.cache_hits==0 and second.cache_hits==4
    assert first.coverage==second.coverage==original.coverage
    assert first.sessions==second.sessions==original.sessions
    assert torch.equal(torch_state,torch.random.get_rng_state())
    now=np.random.get_state();assert np_state[0]==now[0] and np.array_equal(np_state[1],now[1]) and np_state[2:]==now[2:]
    altered=x.clone();altered[0,0,0]+=0.01
    second(altered,meta,1,1)
    assert second.cache_misses==1


def test_simultaneous_writers_leave_complete_cache(tmp_path):
    x,meta=inputs(8);c={'source_rxs':[1,3],'source_days':[1,2,3],'fast_as_cache':str(tmp_path/'cache'),'fast_as_min_free_bytes':0}
    expected=Original(c)(x,meta,2,300)
    def work(_):return Fast(c)(x,meta,2,300)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(torch.equal(expected,result) for result in pool.map(work,range(4)))
    cached=Fast(c)
    assert torch.equal(expected,cached(x,meta,2,300)) and cached.cache_hits==1
