import importlib
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.pamf_phase3_open_world.star_open_world_episode')
    except ModuleNotFoundError:pytest.fail('open-world episode sampler is not implemented')


def metadata():
    rows=np.asarray([(y,r,d) for y in range(6) for r in (1,3,4,6,8) for d in (1,2,3) for _ in range(70)],np.int64)
    return rows[:,0],rows[:,1],rows[:,2],np.asarray([f'opaque-{i:05}' for i in range(6300)]),np.repeat(np.arange(15,dtype=np.int64),7)


def test_cross_rx_day_support_query_are_disjoint_and_holdouts_rotate():
    m=module();y,rx,day,ids,oe=metadata();sampler=m.EpisodeSampler(y,rx,day,ids,oe)
    for step,want in [(1,(0,1,1)),(6,(5,1,1)),(7,(0,3,1)),(31,(0,1,2)),(90,(5,8,3))]:
        s=sampler.sample(step)
        assert (s['held_class'],s['held_rx'],s['held_day'])==want
        assert s['native_L'].shape==(128,) and s['OE'].shape==(64,)
        assert s['support'].shape==s['query'].shape==(6,16)
        assert np.all(rx[s['query']]==want[1]) and np.all(day[s['query']]==want[2])
        assert np.all(rx[s['support']]!=want[1]) and np.all(day[s['support']]!=want[2])
        assert not set(ids[s['support']].ravel())&set(ids[s['query']].ravel())
        for c in range(6):
            assert np.all(y[s['support'][c]]==c) and np.all(y[s['query'][c]]==c)
            assert len(set(s['support'][c]))==len(set(s['query'][c]))==16
        assert s['support'].dtype==s['query'].dtype==s['OE'].dtype==np.int64


def test_three_rows_share_exact_step_trajectory_and_independent_slot_rng():
    m=module();data=metadata();samplers=[m.EpisodeSampler(*data) for _ in range(3)]
    for step in (1,2,17,91,2220):
        samples=[sampler.sample(step) for sampler in samplers]
        for key in samples[0]:
            for s in samples[1:]:np.testing.assert_array_equal(s[key],samples[0][key])
        for arm,k in [('S0',6),('S1',6),('M',5)]:
            slots=m.active_slots(samples[0],arm)
            assert len(slots)==k and (arm!='M' or samples[0]['held_class'] not in slots)
        np.testing.assert_array_equal(samples[0]['slot_order'],samplers[0].sample(step)['slot_order'])
    changed=m.EpisodeSampler(*data,slot_seed=99)
    first=samplers[0].sample(5);other=changed.sample(5)
    for key in ('native_L','support','query','OE'):np.testing.assert_array_equal(first[key],other[key])
    assert not np.array_equal(first['slot_order'],other['slot_order'])


def test_oe_domains_are_uniform_before_within_domain_iid_sampling():
    m=module();y,rx,day,ids,_=metadata();oe=np.concatenate([np.full(i+1,i,np.int64) for i in range(15)])
    sampler=m.EpisodeSampler(y,rx,day,ids,oe)
    counts=np.bincount(np.concatenate([oe[sampler.sample(i)['OE']] for i in range(1,301)]),minlength=15)
    assert counts.min()>1000 and counts.max()<1500


def test_invalid_roles_duplicates_and_insufficient_cells_fail_without_padding():
    m=module();data=metadata();y,rx,day,ids,oe=data
    with pytest.raises(TypeError):m.EpisodeSampler(*data,target_truth='forbidden')
    with pytest.raises(ValueError):m.EpisodeSampler(y,rx,day,np.full(6300,'same'),oe)
    keep=np.ones(6300,bool);keep[:55]=False
    with pytest.raises(ValueError,match='16'):m.EpisodeSampler(y[keep],rx[keep],day[keep],ids[keep],oe)
    with pytest.raises(ValueError):m.EpisodeSampler(y,rx,day,ids,np.zeros(20,np.int64))
    sampler=m.EpisodeSampler(*data)
    with pytest.raises(ValueError):sampler.sample(0)
    with pytest.raises(TypeError):sampler.sample(1,arm='M')
