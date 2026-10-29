import copy
import importlib
import numpy as np
import pytest


def core():
    try: return importlib.import_module('experiments.pamf_phase3_open_world.star_binary_graph_core')
    except ModuleNotFoundError: pytest.fail('binary graph core is not implemented')


def fixture():
    x=np.eye(2,160,dtype=np.float32); l=np.zeros((2,6),np.float32); l[0,0]=10
    return x,l,np.asarray([1.,-1.]),np.asarray(['z','a'])


def test_two_node_spd_solution_matches_hand_calculation_and_maximum_principle():
    c=core();x,l,m,ids=fixture();s,d=c.fit(x,l,m,ids)
    a=(10+np.tanh(1))/(11+2*np.exp(-1))
    np.testing.assert_allclose(s['f'],[a,-a],rtol=0,atol=1e-14)
    np.testing.assert_allclose(s['sigma'],np.sqrt(2),rtol=0,atol=1e-14)
    assert d['positive_count']==d['negative_count']==d['edge_count']==1
    assert d['solve_residual_linf']<1e-12 and d['energy_after']<=d['energy_before']
    assert np.max(abs(s['f']))<=1+1e-12 and not s['skipped']


def test_q_is_row_batch_order_independent_and_does_not_modify_frozen_u():
    c=core();x,l,m,ids=fixture();s,_=c.fit(x,l,m,ids); before=copy.deepcopy(s)
    q=np.concatenate([x,x[::-1]]);mq=np.asarray([0.4,-0.2,0.7,-1.])
    batch=c.margins(q,mq,s)
    np.testing.assert_array_equal(batch,np.concatenate([c.margins(q[i:i+1],mq[i:i+1],s) for i in range(4)]))
    perm=np.asarray([3,1,0,2]);np.testing.assert_array_equal(c.margins(q[perm],mq[perm],s),batch[perm])
    for k in s:
        if isinstance(s[k],np.ndarray):np.testing.assert_array_equal(s[k],before[k])
        else:assert s[k]==before[k]
    s2,_=c.fit(x[::-1],l[::-1],m[::-1],ids[::-1])
    np.testing.assert_array_equal(c.margins(q,mq,s2),batch)


@pytest.mark.parametrize('positive',[True,False])
def test_empty_seed_side_is_exact_da0_identity(positive):
    c=core();x,l,m,ids=fixture();l[:]=0
    if positive:l[:,0]=10;m[:]=1
    else:m[:]=-1
    s,d=c.fit(x,l,m,ids)
    assert s['skipped'] and d['status']=='SKIPPED_EMPTY_SEED_SIDE'
    np.testing.assert_array_equal(s['f'],np.tanh(m))
    query=np.asarray([5.,-5.]);np.testing.assert_array_equal(c.margins(x,query,s),np.tanh(query))


def test_q_neighbor_ties_use_opaque_ids_not_u_row_order():
    c=core();x=np.zeros((12,160),np.float32);x[:,0]=1
    ids=np.asarray([f'id{i:02}' for i in range(11,-1,-1)])
    s=dict(schema=c.STATE_SCHEMA,unit=x,ids=ids,sigma=np.ones(12),f=np.arange(11,-1,-1,dtype=np.float64)/11,skipped=False)
    value=c.margins(x[:1],np.zeros(1),s)
    np.testing.assert_allclose(value,[45/121],rtol=0,atol=1e-15)


def test_invalid_graph_inputs_fail_without_silent_conversion():
    c=core();x,l,m,ids=fixture()
    for args in [(x.astype(np.float64),l,m,ids),(x,l,m.astype(np.float32),ids),(x,l,m,np.asarray(['a','a']))]:
        with pytest.raises(ValueError):c.fit(*args)
    s,_=c.fit(x,l,m,ids);s['f'][0]=np.nan
    with pytest.raises(ValueError):c.margins(x,m,s)
