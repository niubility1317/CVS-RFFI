from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools'),str(ROOT/'experiments/adv3b02_xuc/code')]
from build_d92_confirmation_data import assignments


@pytest.mark.parametrize('support,query,available',[(36,30,200),(30,30,184)])
def test_allocations_are_disjoint_and_deterministic(support,query,available):
    records=[(0,i) for i in range(available)]
    kwargs=dict(seed=2026092705,receiver='20-19',class_id='test',support_pool_size=support,query_size=query)
    result=assignments(records,**kwargs)
    assert result==assignments(records,**kwargs)
    all_records=[]
    for roles in result.values():
        assert len(roles['support_pool'])==support
        assert len(roles['query'])==query
        all_records+=roles['support_pool']+roles['query']
    assert len(all_records)==len(set(all_records))==3*(support+query)
    assert set(all_records)<=set(records)


def test_defaults_preserve_existing_allocation_and_insufficient_records_fail():
    kwargs=dict(seed=1,receiver='rx',class_id='tx')
    assert assignments(list(range(200)),**kwargs)==assignments(list(range(200)),support_pool_size=36,query_size=30,**kwargs)
    with pytest.raises(ValueError,match='180'):
        assignments(list(range(179)),support_pool_size=30,query_size=30,**kwargs)
    with pytest.raises(ValueError,match='Repeated'):
        assignments([0]*200,**kwargs)
    with pytest.raises(ValueError,match='Positive integer'):
        assignments(list(range(200)),support_pool_size=True,**kwargs)
