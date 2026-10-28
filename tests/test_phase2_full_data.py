import numpy as np
import pytest
from experiments.pamf_phase2_practical.full_data import cell_plan,quota_matrix,balanced_query_counts,rows


def test_independent_eight_mixed_three_pure_after_query_history_exclusion():
    available=[list(range(n)) for n in [800,550,1100,900]]
    # The first 500 observations in a date are historical query, never support.
    support=[list(range(500,n)) for n in [800,550,1100,900]]
    sizes=[150]*8+[500]*3
    plans,quotas=cell_plan(support,available,392005,sizes)
    all_s=[];all_q=[]
    for p,nq in zip(plans,sizes):
        assert len(p['support'])==24 and len(p['query'])==nq
        assert all(si>=500 for di,si in p['support'])
        all_s.extend(map(tuple,p['support']));all_q.extend(map(tuple,p['query']))
    assert len(set(all_s+all_q))==2964
    assert plans==cell_plan(support,available,392005,sizes)[0]
    assert np.asarray(quotas['query_per_domain_date']).sum(1).tolist()==sizes


def test_legacy_query_can_never_fill_insufficient_support():
    with pytest.raises(ValueError,match='Insufficient'):
        cell_plan([[0]]*4,[list(range(1000))]*4,1,[150]*8+[500]*3)


def test_query_capacity_checked_after_reserving_all_support():
    with pytest.raises(ValueError,match='Insufficient'):
        cell_plan([list(range(100))]*4,[list(range(100))]*4,1,[150]*8+[500]*3)


def test_date_quota_transport_handles_scarce_and_absent_days():
    q=balanced_query_counts([0,50,100,900],500)
    m=np.asarray(quota_matrix([100,100,300],q))
    assert m.sum(1).tolist()==[100,100,300]
    assert m.sum(0).tolist()==q
    assert m[:,0].sum()==0


def test_raw_mapping_is_exact_and_does_not_normalize():
    x=np.zeros((2,256,2),np.float64);x[1,0,0]=1
    assert len(np.unique(rows(x)))==2
    with pytest.raises(ValueError,match='representation'):
        rows(x.astype(np.float32))


def test_a_stage_opens_no_support_or_future_class_data(tmp_path):
    import json
    from experiments.pamf_phase2_practical.full_io import load_query
    capsule=tmp_path/'N00';capsule.mkdir()
    (capsule/'manifest.json').write_text(json.dumps(dict(storage='support-plus-shared-query-v1',
        protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',query_count=3)))
    np.save(capsule/'query_iq.npy',np.zeros((3,2,256),np.float32))
    np.save(capsule/'query_ids.npy',np.array(['opaque-a','opaque-b','opaque-c']))
    # A must work without any target support, split, or later registration file.
    result=load_query(capsule)
    assert set(result)=={'manifest','query_x','query_ids'}
    assert result['query_x'].shape==(3,2,256)
