import random
import pytest
from cvsrffi.game_tracking import data


def records():
    return [dict(index=i,tx=t,rx=r,day=d,capture_group=f'{t}:{r}:{d}',sample_id=f'id{i}',role='train')
            for i,(t,r,d,k) in enumerate((t,r,d,k) for t in range(6) for r in [1,3,4,6,8]
                                         for d in [1,2,3] for k in range(4))]


def test_reference_balanced_covers_all_ninety_containers_without_rng_side_effect():
    assert hasattr(data,'balanced_reference_indices'), 'balanced source reference is missing'
    before=random.getstate()
    indices,coverage=data.balanced_reference_indices(records(),1,17)
    assert len(indices)==90 and len(set(indices))==90
    selected=[records()[i] for i in indices]
    assert len({(r['tx'],r['rx'],r['day']) for r in selected})==90
    assert coverage['domains']==15 and coverage['txs']==6 and coverage['valid']
    assert random.getstate()==before
    assert data.balanced_reference_indices(records(),1,17)[0]==indices
    assert data.balanced_reference_indices(records(),1,18)[0]!=indices


def test_reference_rejects_hidden_target_and_missing_container_metadata():
    assert hasattr(data,'balanced_reference_indices')
    for override in [dict(role='target'),dict(tx=-1),dict(capture_group='')]:
        rows=records();rows[0].update(override)
        with pytest.raises(ValueError): data.balanced_reference_indices(rows,1,17)


def test_v2_reference_uses_whole_containers_and_keeps_h1_sides_disjoint():
    assert hasattr(data,'audit_indices_v2')
    from cvsrffi.game_tracking.config import parse_args
    source=data.build_source(parse_args(['--output_dir','unused','--game_synthetic']))
    indexes=data.audit_indices_v2(source,per_capture=2,seed=7)
    assert len(indexes['lag_reference'])==108
    assert len(indexes['gradient_reference'])==54
    fit={source.train[i][3]['capture_group'] for i in indexes['response_fit']}
    mon={source.train[i][3]['capture_group'] for i in indexes['response_monitor']}
    assert not fit & mon and len(fit)==len(mon)==27
