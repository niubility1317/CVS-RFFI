import copy,json
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_consensus_m1.objective import select_mask
from experiments.pamf_tri_u_consensus_m1 import plan


def test_random_control_has_same_size_and_only_originally_accepted():
    raw=torch.tensor([True,True,False,True,True,False,True])
    stable=torch.tensor([True,False,True,True,False,False,False])
    actual=select_mask(raw,stable,False,123);random=select_mask(raw,stable,True,123)
    assert torch.equal(actual,raw&stable) and int(random.sum())==int(actual.sum())
    assert not bool((random&~raw).any())
    assert torch.equal(random,select_mask(raw,stable,True,123))
    with pytest.raises(ValueError):select_mask(raw,stable.float(),False,1)


def test_registered_source_only_matrix():
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source-handle context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==18
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['consensus_recipe']['weak_threshold']=.1
        with pytest.raises(ValueError):plan.validate_candidate(bad)
