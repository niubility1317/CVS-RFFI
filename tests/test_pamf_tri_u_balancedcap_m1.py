import copy,json
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_balancedcap_m1 import plan
from experiments.pamf_tri_u_balancedcap_m1.objective import balanced_mask


def test_cap_floor_and_ranking_preserve_original_acceptance():
    raw=torch.tensor([True]*14+[False]);pseudo=torch.tensor([0]*9+[1]*5+[0]);risk=torch.arange(15).float()
    assert int(balanced_mask(raw,pseudo,risk,1.125).sum())==10
    assert int(balanced_mask(raw,pseudo,risk,1.25).sum())==11
    selected=balanced_mask(raw,pseudo,risk,1.5)
    assert int(selected.sum())==12 and not bool((selected&~raw).any())
    assert torch.equal(selected[:7],torch.ones(7,dtype=torch.bool))
    with pytest.raises(ValueError):balanced_mask(raw,pseudo,risk,2.)


def test_registered_three_seed_source_only_matrix():
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==27
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['balanced_recipe']['cap']=4.
        with pytest.raises(ValueError):plan.validate_candidate(bad)
