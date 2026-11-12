import copy,json
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_rankdiversity_m1 import plan
from experiments.pamf_tri_u_rankdiversity_m1.objective import balanced_mask


def test_physical_rank_policies_have_same_class_counts_and_different_membership():
    raw=torch.tensor([True]*8+[False]);pseudo=torch.tensor([0]*6+[1]*3);risk=torch.tensor([0.,1.,2.,3.,4.,5.,0.,1.,9.])
    high=balanced_mask(raw,pseudo,risk,1,'high')
    mid=balanced_mask(raw,pseudo,risk,1,'middle')
    tails=balanced_mask(raw,pseudo,risk,1,'tails')
    assert torch.equal(high,torch.tensor([False,False,False,False,True,True,True,True,False]))
    assert torch.equal(mid,torch.tensor([False,True,True,False,False,False,True,True,False]))
    assert torch.equal(tails,torch.tensor([True,False,False,False,False,True,True,True,False]))
    for x in (high,mid,tails):assert not bool((x&~raw).any()) and int(x.sum())==4
    with pytest.raises(ValueError):balanced_mask(raw,pseudo,risk,1,'unregistered')


def test_registered_three_seed_source_only_matrix():
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==18
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['balanced_recipe']['rank']='random'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
