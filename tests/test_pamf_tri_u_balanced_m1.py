import copy,json
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_balanced_m1 import plan
from experiments.pamf_tri_u_balanced_m1.objective import balanced_mask,js_divergence


def test_balance_retains_only_original_pseudolabels_and_ranking_changes_membership():
    raw=torch.tensor([True,True,True,True,True,True,True,False])
    pseudo=torch.tensor([0,0,0,0,1,1,2,2])
    risk=torch.tensor([.9,.2,.1,.3,.5,.1,.2,0.])
    mask=balanced_mask(raw,pseudo,risk,1)
    assert torch.equal(mask,torch.tensor([False,False,True,False,False,True,True,False]))
    relaxed=balanced_mask(raw,pseudo,risk,2)
    assert [int((relaxed & (pseudo==k)).sum()) for k in range(3)]==[2,2,1]
    assert not bool((relaxed & ~raw).any())
    assert not bool(balanced_mask(torch.zeros_like(raw),pseudo,risk,1).any())
    with pytest.raises(ValueError):balanced_mask(raw,pseudo,risk.float()[:-1],1)


def test_physical_rank_detects_distribution_change_without_reading_labels():
    full=torch.tensor([[4.,0.],[4.,0.],[4.,0.]])
    altered=torch.tensor([[4.,0.],[2.,2.],[0.,4.]])
    risk=js_divergence(full,altered)
    assert risk[0]<risk[1]<risk[2]
    assert torch.allclose(risk,js_divergence(altered,full))


def test_registered_source_only_three_seed_matrix():
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==18
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['balanced_recipe']['cap']=20
        with pytest.raises(ValueError):plan.validate_candidate(bad)
