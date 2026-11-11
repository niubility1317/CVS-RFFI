import copy,json
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_ranktemp_m1 import plan
from experiments.pamf_tri_u_ranktemp_m1.objective import js_divergence,balanced_mask


def test_temperature_softens_saturated_branch_predictions():
    full=torch.tensor([[12.,0.,0.],[12.,0.,0.]])
    view=torch.tensor([[11.,0.,0.],[9.,0.,0.]])
    risk=js_divergence(full,view,4.)
    assert 0<risk[0]<risk[1] and risk[1]>js_divergence(full,view,1.)[1]
    assert torch.allclose(risk,js_divergence(view,full,4.))
    with pytest.raises(ValueError):js_divergence(full,view,0.)


def test_fixed_pseudo_class_cap_and_registered_matrix():
    raw=torch.tensor([True,True,True,True,True,False]);pseudo=torch.tensor([0,0,0,1,1,1]);risk=torch.tensor([.8,.1,.3,.4,.5,0.])
    assert torch.equal(balanced_mask(raw,pseudo,risk,1),torch.tensor([False,True,True,True,True,False]))
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==21
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['balanced_recipe']['temperature']=3.
        with pytest.raises(ValueError):plan.validate_candidate(bad)
