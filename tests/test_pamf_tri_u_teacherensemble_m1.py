import copy,json
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_teacherensemble_m1 import plan
from experiments.pamf_tri_u_teacherensemble_m1.objective import ensemble_kl


def test_teacher_ensemble_is_detached_and_excludes_rejected_packets():
    full=torch.tensor([[4.,1.,0.],[0.,3.,1.],[1.,0.,3.]],requires_grad=True)
    view=torch.tensor([[0.,4.,1.],[0.,1.,3.],[1.,3.,0.]],requires_grad=True)
    logits=torch.randn(3,3,requires_grad=True);accepted=torch.tensor([True,False,True])
    loss=ensemble_kl(logits,full,view,accepted,2.,.5);loss.backward()
    assert full.grad is None and view.grad is None
    assert torch.count_nonzero(logits.grad[1])==0 and torch.count_nonzero(logits.grad[[0,2]])>0
    assert not torch.allclose(ensemble_kl(logits,full,view,accepted,2.,.5),
                              ensemble_kl(logits,full,full,accepted,2.,.5))
    assert abs(float(ensemble_kl(full.detach(),full,full,accepted,2.,.5)))<1e-6
    with pytest.raises(ValueError):ensemble_kl(logits,full,view,accepted.float(),2.,.5)


def test_registered_source_only_matrix():
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==18
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['teacher_branch']['mix']=.9
        with pytest.raises(ValueError):plan.validate_candidate(bad)
