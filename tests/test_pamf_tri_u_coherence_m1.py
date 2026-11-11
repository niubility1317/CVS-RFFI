import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from experiments.pamf_tri_u_coherence_m1.objective import attenuated,masked_loss,scales
from experiments.pamf_tri_u_coherence_m1 import plan


def test_coherence_scale_and_matched_mean_control():
    q=torch.tensor([0.,.2,.8,1.])
    adaptive=scales(q,'adaptive',.25);uniform=scales(q,'uniform',.25)
    assert torch.allclose(adaptive,torch.tensor([.25,.4,.85,1.]))
    assert torch.allclose(uniform,adaptive.mean().expand_as(adaptive))
    with pytest.raises(ValueError):scales(torch.tensor([float('nan')]),'adaptive',.25)


def test_branch_hook_is_temporary_and_preserves_other_paths():
    ref=nn.Linear(3,3,bias=False);freq=nn.Linear(3,3,bias=False);stats=nn.Linear(3,3,bias=False)
    core=SimpleNamespace(f_proj=freq,freq_stats_proj=stats)
    encoder=SimpleNamespace(response_projection=ref,core=SimpleNamespace(id_backbone=core),sequence_pool=nn.Identity())
    model=SimpleNamespace(id_backbone=SimpleNamespace(encoder=encoder))
    x=torch.randn(2,3);gamma=torch.tensor([.25,.8]);base=ref(x);before=freq(x)
    with attenuated(model,gamma,frequency=False):
        assert torch.allclose(ref(x),base*gamma[:,None])
        assert torch.allclose(freq(x),before)
    assert torch.allclose(ref(x),base)
    with pytest.raises(RuntimeError):
        with attenuated(model,gamma,frequency=True):
            assert torch.allclose(freq(x),before*gamma[:,None])
            raise RuntimeError('test cleanup')
    assert torch.allclose(freq(x),before)


def test_pseudo_label_loss_masks_rejected_samples():
    logits=torch.randn(3,4,requires_grad=True);labels=torch.tensor([0,1,2]);accepted=torch.tensor([True,False,True])
    masked_loss(logits,labels,accepted).backward()
    assert logits.grad[1].count_nonzero()==0 and logits.grad[[0,2]].abs().sum()>0


def test_own_data_contract_and_four_process_limit(monkeypatch):
    from experiments.pamf_repl_completion import runner as capacity
    from experiments.pamf_tri_u_coherence_m1.runner import admits_shared
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source-handle context unavailable')
    context=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows()
    assert len(rows)==12
    for row in rows:
        c=plan.configuration(context,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
    monkeypatch.setattr(capacity,'capacity',lambda gpu:({1,2,3},20000))
    monkeypatch.setattr(capacity,'processes',lambda:{1:7,2:7,3:7})
    assert admits_shared(7,{},6000)
    monkeypatch.setattr(capacity,'processes',lambda:{1:7,2:7,3:7,4:7})
    assert not admits_shared(7,{},6000)
