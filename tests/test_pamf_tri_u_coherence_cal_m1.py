import copy,json
from contextlib import contextmanager
from pathlib import Path
import pytest,torch
from experiments.pamf_tri_u_coherence_cal_m1 import objective,plan,source
from experiments.pamf_tri_u_coherence_m1 import objective as base


def test_source_range_expansion_and_equal_dose_controls():
    q=torch.tensor([1.,.997,.99,.5]);a=objective.scales(q,'expanded',.25)
    assert torch.allclose(a,torch.tensor([1.,.7,.25,.25]),atol=2e-6)
    assert torch.allclose(objective.scales(q,'expanded_uniform',.25),a.mean().expand_as(a))
    assert torch.equal(objective.scales(q,'identity',.25),torch.ones_like(q))
    with pytest.raises(ValueError):objective.scales(torch.tensor([float('nan')]),'expanded',.25)


def test_contract_and_scope_restoration(monkeypatch):
    path=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not path.exists():pytest.skip('Local source-handle context unavailable')
    ctx=json.loads(path.read_text(encoding='utf8'));rows=plan.candidate_rows();assert len(rows)==15
    before=base.scales,base.masked_loss
    @contextmanager
    def dummy(*args,**kwargs):yield
    monkeypatch.setattr(source.model,'installed',dummy)
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        with source.installed(c,{}):
            assert base.scales is objective.scales and base.masked_loss is objective.masked_loss
        assert (base.scales,base.masked_loss)==before
