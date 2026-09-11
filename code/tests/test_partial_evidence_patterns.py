import sys
from pathlib import Path
from dataclasses import replace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_anchored_cache import toy_rows
from cvsrffi.partial_evidence_fit import PartialEvidenceHead,fit_partial_evidence
from cvsrffi.mask_pattern_calibration import PatternSpec,PatternCalibrator
from cvsrffi.partial_gaussian_head import gaussian_scores


def test_e_shared_marginal_reuses_lowrank_and_ignores_missing_values():
    torch.set_num_threads(2);rows=toy_rows(dimension=9)
    spec=PatternSpec(('t','f','pa'),(3,3,3))
    head=fit_partial_evidence(rows,spec,rank=2)
    mask=spec.mask('missing_t',len(rows));z=rows.h.clone();z[~mask]=float('nan')
    out=head(z,mask,rows.quality)
    other=rows.h.clone();other[~mask]=1e20
    torch.testing.assert_close(out['scores'],head(other,mask,rows.quality)['scores'])
    diag=head.diagonal[None]+head.error(rows.quality)
    covariance=torch.diag_embed(diag)+head.factor@head.factor.T
    dense=gaussian_scores(z,head.means,covariance,mask)
    torch.testing.assert_close(out['scores'],dense['scores'],atol=1e-4,rtol=1e-4)
    assert out['solver_calls']==1
    assert out['logdet'].shape==out['scores'].shape
    assert not any(p.requires_grad for p in head.parameters())
    restored=PartialEvidenceHead.from_export(head.export_state())
    torch.testing.assert_close(restored(z,mask,rows.quality)['scores'],out['scores'])
    with pytest.raises(ValueError,match='L_s'):
        fit_partial_evidence(replace(rows,identity=replace(rows.identity,role='V')),spec,rank=2)


def test_pattern_groups_do_not_merge_equal_dimensions_and_unknown_defers():
    n=24;labels=torch.zeros(n,dtype=torch.long)
    scores=torch.tensor([[2.,0.]]).repeat(n,1);mahal=torch.ones_like(scores)
    quality=torch.zeros(n,3);count=torch.full((n,),6)
    ids=[f'id{i}' for i in range(n)]
    cal=PatternCalibrator(min_group_samples=20).fit(scores,mahal,count,quality,labels,
                                                   ['missing_t']*n,ids,role='V')
    known=cal.predict(scores,mahal,count,quality,['missing_t']*n)
    other=cal.predict(scores,mahal,count,quality,['missing_f']*n)
    assert known['calibration_available'].all()
    assert not other['accepted'].any() and set(other['status'])=={'defer'}
    none=cal.predict(scores,mahal,torch.zeros(n,dtype=torch.long),quality,['all_missing']*n)
    assert not none['accepted'].any()
    repeat=PatternCalibrator(min_group_samples=20).fit(scores,mahal,count,quality,labels,
                                                      ['missing_t']*n,['one_id']*n,role='V')
    assert not repeat.predict(scores,mahal,count,quality,['missing_t']*n)['calibration_available'].any()
    restored=PatternCalibrator.from_state_dict(cal.state_dict())
    assert restored.predict(scores,mahal,count,quality,['missing_t']*n)['accepted'].equal(known['accepted'])


def test_pattern_masks_use_named_actual_blocks():
    spec=PatternSpec(('t','f','pa'),(2,3,4))
    assert spec.mask('missing_t',1).sum()==7
    assert spec.mask('missing_f',1).sum()==6
    assert spec.mask('all_missing',1).sum()==0
    with pytest.raises(ValueError,match='pattern'):spec.mask('iq_time_gap',1)
