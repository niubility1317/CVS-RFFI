import sys
from pathlib import Path
from dataclasses import replace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_anchored_cache import toy_rows
from cvsrffi.anchored_cache import TENSOR_FIELDS
from cvsrffi.anchored_fit import FitConfig, physical_batches, paired_weights, keep_loss, fit_expert


def test_physical_batches_cover_once_and_all_groups():
    rows=toy_rows(per_group=4)
    batches=list(physical_batches(rows,2,8))
    assert len(batches)==2
    physical=[]
    for take in batches:
        b=rows.take(take);ids=set(b.physical_ids)
        assert len(ids)==3*3*2
        assert len(b)==2*len(ids)
        physical.extend(ids)
    assert len(set(physical))==len(physical)==36
    assert [x.tolist() for x in batches]!=[x.tolist() for x in physical_batches(rows,2,9)]


def test_formal_6300_physical_budget_and_fold_batch_sizes():
    from cvsrffi.anchored_cache import FeatureRows
    identity=toy_rows().identity;metadata=[(y,rx,d,i) for y in range(6) for rx in (1,3,4,6,8) for d in (1,2,3) for i in range(70)]
    keys=tuple(':'.join(map(str,m)) for m in metadata for _ in range(2));n=len(keys)
    labels=torch.tensor([m[0] for m in metadata for _ in range(2)])
    rx=torch.tensor([m[1] for m in metadata for _ in range(2)]);day=torch.tensor([m[2] for m in metadata for _ in range(2)])
    rows=FeatureRows(torch.zeros(n,8),torch.zeros(n,6),torch.ones(n,3),torch.ones(n,8,dtype=torch.bool),labels,keys,
                     ('clean','leo_clear_weak')*len(metadata),rx,day,torch.ones(n,dtype=torch.bool),identity)
    for receivers,expected_rows in (((1,3,4,6,8),360),((1,3,4,6),288),((1,3,4),216)):
        subset=rows.take([i for i,r in enumerate(rx.tolist()) if r in receivers]);batches=list(physical_batches(subset,2,392005))
        assert len(batches)==35 and all(len(b)==expected_rows for b in batches)
        indices=torch.cat(batches);assert len(indices.unique())==len(subset)
        assert len(set(subset.physical_ids))==6*len(receivers)*3*70


def test_view_duplication_cannot_change_physical_weight():
    rows=toy_rows();weights=paired_weights(rows,True)
    assert abs(float(weights.sum())-1)<1e-6
    for pid in set(rows.physical_ids):
        take=torch.tensor([p==pid for p in rows.physical_ids])
        assert abs(float(weights[take].sum())-1/36)<1e-6
    idx=[i for i,v in enumerate(rows.view_ids) if v!='clean']
    data={key:torch.cat((getattr(rows,key),getattr(rows,key)[idx])) for key in TENSOR_FIELDS}
    extended=replace(rows,**data,physical_ids=rows.physical_ids+tuple(rows.physical_ids[i] for i in idx),
                     view_ids=rows.view_ids+tuple('leo_rain_weak' for _ in idx))
    extended.validate()
    expanded=paired_weights(extended,True)
    loss=rows.h.square().sum(-1)
    expanded_loss=torch.cat((loss,loss[idx]))
    torch.testing.assert_close((weights*loss).sum(),(expanded*expanded_loss).sum())


def test_keep_only_uses_correct_valid_source_rows():
    s0=torch.tensor([[0.,2.],[3.,0.]])
    sg=torch.tensor([[3.,0.],[1.,2.]],requires_grad=True);y=torch.tensor([0,0])
    loss=keep_loss(s0,sg,y,gamma_max=1.)
    loss.backward()
    assert torch.equal(sg.grad[0],torch.zeros(2)) and sg.grad[1].abs().sum()>0
    zero=keep_loss(s0,sg,y,gamma_max=1.,valid=torch.zeros(2,dtype=torch.bool))
    assert float(zero.detach())==0


def test_fit_is_cache_only_and_consumes_config(tmp_path):
    torch.set_num_threads(2)
    rows=toy_rows();torch.manual_seed(44);w=torch.randn(3,8)
    config=FitConfig(candidate='A4',epochs=2,rank=2)
    result=fit_expert(rows,(1,3),config,14,tmp_path/'fit',w0=w,tau0=10.,validation_rx=(4,))
    assert result.train_rx==(1,3) and len(result.history)==2
    assert result.history[-1]['optimizer_steps']==4
    assert result.history[-1]['physical_visits_total']==48
    assert result.history[-1]['view_visits_total']==96
    assert result.history[-1]['grad_norm_ce']>=0
    assert result.stop_status in {'converged','budget_exhausted'}
    assert (tmp_path/'fit/head_fit_history.csv').is_file()
    with pytest.raises(ValueError,match='unknown'):FitConfig.parse({'candidate':'A4','unused_option':1})
    with pytest.raises(ValueError,match='overlap'):
        fit_expert(rows,(1,3),config,14,tmp_path/'bad',w0=w,tau0=10.,validation_rx=(3,))
    with pytest.raises(ValueError,match='L_s'):
        fit_expert(replace(rows,identity=replace(rows.identity,role='V')),(1,),config,14,tmp_path/'bad2',w0=w,tau0=10.)


def test_imbalanced_physical_group_is_error():
    rows=toy_rows();bad=rows.take(list(range(2,len(rows))))
    with pytest.raises(ValueError,match='balanced'):list(physical_batches(bad,2,1))
