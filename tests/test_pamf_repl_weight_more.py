import contextlib
import pytest

from experiments.pamf_repl_weight_more import plan


def test_more_rows_are_unique_clean_and_source_matched():
    rows=plan.candidate_rows()
    assert len(rows)==16 and len({r['row_id'] for r in rows})==16
    assert {r['setting'] for r in rows}=={'ten_class','source_rx_6'}
    assert all(r['views']==['clean'] and r['model_seed']==2026092701 for r in rows)


def test_zero_extrapolation_keeps_ce_dose():
    c={'hard_pl':{'weight':6.},'hard_pl_gradient_guard':{'weight':6.},
       'hard_pl_joint_contrast':{'weight':.05},
       'hard_pl_receiver_extrapolation':{'ordinary_CE_coefficient':.25,'extrapolated_CE_coefficient':.25},
       'hard_pl_mixstyle':{'probability_per_level':.5}}
    r=plan.apply_recipe(c,'extrap0')['hard_pl_receiver_extrapolation']
    assert r['ordinary_CE_coefficient']==.5 and r['extrapolated_CE_coefficient']==0.


def test_kl_factor_changes_only_kl_term(monkeypatch):
    import torch
    from experiments.pamf_paper_completion import model
    from experiments.pamf_ssl_research import hard_pl_objective as obj
    from experiments.pamf_repl_weight_more.objective import installed
    @contextlib.contextmanager
    def fake(*args,**kwargs):yield
    monkeypatch.setattr(model,'installed',fake)
    a=torch.tensor([[2.,0.],[0.,2.]],requires_grad=True)
    b=torch.tensor([[0.,2.],[2.,0.]],requires_grad=True)
    y=torch.tensor([0,1]);accepted=torch.tensor([True,True])
    full,ce,kl=obj.masked_pair(a,b,y,accepted,1.)
    cfg={'tuning_recipe':{'kl_factor':.5},'hard_pl_smoke':True,'hard_pl':{'weight':6.},
         'hard_pl_receiver_extrapolation':{'extrapolated_CE_coefficient':.25}}
    with installed(cfg,{}):
        half,ce2,kl2=obj.masked_pair(a,b,y,accepted,1.)
    assert ce2.item()==pytest.approx(ce.item()) and kl2.item()==pytest.approx(kl.item())
    assert half.item()==pytest.approx((ce+.5*kl).item())
    assert full.item()==pytest.approx((ce+kl).item())


def test_style_zero_is_not_applied_in_real_forward():
    import torch
    from torch import nn
    from experiments.pamf_ssl_research.hard_pl_mixstyle_objective import style_hooks
    net=nn.Module();net.feature=nn.Identity()
    x=torch.arange(8,dtype=torch.float32).reshape(2,2,2)
    recipe={'levels':['feature'],'eps':1e-6,'probability_per_level':0.}
    with style_hooks(net,'',recipe,torch.tensor([1,0]),torch.tensor([True,True]),torch.ones(2,1,1)*.5,
                     force=False,enabled_by_level={'feature':False}) as applied:
        y=net.feature(x)
    assert torch.equal(x,y) and applied['feature']==0


def test_more_preflight_is_disjoint_from_live_recovery(monkeypatch):
    from experiments.pamf_branch_training import publish as base
    from experiments.pamf_repl_weight_more import publish
    got={}
    monkeypatch.setattr(base,'publish',lambda *args,**kwargs:got.update(script=base.REMOTE))
    publish.publish('repo','out','context','ssh')
    assert "rt['allowed_gpus']=[2,6]" in got['script']
    compile(got['script'].replace('PACKAGE','{}'),'<remote preflight>','exec')
