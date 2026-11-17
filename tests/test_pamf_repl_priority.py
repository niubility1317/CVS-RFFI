import pytest
import torch
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from experiments.pamf_repl_priority.projection import reference_loss


def run_case(labeled, unlabeled, ref, p, cap=.5):
    return reference_loss(labeled, unlabeled, ref, [('net.t2.weight', p)], cap=cap)


def test_reference_is_main_direction_and_backward_matches():
    p = torch.nn.Parameter(torch.tensor([0., 0.]))
    ref = p[0]; labeled = -4*p[0] + 3*p[1]; unlabeled = ref
    value, details, gradients, stats = run_case(labeled, unlabeled, ref, p)
    assert torch.allclose(gradients[0], torch.tensor([1., .5]))
    assert stats['t2_conflict'] == 1 and stats['t2_capped'] == 1
    assert stats['t2_reference_remainder_dot_after'] >= -1e-9
    assert stats['t2_applied_remainder_norm'] <= .500001
    value.backward()
    assert torch.allclose(p.grad, gradients[0])
    assert details['t2']['L_norm'] > 0


def test_zero_reference_falls_back_to_unmodified_joint_gradient():
    p = torch.nn.Parameter(torch.tensor([0., 0.]))
    value, _, gradients, stats = run_case(2*p[0], -3*p[0]+p[1], 0*p[0], p)
    assert torch.allclose(gradients[0], torch.tensor([-1., 1.]))
    assert stats['t2_zero_reference'] == 1
    value.backward()
    assert torch.allclose(p.grad, gradients[0])


def test_reference_rejects_nonfinite_gradients():
    p = torch.nn.Parameter(torch.tensor([0.]))
    with pytest.raises(ValueError, match='finite'):
        run_case(p[0]*float('nan'), p[0], p[0], p)


def test_registered_rows_and_scoped_configuration(tmp_path):
    from experiments.pamf_repl_priority import plan
    from experiments.pamf_repl_component_guard import plan as base
    rows=plan.candidate_rows()
    assert len(rows)==20
    assert sum(r['setting']=='ten_class' for r in rows)==8
    assert sum(r['setting']=='source_rx_6' for r in rows)==12
    with plan.scoped():
        with pytest.raises(FileNotFoundError):
            base.configuration({},rows[0],tmp_path,'commit')


def test_exact_kl_reference_matches_real_objective():
    from experiments.pamf_repl_priority import objective
    from experiments.pamf_ssl_research import hard_pl_objective as hard
    @contextmanager
    def fake_scope(c,holder):yield
    y=torch.tensor([0,1,0,1]); accepted=torch.tensor([True,True,False,True])
    physical=torch.tensor([False,True,False,True]); aux=torch.tensor(0.)
    torch.manual_seed(7)
    logits=[torch.randn(4,2,requires_grad=True) for _ in range(6)]
    holder={'objective':SimpleNamespace(parts={})}
    c={'tuning_recipe':{'kl_factor':2.},'hard_pl':{'weight':3.}}
    with patch.object(objective.original,'installed',fake_scope):
        with objective.installed(c,holder):
            changed,_=hard.mechanism_loss(*logits,y,accepted,physical,aux)
            reference=holder['objective'].parts['pair_kl']
    native=hard.masked_pair
    o=native(logits[0],logits[1],y,accepted,0.)[0]
    m=native(logits[2][~physical],logits[3][~physical],y[~physical],accepted[~physical],0.)[0]
    a=native(logits[2][physical],logits[3][physical],y[physical],accepted[physical],0.)[0]
    r=native(logits[4],logits[5],y,accepted,0.)[0]
    no_kl=(.5*o+.25*m+.25*a+.25*r)/1.25
    assert torch.allclose(3*(changed-no_kl),reference,atol=1e-6)
    assert any(g is not None and bool(g.abs().sum()>0) for g in
               torch.autograd.grad(reference,logits,allow_unused=True))


def test_update_observer_never_changes_live_parameters():
    from experiments.pamf_repl_priority.observer import ReferenceObserver
    from experiments.pamf_ssl_research import optimizer_step_guard as native
    p=torch.nn.Parameter(torch.tensor([2.,3.])); obs=ReferenceObserver(None); optimizer=object()
    obs.pending=(None,[('net.t2.weight',p)],None);obs.reference=(torch.tensor([1.,3.]),)
    obs.before=(torch.tensor([2.,2.]),);obs.optimizer=optimizer;obs.source_step=100
    obs.hook=SimpleNamespace(remove=lambda:None)
    token=native._SCOPE.set({'pending':obs})
    try:obs.after_step(optimizer,(),{})
    finally:native._SCOPE.reset(token)
    assert torch.equal(p.detach(),torch.tensor([2.,3.]))
    assert obs.calls==1 and obs.last_details['t2']['cosine']==pytest.approx(2**-.5)


def test_observer_hook_is_removed_between_sampled_optimizer_steps():
    from experiments.pamf_repl_priority.observer import ReferenceObserver
    from experiments.pamf_ssl_research import optimizer_step_guard as native
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    opt=torch.optim.SGD([p],lr=.1);obs=ReferenceObserver(None);obs.optimizer=opt
    scope={'pending':None};token=native._SCOPE.set(scope)
    try:
        for source_step in (100,200):
            obs.pending=(None,[('net.t2.weight',p)],None)
            obs.before=(p.detach().clone(),);obs.reference=(p.detach().clone()-1.,)
            obs.source_step=source_step;scope['pending']=obs
            obs.hook=opt.register_step_post_hook(obs.after_step)
            p.grad=torch.ones_like(p);opt.step()
            assert obs.hook is None and scope['pending'] is None
            # The following real optimizer step is unsampled and must not call after_step.
            p.grad=torch.ones_like(p);opt.step()
        assert obs.calls==2
    finally:
        obs.close();native._SCOPE.reset(token)
