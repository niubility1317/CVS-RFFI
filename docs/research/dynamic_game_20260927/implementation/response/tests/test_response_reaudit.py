"""Independent adversarial examples for the second acceptance pass."""
import sys
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy
import pytest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.response_dric import constrained_game
from cvsrffi.xuc_fusion.response_fields import cross_tx_fields
from cvsrffi.xuc_fusion.response_context import passive
from cvsrffi.xuc_fusion.response_config import resolve
from cvsrffi.xuc_fusion.response_solver import ResponseSolver
from cvsrffi.xuc_fusion.response_replay import LocalReplay

@pytest.mark.parametrize('radius',[.1,.01,1e-5,0.])
def test_ball_actually_active_scalar_solution(radius):
    z,info=constrained_game(torch.eye(2),torch.tensor([-1.,.5]),1,
        torch.empty(0,1),torch.empty(0),radius)
    torch.testing.assert_close(z,torch.tensor([radius,-.5],dtype=torch.double),rtol=1e-5,atol=1e-10)
    assert info['residual']<=1e-5 and info['iterations']<=5

def test_replay_detached_forward_dependency():
    class DetachedPath(nn.Module):
        def __init__(self):
            super().__init__();self.tail=nn.Linear(2,2,bias=False);self.fixed=nn.Linear(2,2,bias=False)
        def forward(self,x):return self.fixed(self.tail(x).detach())
    torch.manual_seed(4);model=DetachedPath();x=torch.randn(4,2);replay=LocalReplay(model,model.tail.parameters())
    with replay.session('same'):model(x)
    with torch.no_grad():model.tail.weight.add_(2.)
    expected=model(x)
    with replay.session('same'):actual=model(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

def test_xt_zero_inner_step_reports_zero_changes_with_dropout():
    torch.manual_seed(17);head=nn.Sequential(nn.Linear(4,8),nn.Dropout(.6),nn.Linear(8,15)).train()
    z=torch.randn(90,4,requires_grad=True);y=torch.arange(6).repeat_interleave(15);d=torch.arange(15).repeat(6)
    _,diag=cross_tx_fields(head,z,y,d,0,0.,.1,.35)
    for direction in ('A_to_B','B_to_A'):
        assert diag[direction]['fit_change']==0.
        assert diag[direction]['monitor_change']==0.

def test_cf_risk_uses_final_committed_buffers():
    torch.manual_seed(6)
    model=nn.Sequential(nn.BatchNorm1d(2,momentum=1.),nn.Linear(2,1)).train()
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.randn(8,2)+4;monitor=torch.randn(7,2)
    solver=ResponseSolver(model,opt,resolve(dict(fixed_beta=0.)))
    result=solver.cf_step(lambda on:model(x).square().mean()*(2 if on else 1),lambda:model(monitor).mean(0))
    with passive(model),torch.no_grad():actual=model(monitor).mean(0)
    torch.testing.assert_close(torch.tensor(result.telemetry['risk_reference']),actual,rtol=0,atol=0)

def test_replay_never_suppresses_buffer_updates():
    class Counter(nn.Module):
        def __init__(self):super().__init__();self.register_buffer('count',torch.zeros(()))
        def forward(self,x):self.count.add_(1);return x+self.count
    model=nn.Sequential(nn.Linear(2,2),Counter());replay=LocalReplay(model,[]);x=torch.randn(3,2)
    with replay.session('same'):model(x)
    with replay.session('same'):actual=model(x)
    assert model[1].count==2
    torch.testing.assert_close(actual,model[0](x)+2)

@pytest.mark.parametrize('n',[1,2,4])
@pytest.mark.parametrize('radius',[.01,.3])
@pytest.mark.parametrize('active_halfspace',[False,True])
def test_coupled_ball_solution_against_manufactured_kkt(n,radius,active_halfspace):
    gen=torch.Generator().manual_seed(100+n);size=n+2
    raw=torch.randn(size,size,generator=gen,dtype=torch.double)
    J=raw@raw.T+torch.eye(size,dtype=torch.double)
    skew=torch.randn(size,size,generator=gen,dtype=torch.double);J=J+.2*(skew-skew.T)
    theta=torch.ones(n,dtype=torch.double)*radius/n**.5
    z=torch.cat((theta,torch.tensor([.1,-.2],dtype=torch.double)))
    A=torch.eye(n,dtype=torch.double)[:1] if active_halfspace else torch.empty(0,n,dtype=torch.double)
    eps=A@theta;normal=torch.zeros(size,dtype=torch.double)
    if active_halfspace:normal[:n]=A[0]*.7
    normal[:n]+=11.*theta
    solved,info=constrained_game(J,-J@z-normal,n,A,eps,radius)
    torch.testing.assert_close(solved,z,rtol=2e-5,atol=1e-8)
    assert info['residual']<=1e-5

def test_xt_diagnostics_do_not_change_training_loss_gradients_or_rng():
    torch.manual_seed(19);base=nn.Sequential(nn.Linear(4,8),nn.Dropout(.6),nn.Linear(8,15)).train()
    z0=torch.randn(90,4);y=torch.arange(6).repeat_interleave(15);d=torch.arange(15).repeat(6)
    records=[]
    for enabled in (False,True):
        head=deepcopy(base);z=z0.clone().requires_grad_();torch.manual_seed(51)
        loss,_=cross_tx_fields(head,z,y,d,0,.01,.1,.35,diagnostics=enabled)
        gs=torch.autograd.grad(loss,[z]+list(head.parameters()));records.append((loss.detach(),gs,torch.get_rng_state()))
    torch.testing.assert_close(records[0][0],records[1][0],rtol=0,atol=0)
    for a,b in zip(records[0][1],records[1][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(records[0][2],records[1][2])

def test_replay_exception_restores_original_method_bindings():
    model=nn.Sequential(nn.Linear(2,2));replay=LocalReplay(model,[])
    assert 'forward' not in model[0].__dict__
    with pytest.raises(RuntimeError):
        with replay.session('failure'):
            model(torch.randn(2,2));raise RuntimeError('abort')
    assert 'forward' not in model[0].__dict__

@pytest.mark.parametrize('change',[{'beta_candidates':[]},{'margin_tolerance':True},{'residual_tolerance':0.},
    {'min_monotonicity':0.},{'transport_gamma':float('nan')},{'fixed_beta':.5,'random_beta':True}])
def test_malformed_response_config_rejected_before_model(change):
    with pytest.raises(ValueError):resolve(change)

def test_stage_table_matches_global_clock_and_rotations_cover_all_slots():
    from cvsrffi.xuc_fusion.response_config import stage_table,schedule,cadence
    for method in ('DRIC','CF_EG','TR_EG','XT_DANN','EG'):
        c=resolve(dict(method=method))
        for row in stage_table(c):
            start=(row['epoch']-1)*222
            expected=[i+1 for i in range(222) if method!='EG' and
                schedule(c,row['epoch'],start+i)['active'] and schedule(c,row['epoch'],start+i)['strength']>0]
            assert row['correction_steps_in_epoch']==expected and row['correction_count']==len(expected)
        step=cadence(c)
        indices=[accepted_before//step%70 for accepted_before in range(step-1,step*70,step)]
        assert len(set(indices))==70

class TinyTransport(nn.Module):
    def __init__(self):
        super().__init__();self.enc=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4))
        self.adv_head=nn.Sequential(nn.Linear(4,6),nn.Dropout(.3),nn.Tanh(),nn.Linear(6,2))
    def forward(self,x,**kw):
        z=self.enc(x);return {'tx_logits':z,'adv_dom_logits':self.adv_head(z),'z_id':z}

def test_tr_gamma_zero_exactly_matches_full_eg_including_buffers():
    from cvsrffi.game_tracking.solvers import GameSolver
    torch.manual_seed(39);model=TinyTransport();ref=deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.01);refopt=torch.optim.AdamW(ref.parameters(),lr=.01)
    x=torch.randn(8,3);monitor=torch.randn(6,3);domain=torch.arange(6)%2
    def loss(m):
        out=m(x);return out['tx_logits'].square().mean()+out['adv_dom_logits'].square().mean()
    torch.manual_seed(44)
    ResponseSolver(model,opt,resolve(dict(method='TR_EG',transport_gamma=0.)),max_grad_norm=5.).tr_step(lambda on:loss(model),(x,monitor,domain),1.)
    rng=torch.get_rng_state()
    torch.manual_seed(44);GameSolver(ref,refopt,mode='extragradient',max_grad_norm=5.).step(lambda:loss(ref))
    torch.testing.assert_close(model.state_dict(),ref.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(opt.state_dict(),refopt.state_dict(),rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())

def test_real_dric_cached_and_uncached_update_are_equal(monkeypatch):
    from test_response_games import test_real_native_response_path
    original=ResponseSolver.response_step;records=[]
    def capture(self,*a,**kw):
        result=original(self,*a,**kw)
        records.append((deepcopy(self.model.state_dict()),deepcopy(self.optimizer.state_dict()),torch.get_rng_state()))
        return result
    monkeypatch.setattr(ResponseSolver,'response_step',capture)
    test_real_native_response_path('DRIC_PARITY',40,3,monkeypatch)
    @contextmanager
    def uncached(self,tag):yield
    monkeypatch.setattr(LocalReplay,'session',uncached)
    test_real_native_response_path('DRIC_PARITY',40,3,monkeypatch)
    torch.testing.assert_close(records[0][0],records[1][0],rtol=0,atol=0)
    torch.testing.assert_close(records[0][1],records[1][1],rtol=0,atol=0)
    assert torch.equal(records[0][2],records[1][2])

def test_real_dric_failure_rolls_back_entire_transaction(monkeypatch):
    from test_response_games import test_real_native_response_path
    from cvsrffi.xuc_fusion import response_dric
    original=ResponseSolver.response_step;checked=[]
    def failure(*a,**kw):raise RuntimeError('forced local residual failure')
    monkeypatch.setattr(response_dric,'constrained_game',failure)
    def capture(self,*a,**kw):
        state=deepcopy(self.model.state_dict());opt=deepcopy(self.optimizer.state_dict());rng=torch.get_rng_state()
        with pytest.raises(RuntimeError,match='forced local residual failure'):original(self,*a,**kw)
        torch.testing.assert_close(self.model.state_dict(),state,rtol=0,atol=0)
        torch.testing.assert_close(self.optimizer.state_dict(),opt,rtol=0,atol=0)
        assert self.steps==0 and torch.equal(rng,torch.get_rng_state());checked.append(True)
        raise RuntimeError('checked rollback')
    monkeypatch.setattr(ResponseSolver,'response_step',capture)
    with pytest.raises(RuntimeError,match='checked rollback'):test_real_native_response_path('DRIC_FAILURE',40,3,monkeypatch)
    assert checked==[True]
