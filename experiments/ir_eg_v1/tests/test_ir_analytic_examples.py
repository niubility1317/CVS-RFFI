"""Independent finite-step examples; none establish global game convergence."""
import os
import pytest
import torch
from cvsrffi.xuc_fusion.ir_types import ResponseMetric
from cvsrffi.xuc_fusion.ir_cg import solve_response,cg_quadratic_energy

@pytest.fixture(autouse=True)
def _acceptance_device():
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('invalid IR_ACCEPTANCE_DEVICE')
    with torch.device(device):yield

def metric(diagonal):
    return ResponseMetric(diagonal,diagonal.sqrt(),torch.arange(diagonal.numel()),1.)

def linear_game_step(q,A,eta,kappa):
    """Analytic SGD-like scalar metric example, separate from native AdamW."""
    predicted=q-eta*(A@q)
    e=(A@predicted-A@q)[1:]
    result=solve_response(e,metric(q.new_tensor([eta])),lambda v:A[1,1]*v,rtol=1e-10)
    response=predicted.clone();response[1:]+=kappa*eta**.5*result.solution
    return q-eta*(A@response)

def test_pure_rotation_EG_and_IR_closed_update_matrices():
    eta=.2;kappa=.25;A=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
    basis=torch.eye(2,dtype=A.dtype)
    numerical_eg=torch.stack([linear_game_step(basis[:,i],A,eta,0.) for i in range(2)],dim=1)
    numerical_ir=torch.stack([linear_game_step(basis[:,i],A,eta,kappa) for i in range(2)],dim=1)
    expected_eg=torch.tensor([[1-eta**2,eta],[-eta,1-eta**2]],dtype=A.dtype)
    expected_ir=torch.tensor([[1-eta**2,eta-kappa*eta**3],[-eta,1-eta**2]],dtype=A.dtype)
    torch.testing.assert_close(numerical_eg,expected_eg,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(numerical_ir,expected_ir,atol=1e-12,rtol=1e-10)
    # Independent polynomial expression for the two-stage EG update.
    torch.testing.assert_close(expected_eg,basis-eta*A+eta**2*A@A)
    assert not torch.equal(numerical_eg,numerical_ir)

def test_strong_task_curvature_response_is_not_monotonically_safer():
    # T(theta)=10 theta^2; F=(20 theta-phi, theta+phi).
    # A finite response is not a trust region or a task-risk guarantee.
    eta=.1;kappa=.25;q=torch.tensor([1.,0.],dtype=torch.float64)
    A=torch.tensor([[20.,-1.],[1.,1.]],dtype=q.dtype)
    eg=linear_game_step(q,A,eta,0.);ir=linear_game_step(q,A,eta,kappa)
    torch.testing.assert_close(eg,q.new_tensor([2.99,.11]),atol=1e-12,rtol=1e-10)
    # hp-h0=-2.1, delta_phi=0.21/1.1, accepted theta changes by eta*kappa*delta.
    expected_theta=2.99+eta*kappa*(.21/1.1)
    torch.testing.assert_close(ir[0],q.new_tensor(expected_theta),atol=1e-12,rtol=1e-10)
    assert 10*ir[0].square()>10*eg[0].square()
    assert ir.square().sum()>eg.square().sum()

def test_quadratic_G_is_exact_Hessian_and_kappa_one_fixed_point():
    B=torch.tensor([[2.,.3],[.1,1.]],dtype=torch.float64)
    Q=B.T@B;linear=torch.tensor([.2,-.4],dtype=Q.dtype)
    phip=torch.tensor([.4,-.2],dtype=Q.dtype);h0=torch.tensor([.1,.3],dtype=Q.dtype)
    loss=lambda p:.5*(B@p).square().sum()+linear@p
    H=torch.func.hessian(loss)(phip)
    torch.testing.assert_close(H,Q,atol=1e-12,rtol=1e-10)
    hp=torch.func.grad(loss)(phip);e=hp-h0
    R=phip.new_tensor([.2,.5]);m=metric(R)
    result=solve_response(e,m,lambda v:Q@v,rtol=1e-10)
    delta=m.sqrt_diagonal*result.solution
    exact=torch.linalg.solve(torch.diag(1/R)+Q,-e)
    torch.testing.assert_close(delta,exact,atol=1e-12,rtol=1e-10)
    assert result.status=='early_converged' and result.iterations<=2
    phir=phip+delta # kappa=1
    torch.testing.assert_close(phir-phip+R*(torch.func.grad(loss)(phir)-h0),torch.zeros_like(phir),atol=1e-12,rtol=0.)

def test_asymmetric_U_head_supervision_indirectly_changes_theta_response():
    # U uses a detached feature with asymmetric scaling 3*theta.
    # Both comparisons keep the same predictor anchor to isolate head response.
    alphaL=.7;gammaL=.4;R=.2;kappa=.25
    theta0=torch.tensor(.5,dtype=torch.float64);phi0=torch.tensor(.1,dtype=torch.float64)
    thetaP=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    phiP=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    du=.5*(phiP-3*thetaP.detach()).square()
    gtheta,gphi=torch.autograd.grad(du,(thetaP,phiP),allow_unused=True)
    assert gtheta is None and gphi.abs()>0
    def theta_field(alphaU):
        h0=alphaL*(phi0-theta0)+alphaU*(phi0-3*theta0)
        hp=alphaL*(phiP.detach()-thetaP.detach())+alphaU*(phiP.detach()-3*thetaP.detach())
        G=alphaL+alphaU
        result=solve_response((hp-h0).reshape(1),metric(phiP.new_tensor([R])),lambda v:G*v,rtol=1e-10)
        phir=phiP.detach()+kappa*R**.5*result.solution[0]
        # -alphaL*gammaL*d_theta D_L. No U term is added here.
        return alphaL*gammaL*(phir-thetaP.detach())
    with_U=theta_field(.9);without_U=theta_field(0.)
    assert not torch.isclose(with_U,without_U)
    expected_delta_no_U=-R*(-.14)/(1+R*.7)
    expected_delta_U=-R*(-.14-.72)/(1+R*1.6)
    torch.testing.assert_close(with_U-without_U,phiP.new_tensor(alphaL*gammaL*kappa*(expected_delta_U-expected_delta_no_U)),atol=1e-12,rtol=1e-10)

def test_cg_quadratic_energy_matches_explicit_active_system_and_preserves_state():
    e=torch.tensor([.3,-.1,.5],dtype=torch.float64)
    Q=torch.tensor([[2.,.1,.2],[.1,1.,.1],[.2,.1,3.]],dtype=e.dtype)
    diagonal=e.new_tensor([.2,.4]);m=ResponseMetric(diagonal,diagonal.sqrt(),torch.tensor([0,2]),1.)
    result=solve_response(e,m,lambda v:Q@v,rtol=1e-10)
    u=result.solution;S=torch.diag(m.sqrt_diagonal)
    A=torch.eye(2,dtype=e.dtype)+S@Q[m.active_indices][:,m.active_indices]@S
    b=-m.sqrt_diagonal*e[m.active_indices]
    expected=.5*u@A@u-b@u
    before=(e.clone(),Q.clone(),u.clone(),m.diagonal.clone(),torch.get_rng_state())
    energy=cg_quadratic_energy(e,m,lambda v:Q@v,u)
    assert energy==pytest.approx(float(expected),abs=1e-12,rel=1e-10)
    assert energy<0 and cg_quadratic_energy(e,m,lambda v:Q@v,torch.zeros_like(u))==0.
    assert all(torch.equal(a,b) for a,b in zip(before,(e,Q,u,m.diagonal,torch.get_rng_state())))

def test_cg_quadratic_energy_empty_and_nonfinite_boundaries():
    e=torch.ones(2,dtype=torch.float64);empty=e.new_empty(0)
    m=ResponseMetric(empty,empty,torch.empty(0,dtype=torch.long),1.)
    assert cg_quadratic_energy(e,m,lambda v:(_ for _ in ()).throw(AssertionError()),empty)==0.
    m=metric(torch.ones(2,dtype=torch.float64))
    with pytest.raises(ValueError,match='nonfinite'):
        cg_quadratic_energy(e,m,lambda v:v*float('nan'),e)
