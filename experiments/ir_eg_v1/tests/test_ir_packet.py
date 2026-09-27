import torch
import pytest
import torch.nn.functional as F
from cvsrffi.xuc_fusion.ir_objective import build_objective_packet
from ir_fixtures import make_case,GRL,assert_nested


def test_replace_only_registered_adversarial_leaves_and_no_native_phi_gradient_leak():
    a=make_case();before=[p.detach().clone() for p in a.model.parameters()]
    with a.solver._capture(a.ctx):
        native=a.closure();packet=build_objective_packet(a.ctx)
        leaves=tuple((p.detach()+.001).clone().requires_grad_(True) for p in a.solver.phi)
        value=packet.assemble(leaves)
        actual=torch.autograd.grad(value,tuple(a.solver.parameters)+leaves,allow_unused=True)
        mapped={id(p):g for p,g in zip(a.solver.parameters,actual)}
        assert all(mapped[id(p)] is None for p in a.solver.phi)
        assert all(g is not None and torch.isfinite(g).all() for g in actual[-4:])
        packet.release()
    assert_nested(before,[p.detach() for p in a.model.parameters()])


def test_phi_leaf_gradient_mapping_debug_zero_increment():
    a=make_case()
    with a.solver._capture(a.ctx):
        native=a.closure();packet=build_objective_packet(a.ctx)
        leaves=tuple(p.detach().clone().requires_grad_(True) for p in a.solver.phi)
        value=packet.assemble(leaves)
        assert torch.equal(value,native)
        old=torch.autograd.grad(native,a.solver.parameters,retain_graph=True,allow_unused=True)
        new=torch.autograd.grad(value,a.solver.psi+leaves,allow_unused=True)
        mapping={id(p):g for p,g in zip(a.solver.psi,new[:len(a.solver.psi)])}
        mapping.update({id(p):g for p,g in zip(a.solver.phi,new[-4:])})
        assert_nested(old,[mapping[id(p)] for p in a.solver.parameters],1e-6,1e-5)


def test_U_GRL_zero_and_U_task_nonzero():
    a=make_case();z=a.model.encoder(a.ctx.u)
    adv=a.model.adv_head(GRL.apply(z,0.));task=a.model.tx_head(z).square().mean()
    domain=F.cross_entropy(adv,a.ctx.ud)
    g=torch.autograd.grad(domain,z,retain_graph=True)[0]
    task_g=torch.autograd.grad(task,z)[0]
    assert torch.equal(g,torch.zeros_like(g)) and task_g.norm()>0


def test_scale_call_sequence_preserved():
    from ir_fixtures import run
    a=make_case();run(a)
    assert len(a.ctx.field_components)==2
    assert a.ctx.pending['used']==1.
    assert a.ctx.field_components[0]==a.ctx.field_components[1]


def test_unregistered_head_dependency_aborts():
    from ir_fixtures import run
    a=make_case();original=a.closure
    def invalid():
        loss=original();extra=.01*sum(p.square().sum() for p in a.solver.phi)
        if getattr(a.ctx,'ir_tape',None) is not None:
            assembly=a.ctx.ir_assembly
            a.ctx.ir_assembly=lambda leaves:assembly(leaves)+extra
        return loss+extra
    a.closure=invalid
    with pytest.raises(ValueError,match='gradient leak'):run(a)
    assert a.solver.steps==0 and not a.opt.state and a.external.commits==0
