import torch

from experiments.pamf_unrolled_restore.wisig_observed_solve import Action, dot, objective, solve
from experiments.pamf_unrolled_restore.wisig_observed_solve_source import build, proposals


def test_real_adjoint_and_observation_solver():
    rng = torch.Generator().manual_seed(17)
    def random():
        return torch.randn((3,31),dtype=torch.complex64,generator=rng)
    x,y = random(),random()
    main,image = 1.+.2*random(),.1*random()
    operator = Action(main,image,.28*random(),.21*random())
    torch.testing.assert_close(dot(operator.apply(x),y),dot(x,operator.adjoint(y)),atol=2e-6,rtol=1e-6)
    weight = torch.rand((3,31),generator=rng)
    penalty = .3

    def risk(result):
        return (weight*(operator.apply(result)-y).abs().square()+penalty*(result-x).abs().square()).sum(-1)

    diagonal = solve(operator,y,x,weight,penalty,coupled=False)
    coupled = solve(operator,y,x,weight,penalty,coupled=True)
    assert torch.all(risk(coupled)<risk(diagonal))
    separable = Action(main,image,torch.zeros_like(main),torch.zeros_like(main))
    torch.testing.assert_close(solve(separable,y,x,weight,penalty,coupled=False),
                               solve(separable,y,x,weight,penalty,coupled=True),atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(solve(operator,y,x,torch.zeros_like(weight),penalty,coupled=True),x)


def test_both_arms_receive_gradients_and_gating_keeps_exact_observation():
    rng = torch.Generator().manual_seed(23)
    orig = torch.randn((5,256),dtype=torch.complex64,generator=rng)
    for name in ('diagonal','coupled'):
        model = build(name)
        loss,components = objective(model,orig,rng,identity=True)
        loss.backward()
        assert torch.isfinite(components).all()
        for module in (model.timing,model.estimator,model.coarse,model.refine):
            assert torch.isfinite(module[-1].weight.grad).all() and module[-1].weight.grad.norm()>0
        model.eval()
        with torch.no_grad():
            predictions,active,_ = proposals(model,(float('inf'),float('inf')),orig)
            assert not active.any() and torch.equal(predictions[5],orig)
            if name=='coupled':
                assert all(torch.equal(readout,orig) for readout in predictions[-3:])
