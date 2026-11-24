import torch

from experiments.pamf_lora_main.dura_identity_action_source import ARM as NATIVE
from experiments.pamf_lora_main.dura_prior_width_source import CALCULATE as NATIVE_LOSS
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.dura_stage_supervision_source import ARM, CALCULATE, RECIPES


def test_training_reveals_first_map_without_changing_either_inference_endpoint():
    torch.manual_seed(2315)
    recipe = RECIPES[ARM]
    model = recipe.model().eval()
    observed = 1 + .08 * torch.randn(1,6,256,dtype=torch.complex64)
    phase = torch.linspace(0,9,256).expand_as(observed.real)
    rotation = torch.polar(torch.ones_like(phase), phase)
    with torch.no_grad():
        coarse, final = model(observed, None, rotation)
        fit_coarse, first, fit_final = recipe.restore(model,observed,None,rotation)
    torch.testing.assert_close(fit_coarse,coarse,atol=0.,rtol=0.)
    torch.testing.assert_close(fit_final,final,atol=0.,rtol=0.)
    assert first.shape == observed.shape


def test_matching_source_pair_adds_the_registered_loss_and_direct_first_stage_gradient():
    torch.manual_seed(2253)
    model = RECIPES[ARM].model()
    tracker = PacketRestorer('consensus').eval()
    iq = torch.randn(2,6,2,256)
    rng = lambda: torch.Generator().manual_seed(2254)
    args = dict(with_identity=False,step=1)
    baseline, _ = NATIVE_LOSS(model,tracker,iq,rng(),recipe=RECIPES[NATIVE],**args)
    revised, terms = CALCULATE(model,tracker,iq,rng(),recipe=RECIPES[ARM],**args)
    auxiliary = revised - baseline
    expected = .5 * (.30*terms[6] + .35*terms[7])
    torch.testing.assert_close(auxiliary,expected,atol=1e-6,rtol=1e-5)
    gradient, = torch.autograd.grad(auxiliary,model.restorer.prior[0].out.weight)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
