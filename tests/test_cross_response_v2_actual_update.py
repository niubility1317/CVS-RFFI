"""The source audit must reproduce the actual full-loss optimizer transaction."""
import copy
import pytest
import torch
from torch import nn
from test_cross_response_v2_integration import Model
from cvsrffi.cross_response.training import parameter_roles, response_backward
from cvsrffi.cross_response.replay_audit import first_divergence, capture_rng_state


@pytest.mark.parametrize('max_grad_norm', [0., .05])
def test_full_graph_counterfactual_matches_real_adamw_and_keeps_live_state(max_grad_norm):
    from cvsrffi.cross_response.source_mechanism import paired_actual_optimizer_update
    torch.manual_seed(51)
    model = Model().double()
    auxiliary = nn.Linear(3, 2).double()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(auxiliary.parameters()), lr=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    x, y = torch.randn(12, 2, 8, dtype=torch.float64), torch.arange(12) % 4
    # Nonempty AdamW state matters; a fresh-SGD approximation would not match.
    warm = model(x)
    (warm['tx_logits'].square().mean() + auxiliary(warm['z_dom']).square().mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    initial = copy.deepcopy((model, auxiliary, optimizer, scaler))

    def losses(m, aux):
        out = m(x)
        ce = nn.functional.cross_entropy(out['tx_logits'], y)
        # Deliberately include non-CE objectives in the graph. The auditor must
        # consume this graph, not reconstruct a guessed CE+orth substitute.
        complete = ce + .7*out['z_dom'].square().mean() + .2*(out['z_id']*out['z_dom']).mean()
        response = (aux(out['z_id']) + aux(out['z_dom']) - 1.).square().mean()
        return ce, complete, response, out['z_id'].mean().square(), out['z_dom'].mean().square()

    ce, complete, response, decision, cross = losses(model, auxiliary)
    roles = parameter_roles(model, auxiliary, ['id_backbone.cls_head'])
    before = copy.deepcopy((model.state_dict(), auxiliary.state_dict(), optimizer.state_dict(), scaler.state_dict()))
    rng = capture_rng_state()
    results = paired_actual_optimizer_update(model, auxiliary, optimizer, scaler,
        baseline_loss=complete, identity_loss=ce, response_loss=response, decision_loss=decision,
        cross_loss=cross, roles=roles, lambda_resp=.1, lambda_dec=.03, lambda_cross=.02,
        gradient_cap=.1, max_grad_norm=max_grad_norm,
        evaluate=lambda m: dict(logits=m(x)['tx_logits'].detach()), capture_tensors=True)
    after = (model.state_dict(), auxiliary.state_dict(), optimizer.state_dict(), scaler.state_dict())
    assert first_divergence(before, after) is None
    assert first_divergence(rng, capture_rng_state()) is None
    assert all(p.grad is None for p in list(model.parameters())+list(auxiliary.parameters()))
    for key, weight in [('base', 0.), ('base_response', .1)]:
        m, aux, opt, sc = copy.deepcopy(initial)
        ce, complete, resp, dec, cr = losses(m, aux)
        response_backward(baseline_loss=complete, identity_loss=ce, response_loss=resp,
            decision_loss=dec, cross_loss=cr, roles=parameter_roles(m, aux, ['id_backbone.cls_head']),
            scaler=sc, lambda_resp=weight, lambda_dec=.03, lambda_cross=.02,
            joint_open=True, gradient_cap=.1)
        sc.unscale_(opt)
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(aux.parameters()), max_grad_norm)
        sc.step(opt)
        sc.update()
        assert first_divergence(m.state_dict(), results[key]['model_state']) is None
        assert first_divergence(aux.state_dict(), results[key]['auxiliary_state']) is None
        assert first_divergence(opt.state_dict(), results[key]['optimizer_state']) is None
    assert results['metadata']['baseline'] == 'actual_complete_training_loss_graph'
