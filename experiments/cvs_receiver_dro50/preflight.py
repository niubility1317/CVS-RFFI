import json
import torch
from . import design as d
from .objective import group_risk
from .checks import checks
from experiments.cvs_source_signature_order.probe import model


def main():
    torch.set_num_threads(2)
    local = checks(); m = model().eval().requires_grad_(True)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    flags = [p.requires_grad for p in m.parameters()]
    result = m.forward_identity_only(torch.zeros(10, 2, 256))
    logits = result['tx_logits']; labels = torch.arange(10) % 6
    receivers = torch.tensor([r for r in d.RX for _ in range(2)])
    params = [dict(m.named_parameters())[n] for n in d.PARAMETERS]
    W = m.id_backbone.encoder.core.id_backbone.cls_head.weight
    for mode in ('ERM', 'BALANCED', 'DRO'):
        loss, state, details = group_risk(logits, labels, receivers, torch.zeros(5, dtype=torch.float64), mode)
        gradients = torch.autograd.grad(loss, [*params, W], retain_graph=True, allow_unused=False)
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Real-checkpoint receiver-risk derivative smoke failed')
    if flags != [p.requires_grad for p in m.parameters()] or any(p.grad is not None for p in m.parameters()):
        raise ValueError('No-query smoke changed flags or accumulated gradients')
    if any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items()):
        raise ValueError('No-query smoke changed model state')
    path = d.ROOT / 'preflight.json'
    if path.exists():
        raise FileExistsError('Immutable preflight exists')
    value = dict(status='PASS', checks=local, real_checkpoint_no_query_smoke=True,
        synthetic_IQ_packets=10, all_three_risk_gradients_finite=True,
        source_checkpoint='Original fixed source SIG_TIME final50; existing own_model source provenance checked',
        training_inherits_checkpoint=False, query_read=False, head_fit_updates=0, optimizer_updates=0)
    d.write(path, value); print(json.dumps(value))


if __name__ == '__main__':
    main()
