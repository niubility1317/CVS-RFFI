"""Source-only actual AdamW auxiliary increment, with fixed default W readout."""
import math
import torch
from torch.nn import functional as F
from experiments.cvs_u_schedule50.counterfactual import simulate, _preserve_rng

SOURCE_RX = (1, 3, 4, 6, 8)


def add_gradients(base, auxiliary):
    if len(base) != len(auxiliary):
        raise ValueError('Gradient order/count differs')
    return tuple(a if b is None else b if a is None else a + b for a, b in zip(base, auxiliary))


def _validate_source(x, y, rx, anchor_rx, role):
    if role != 'L_s' or anchor_rx not in SOURCE_RX:
        raise ValueError('Only visible source L and fixed source anchor RX')
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (2, 256)
            or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite real source IQ[N,2,256] required')
    if (y.shape != (len(x),) or rx.shape != (len(x),) or y.device != x.device
            or rx.device != x.device or y.dtype != torch.long or rx.dtype != torch.long):
        raise ValueError('Visible source labels/RX must be matching long metadata')
    if not bool(((y >= 0) & (y < 6)).all()) or any(r not in SOURCE_RX for r in rx.tolist()):
        raise ValueError('Only original source labels and RX')
    if set(rx.tolist()) != set(SOURCE_RX):
        raise ValueError('Credit requires all five source RX in existing L batch')


def assess(model, optimizer, base_grads, auxiliary_grads, x, y, rx, anchor_rx,
           *, default_weight_name, role='L_s', tolerance=1e-6):
    """Compare base vs base+one-RX aux from exactly the same optimizer state.

    This is utility on other RX in the existing visible L batch. Main CE has
    already seen them; they are not unseen/held-out receivers or V. Actual
    saved moments/lr/weight decay and independently clip5 steps are consumed.
    Both detached temporary models use the same pre-step default classifier W,
    so its fitting cannot explain credit. It is an evaluation override only;
    the ordinary main head/backbone training remains jointly trainable.

    Accept mean credit>tolerance and each otherRX credit>=-tolerance. No query,
    V, pseudo-label, second-order meta-gradient or extra real update exists.
    Final-state diagnosis is hypothetical, not an original historical step.
    """
    _validate_source(x, y, rx, anchor_rx, role)
    if tolerance != 1e-6:
        raise ValueError('Fixed numerical tolerance; no scan')
    named = dict(model.named_parameters())
    if default_weight_name not in named or named[default_weight_name].shape != (6, 160):
        raise ValueError('Original default6x160 classifier weight required')
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    by_id = {id(p): name for name, p in named.items()}
    if any(id(p) not in by_id for p in parameters):
        raise ValueError('Optimizer contains parameters outside this exact source model')
    names = [by_id[id(p)] for p in parameters]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate optimizer parameter')
    def label(name):
        if name == default_weight_name:return 'default_W'
        if '.sequence_pool.' in name:return 'SIG_TIME_sequence_pool'
        if '.t1.' in name:return 'TIME_t1'
        if '.t2.' in name:return 'TIME_t2'
        if '.cls_head.' in name:return 'default_head_components'
        return 'other_model_parameters'
    proposal = simulate(optimizer, base_grads, add_gradients(base_grads, auxiliary_grads), names=[label(n) for n in names])
    fixed_w = named[default_weight_name].detach().clone()
    original = {name: p.detach().clone() for name, p in named.items()}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    flags = {module: module.training for module in model.modules()}
    other = rx != anchor_rx
    with _preserve_rng(), torch.no_grad():
        try:
            model.eval()
            errors = []
            for after in (proposal['base_after'], proposal['total_after']):
                state = dict(original)
                state.update(zip(names, after))
                state[default_weight_name] = fixed_w
                # Separate cloned buffers ensure no branch can affect the next.
                logits = torch.func.functional_call(
                    model, (state, {k: v.clone() for k, v in buffers.items()}), (x[other],), strict=True)
                if logits.shape != (int(other.sum()), 6) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Actual default-head source predictions differ')
                errors.append(F.cross_entropy(logits, y[other], reduction='none'))
        finally:
            for module, training in flags.items():
                module.training = training
    signed = errors[0] - errors[1]
    groups = {}
    for r in SOURCE_RX:
        if r == anchor_rx:
            continue
        mask = rx[other] == r
        groups[str(r)] = dict(count=int(mask.sum()),
                              base_CE=float(errors[0][mask].mean()),
                              auxiliary_CE=float(errors[1][mask].mean()),
                              credit=float(signed[mask].mean()))
    mean = float(signed.mean())
    accepted = mean > tolerance and all(g['credit'] >= -tolerance for g in groups.values())
    if (any(not torch.equal(original[k], p) for k, p in model.named_parameters())
            or any(not torch.equal(buffers[k], v) for k, v in model.named_buffers())):
        raise RuntimeError('Temporary credit evaluation modified live model')
    summary = dict(anchor_RX=anchor_rx, other_RX=groups, mean_credit=mean, accepted=accepted,
                   tolerance=tolerance, comparator='same pre-step defaultW; actualAdamW base vs base+RXaux',
                   seen_L_training_receivers=True, unseen_RX_evaluation=False,
                   optimizer_updates=0, head_fit_updates=0, V_gradient_updates=0,
                   auxiliary_loss_updates=0, target_access=False, U_truth_read=False,
                   temporary_AdamW_steps=2, extra_IQ_forwards=int(other.sum()) * 2,
                   live_parameters_buffers_unchanged=True, step=proposal['summary'])
    return dict(summary=summary, proposal=proposal)
