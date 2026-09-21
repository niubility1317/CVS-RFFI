"""Low-frequency parameter-gradient evidence; never calls backward or steps."""
import math
import torch


def audit_decision_parameter_gradients(logits, labels, reference, valid_mask,
        named_identity_parameters, *, named_classifier_parameters=(), compared_losses=None,
        delta=0.0, pair_weights=None, decision_weight=1.0):
    """Audit weighted TX/pair contributions with the original global denominator.

    compared_losses maps e.g. identity_supervision/response/base to graph-bearing
    scalar losses. A common parameter is autograd-reachable (gradient not None)
    from decision and every supplied loss, even when a particular gradient is
    numerically zero. Call before backward, unscaled by GradScaler. Classifier
    parameters are reported separately and never admitted to common identity.
    Parameter .grad and optimizer state are not changed; graph is retained.
    """
    identity, classifier = list(named_identity_parameters), list(named_classifier_parameters)
    all_parameters = identity + classifier
    if len({name for name,_ in all_parameters}) != len(all_parameters):
        raise ValueError('gradient audit parameter names must be unique')
    if len({id(p) for _,p in all_parameters}) != len(all_parameters):
        raise ValueError('identity and classifier parameter sets must be disjoint')
    if not all(p.requires_grad for _,p in all_parameters):
        raise ValueError('gradient audit accepts trainable parameters only')
    if not math.isfinite(float(decision_weight)) or decision_weight < 0:
        raise ValueError('decision weight must be finite and nonnegative')
    labels=torch.as_tensor(labels,dtype=torch.long,device=logits.device)
    reference=torch.as_tensor(reference,device=logits.device).detach()
    valid=torch.as_tensor(valid_mask,dtype=torch.bool,device=logits.device).clone()
    if reference.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError('reference and valid mask must match logits')
    margins=logits.gather(1,labels[:,None])-logits
    noise=torch.broadcast_to(torch.as_tensor(delta,device=logits.device,dtype=logits.dtype).detach(),logits.shape)
    weights=torch.ones_like(logits) if pair_weights is None else torch.broadcast_to(
        torch.as_tensor(pair_weights,device=logits.device,dtype=logits.dtype).detach(),logits.shape)
    if (noise<0).any() or torch.isinf(noise).any() or not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError('invalid detached noise or weights')
    valid &= torch.isfinite(noise) & (weights>0)
    valid.scatter_(1,labels[:,None],False)
    denominator=max(int(valid.sum()),1)
    gap=torch.relu(reference-torch.nan_to_num(noise)-margins)
    terms=gap.square()*weights*valid*float(decision_weight)/denominator
    params=[p for _,p in all_parameters]
    def gradients(loss):
        if not params or not loss.requires_grad:
            return tuple(None for _ in params)
        return torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    total_loss=terms.sum()
    losses=dict(compared_losses or {})
    if 'decision' in losses:
        raise ValueError('decision is a reserved compared loss name')
    loss_grads={key:gradients(loss) for key,loss in dict(losses,decision=total_loss).items()}
    common=[i for i in range(len(identity)) if all(gs[i] is not None for gs in loss_grads.values())]
    classifier_indices=list(range(len(identity),len(all_parameters)))
    def summary(gs,indices):
        rows={}
        norm_sq=0.0
        for i in indices:
            grad=gs[i]
            norm=None if grad is None else float(torch.linalg.vector_norm(grad.detach().float()))
            rows[all_parameters[i][0]]=dict(is_none=grad is None,is_zero=grad is not None and norm==0,
                finite=grad is None or bool(torch.isfinite(grad).all()),l2=norm)
            if norm is not None:
                norm_sq+=norm**2
        return dict(l2=math.sqrt(norm_sq),parameters=rows,
            none_mask={k:r['is_none'] for k,r in rows.items()},
            zero_mask={k:r['is_zero'] for k,r in rows.items()})
    def report(loss,gs=None):
        gs=gradients(loss) if gs is None else gs
        return dict(weighted_loss=float(loss.detach()),identity_common=summary(gs,common),
            identity_all=summary(gs,range(len(identity))),classifier_only=summary(gs,classifier_indices))
    by_tx,by_pair={},{}
    for target in sorted(set(labels.detach().cpu().tolist())):
        rows=labels==target
        by_tx[target]=report(terms[rows].sum())
        for competitor in range(logits.shape[1]):
            if competitor != target:
                by_pair[(target,competitor)]=report(terms[rows,competitor].sum())
    return dict(common_identity_parameters=[all_parameters[i][0] for i in common],
        common_across_losses=list(loss_grads),compared_loss_reachability={key:summary(gs,range(len(all_parameters))) for key,gs in loss_grads.items()},
        total=report(total_loss,loss_grads['decision']),by_tx=by_tx,by_pair=by_pair,
        valid_comparisons=int(valid.sum()),decision_weight=float(decision_weight),
        denominator='global valid decision comparisons',gradient_scale='unscaled weighted parameter gradients',
        norm_aggregation='each L2 is a vector norm; per-pair norms are not additive',
        common_set_scope='decision plus caller-supplied compared losses')
