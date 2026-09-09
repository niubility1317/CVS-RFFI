"""L_s-only branch supervision for the existing Fisher gate."""
import torch
import torch.nn.functional as F


def fisher_labeled_objective(model, auxiliary, labels, count, epoch):
    bank = model.id_backbone.nmfdu_gate
    logits = auxiliary['nmfdu_branch_logits']
    y = labels[:count]
    names = bank.branch_names
    branch_loss = torch.stack([F.cross_entropy(logits[name][:count].float(), y) for name in names]).mean()
    # Label information updates source training evidence only, never inference.
    bank.evidence_state.update_discriminability(
        {name: auxiliary['nmfdu_branch_embeddings'][name][:count] for name in names}, y)
    gate = auxiliary['physical_gate_diag']['per_sample']
    scores = torch.stack([logits[name][:count].float() for name in names], dim=1)
    true_scores = scores.gather(2, y[:, None, None].expand(-1, len(names), 1)).squeeze(-1)
    negative = scores.masked_fill(F.one_hot(y, scores.size(-1))[:, None].bool(), -torch.inf).amax(-1)
    oracle = ((true_scores - negative).detach() / 2.).softmax(1)
    weights = gate['weights'][:count].float()
    conditional = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    route = F.kl_div(conditional.clamp_min(1e-8).log(), oracle, reduction='batchmean')
    ramp = min(1., max(0., (epoch - 20) / 20.))
    total = .1 * branch_loss + .02 * ramp * route
    return total, {'train/fisher_branch_ce': branch_loss.detach(),
                   'train/fisher_route_kl': route.detach(),
                   'train/fisher_loss_weighted': total.detach(),
                   'train/fisher_source_evidence_updates': float(bank.evidence_state.update_count),
                   'train/fisher_null_mean': gate['null_weight'].detach().mean()}
