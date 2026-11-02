"""Match aligned IQ and frozen-reader predictions across paired source views."""
import torch
from torch.nn import functional as F

from experiments.phase1_volterra_aligned.objective import source_loss as aligned_source_loss


def source_loss(reader, output, clean, clean_features, groups):
    if len(output) % 3:
        raise ValueError('Three matched source views per record are required')
    count = len(output)//3
    if (not torch.equal(clean[:count], clean[count:2*count])
            or not torch.equal(clean[:count], clean[2*count:])
            or not torch.equal(groups[:count], groups[count:2*count])
            or not torch.equal(groups[:count], groups[2*count:])):
        raise ValueError('Views must refer to the same ordered source records')
    base, terms = aligned_source_loss(reader, output, clean, clean_features, groups)
    features = reader.features(output)
    logits = reader.classify_features(features)
    if logits.shape != (len(output), 6) or not torch.isfinite(logits).all() or not torch.isfinite(features).all():
        raise FloatingPointError('Finite frozen-reader view predictions are required')
    unit = F.normalize(features, dim=1, eps=1e-8).reshape(3, count, -1)
    center = F.normalize(unit.mean(0).detach(), dim=1, eps=1e-8)
    feature_consistency = (1-(unit*center[None]).sum(-1)).mean(0)
    log_prob = F.log_softmax(logits.reshape(3, count, 6)/2., dim=-1)
    mean_prob = log_prob.exp().mean(0).detach()
    distribution_consistency = 4.*F.kl_div(log_prob, mean_prob[None].expand_as(log_prob),
                                            reduction='none').sum(-1).mean(0)
    per = 0.1*distribution_consistency+0.05*feature_consistency
    paired_groups = groups[:count]
    paired = torch.stack([per[paired_groups == group].mean() for group in paired_groups.unique()]).mean()
    terms.update(view_distribution=distribution_consistency, view_feature=feature_consistency)
    return base+paired, terms
