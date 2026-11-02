"""Source-paired IQ, frozen-reader distribution, and pseudo-margin consistency."""
import torch
from torch.nn import functional as F


def source_loss(reader, output, clean, clean_features, groups):
    n = len(output)
    if (reader.training or any(p.requires_grad for p in reader.parameters())
            or clean.shape != output.shape or groups.shape != (n,) or clean_features.shape[0] != n):
        raise ValueError('Frozen reader and aligned paired-source examples are required')
    clean = clean.detach()
    features = reader.features(output)
    predicted = reader.classify_features(features)
    with torch.no_grad(): teacher = reader.classify_features(clean_features.detach())
    if predicted.shape != (n, 6) or teacher.shape != (n, 6) or not all(torch.isfinite(x).all() for x in (features, predicted, teacher)):
        raise FloatingPointError('Finite six-class frozen-reader outputs are required')
    energy = clean.flatten(1).square().sum(1).clamp_min(1e-8)
    wave = (output-clean).flatten(1).square().sum(1)/energy
    feature = 1-F.cosine_similarity(features, clean_features.detach(), dim=1, eps=1e-8)
    target = F.softmax(teacher/2., dim=1)
    distill = 4.*F.kl_div(F.log_softmax(predicted/2., dim=1), target, reduction='none').sum(1)
    top_values, top_indices = teacher.topk(2, dim=1)
    pseudo_class = top_indices[:, :1]
    expected_margin = (top_values[:, 0]-top_values[:, 1]).clamp(0, 4)
    chosen = predicted.gather(1, pseudo_class).squeeze(1)
    competitors = predicted.masked_fill(F.one_hot(pseudo_class.squeeze(1), 6).bool(), -1e9).max(1).values
    confidence = F.softmax(teacher, dim=1).max(1).values
    margin = confidence*F.relu(expected_margin-(chosen-competitors))
    per = wave+0.2*feature+0.2*distill+0.05*margin
    loss = torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
    return loss, dict(wave=wave, feature=feature, distill=distill, margin=margin)
