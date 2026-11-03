"""Source-only pairwise reader-geometry consistency without identity labels."""
import torch
from torch.nn import functional as F
from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def relational_geometry_loss(output_features, clean_features, groups, temperature=.05):
    """Match each restored feature's relation to the source clean-feature batch.

    The target geometry and its source references are detached. At inference no
    source batch or clean reference is used; only the trained restorer remains.
    """
    n = len(output_features)
    if (n < 1 or output_features.ndim != 2 or clean_features.shape != output_features.shape
            or output_features.shape[1] < 1 or groups.shape != (n,) or temperature <= 0):
        raise ValueError('Matching feature batches, receiver groups, and positive temperature required')
    if not torch.isfinite(output_features).all() or not torch.isfinite(clean_features).all():
        raise FloatingPointError('Finite source feature geometry required')
    with torch.no_grad():
        references = F.normalize(clean_features.detach(), dim=1, eps=1e-8)
        teacher_log_probs = F.log_softmax((references @ references.T) / temperature, dim=1)
        teacher_probs = teacher_log_probs.exp()
    restored = F.normalize(output_features, dim=1, eps=1e-8)
    predicted_log_probs = F.log_softmax((restored @ references.T) / temperature, dim=1)
    per_record = (teacher_probs * (teacher_log_probs-predicted_log_probs)).sum(dim=1)
    return torch.stack([per_record[groups == group].mean() for group in groups.unique()]).mean()


def geometry_protected_objective(reader, output, anchor, clean, clean_features, groups):
    protected, terms = protected_source_objective(reader, output, anchor, clean, clean_features, groups)
    restored_features = reader.features(output)
    geometry = relational_geometry_loss(restored_features, clean_features, groups)
    loss = protected + .05*geometry
    if not torch.isfinite(loss):
        raise FloatingPointError('Finite geometry-protected source loss required')
    return loss, dict(terms, relational_geometry=geometry)
