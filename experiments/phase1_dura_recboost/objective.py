"""Source-only confidence-weighted frozen-reader pseudo-class guidance."""
import torch
from torch.nn import functional as F
from experiments.phase1_volterra_aligned.objective import source_loss

def recognition_source_objective(reader,output,anchor,clean,clean_features,groups):
    base,terms=source_loss(reader,output,clean,clean_features,groups)
    with torch.no_grad():
        teacher=reader.classify_features(clean_features.detach())
        confidence,pseudo=F.softmax(teacher,dim=1).max(1)
    logits=reader.classify_features(reader.features(output))
    if logits.shape!=teacher.shape or not torch.isfinite(logits).all():
        raise FloatingPointError('Finite frozen-reader logits are required')
    per=confidence*F.cross_entropy(logits,pseudo,reduction='none')
    guidance=torch.stack([per[groups==g].mean() for g in groups.unique()]).mean()
    return base+0.2*guidance,dict(terms,pseudo_ce=per)

