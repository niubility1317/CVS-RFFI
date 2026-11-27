"""Clean-anchor directional constraints using frozen E104 and labeled source IQ only."""

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def clean_prototypes(joint, reader, frontend, source_iq, labels, device):
    embeddings=[]
    for start in range(0,len(source_iq),32):
        raw=source_iq[start:start+32]
        raw=joint.tensor(raw.view(np.float32),device).view(torch.complex64)
        clean,_=joint.normalize_aperture(frontend.aperture(raw))
        _,feature=reader(clean)
        embeddings.append(F.normalize(feature,dim=-1))
    features=torch.cat(embeddings)
    target=joint.tensor(labels,device).long()
    pooled=torch.stack([features[target==klass].mean(0) for klass in range(10)])
    return F.normalize(pooled,dim=-1)


def clean_geometry_risk(restored, clean, target, prototypes, *, relative):
    prediction=F.normalize(restored,dim=-1)
    anchor=F.normalize(clean.detach(),dim=-1)
    with torch.no_grad():
        alternatives=anchor@prototypes.T
        known=F.one_hot(target,num_classes=prototypes.shape[0]).bool()
        nearest=alternatives.masked_fill(known,-torch.inf).argmax(-1)
        rival=prototypes[nearest]
        margin=(1.-(anchor*rival).sum(-1)).clamp_min(1e-4)
    delta=prediction-anchor
    shortfall=-(delta*anchor).sum(-1)
    if relative:
        shortfall=shortfall+(delta*rival).sum(-1)
    return (shortfall.relu()/margin).log1p().mean()
