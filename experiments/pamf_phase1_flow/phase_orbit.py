"""Input-only phase orbit candidates and frozen-classifier selection."""
import math
import torch
from torch.nn import functional as F


def phase_candidates(iq, phases):
    """Rotate IQ by fixed ordered phases; zero phase must be first."""
    if (iq.ndim != 3 or iq.shape[1:] != (2,256) or not phases or phases[0] != 0.
            or any(not math.isfinite(p) for p in phases)
            or any(abs(a)>abs(b) for a,b in zip(phases,phases[1:]))):
        raise ValueError('Phase1 IQ and ordered zero-first phases required')
    phase=iq.new_tensor(phases)
    cs,sn=phase.cos()[None,:,None],phase.sin()[None,:,None]
    real,imag=iq[:,None,0,:],iq[:,None,1,:]
    return torch.stack((real*cs-imag*sn,real*sn+imag*cs),dim=2)


def select_phase(logits, *, rule):
    """Select an actual candidate by frozen logits; ties use the anchor."""
    if logits.ndim != 3 or logits.shape[1] < 1 or logits.shape[2] < 2 or not torch.isfinite(logits).all():
        raise ValueError('Finite per-candidate frozen-classifier scores required')
    pred=logits.argmax(2)
    if rule=='max_confidence':
        confidence=F.softmax(logits,dim=2).max(2).values
        highest=confidence.max(1).values
        winner=confidence.argmax(1)
        return torch.where((confidence==highest[:,None]).sum(1)>1,torch.zeros_like(winner),winner)
    if rule=='soft_mean':
        average=F.softmax(logits,dim=2).mean(1)
        highest=average.max(1).values
        chosen=average.argmax(1)
        ties=(average==highest[:,None]).sum(1)>1
    elif rule=='majority':
        counts=F.one_hot(pred,num_classes=logits.shape[2]).sum(1)
        best=counts.max(1).values
        chosen=counts.argmax(1)
        ties=(counts==best[:,None]).sum(1)>1
        chosen=torch.where(ties,pred[:,0],chosen)
    else:
        raise ValueError('Registered phase selection rule required')
    matches=pred==chosen[:,None]
    # Phases are ordered by absolute magnitude, so argmax returns the first.
    first=matches.to(torch.int64).argmax(1)
    if rule=='soft_mean':
        first=torch.where(ties,torch.zeros_like(first),first)
    return torch.where(matches.any(1),first,torch.zeros_like(first))
