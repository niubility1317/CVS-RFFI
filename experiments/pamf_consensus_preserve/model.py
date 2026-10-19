import torch
from experiments.pamf_quality_preserve.model import keep_input
from experiments.pamf_change_preserve.model import relative_change

CHANGE_THRESHOLD=.02


def select_iq(received,restored,damage_logit):
    if received.shape!=restored.shape or received.ndim!=3 or received.shape[1:]!=(2,256):
        raise ValueError('Matched IQ packets required')
    if not torch.isfinite(received).all() or not torch.isfinite(restored).all():
        raise ValueError('Finite IQ packets required')
    keep=keep_input(damage_logit)|(relative_change(received,restored)<=CHANGE_THRESHOLD)
    if keep.shape!=(len(received),):raise ValueError('One independent decision per IQ packet required')
    return torch.where(keep[:,None,None],received,restored),keep
