"""Joint replacement of actual weighted source F branches; latent oracle only."""
import torch
from .branch_features import BRANCHES

CASES=('native','PA','signature','PA_signature','all_clean')


def paired_cases(actual,parts,clean,reference):
    if (actual.ndim!=2 or actual.shape[1]!=160 or clean.shape!=actual.shape
        or set(parts)!=set(BRANCHES) or set(reference)!=set(BRANCHES)
        or any(v.shape!=actual.shape for v in [*parts.values(),*reference.values()])):
        raise ValueError('Matched actual four weighted 160D source components required')
    total=parts['identity']+parts['PA']+parts['response']+parts['signature']
    target=reference['identity']+reference['PA']+reference['response']+reference['signature']
    torch.testing.assert_close(total,actual,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(target,clean,atol=2e-6,rtol=2e-6)
    def changed(pa,sig):
        return parts['identity']+pa+parts['response']+sig
    return dict(native=actual,PA=changed(reference['PA'],parts['signature']),
        signature=changed(parts['PA'],reference['signature']),
        PA_signature=changed(reference['PA'],reference['signature']),all_clean=clean)
