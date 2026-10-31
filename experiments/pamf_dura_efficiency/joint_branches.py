"""Joint replacement of actual weighted source F branches; latent oracle only."""
import torch
from .branch_features import BRANCHES

def cases_for(auxiliary):
    if auxiliary not in ('PA','response'):
        raise ValueError('Only actual PA or response plus signature is supported')
    return ('native',auxiliary,'signature',auxiliary+'_signature','all_clean')


CASES=cases_for('PA')


def paired_cases(actual,parts,clean,reference,auxiliary='PA'):
    cases=cases_for(auxiliary)
    if (actual.ndim!=2 or actual.shape[1]!=160 or clean.shape!=actual.shape
        or set(parts)!=set(BRANCHES) or set(reference)!=set(BRANCHES)
        or any(v.shape!=actual.shape for v in [*parts.values(),*reference.values()])):
        raise ValueError('Matched actual four weighted 160D source components required')
    total=parts['identity']+parts['PA']+parts['response']+parts['signature']
    target=reference['identity']+reference['PA']+reference['response']+reference['signature']
    torch.testing.assert_close(total,actual,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(target,clean,atol=2e-6,rtol=2e-6)
    def changed(replace_auxiliary,replace_signature):
        pa=reference['PA'] if auxiliary=='PA' and replace_auxiliary else parts['PA']
        response=reference['response'] if auxiliary=='response' and replace_auxiliary else parts['response']
        sig=reference['signature'] if replace_signature else parts['signature']
        return parts['identity']+pa+response+sig
    return dict(zip(cases,(actual,changed(True,False),changed(False,True),changed(True,True),clean)))
