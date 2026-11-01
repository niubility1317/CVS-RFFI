"""Fixed cyclic IQ timing candidates; circular boundaries are an approximation."""
import torch


def shift_candidates(iq, shifts):
    """Construct zero-first shifts ordered by increasing absolute sample offset."""
    if (iq.ndim!=3 or iq.shape[1:]!=(2,256) or not shifts or shifts[0]!=0
            or any(type(s) is not int or abs(s)>=256 for s in shifts)
            or any(abs(a)>abs(b) for a,b in zip(shifts,shifts[1:]))):
        raise ValueError('Phase1 IQ and ordered bounded integer shifts required')
    return torch.stack([torch.roll(iq,s,dims=2) for s in shifts],dim=1)
