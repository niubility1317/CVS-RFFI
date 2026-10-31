"""Source-only projection and norm control for two loss gradients."""
import math
import torch


def balance_gradients(reference,auxiliary,*,max_aux_ratio=1.):
    """Remove opposing auxiliary component, then limit its total norm.

    This is a pre-optimizer gradient property, not a claim about finite AdamW
    steps or test accuracy. Gradients correspond to the same ordered params.
    """
    if not reference or len(reference)!=len(auxiliary) or not math.isfinite(max_aux_ratio) or max_aux_ratio<0:
        raise ValueError('Aligned nonempty gradients and nonnegative ratio required')
    finite=[]
    for r,a in zip(reference,auxiliary):
        if r.shape!=a.shape or r.device!=a.device:
            raise ValueError('Aligned finite parameter gradients required')
        finite.extend((torch.isfinite(r).all(),torch.isfinite(a).all()))
    if not bool(torch.stack(finite).all()):raise ValueError('Aligned finite parameter gradients required')
    with torch.no_grad():
        ref2=sum((r*r).sum() for r in reference);aux2=sum((a*a).sum() for a in auxiliary)
        dot=sum((r*a).sum() for r,a in zip(reference,auxiliary))
        if not bool(torch.isfinite(torch.stack((ref2,aux2,dot))).all()):raise FloatingPointError('Gradient norm or dot product overflow')
        if float(ref2)<=1e-24:
            return [r.clone() for r in reference],dict(projected=False,scale=0.,ref_norm=0.,aux_norm=float(aux2.sqrt()),cosine=0.)
        projected=bool(dot<0)
        corrected=[a-(dot/ref2)*r if projected else a.clone() for r,a in zip(reference,auxiliary)]
        norm2=sum((a*a).sum() for a in corrected)
        if not bool(torch.isfinite(norm2)):raise FloatingPointError('Projected gradient norm overflow')
        norm=norm2.sqrt();limit=float(max_aux_ratio)*ref2.sqrt()
        scale=torch.clamp(limit/norm.clamp_min(1e-12),max=1.)
        combined=[r+scale*a for r,a in zip(reference,corrected)]
        if not bool(torch.stack([torch.isfinite(v).all() for v in combined]).all()):raise FloatingPointError('Balanced gradient overflow')
        cosine=dot/(ref2.sqrt()*aux2.sqrt()).clamp_min(1e-12)
        return combined,dict(projected=projected,scale=float(scale),ref_norm=float(ref2.sqrt()),
                             aux_norm=float(aux2.sqrt()),cosine=float(cosine))
