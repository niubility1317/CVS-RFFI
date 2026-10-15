"""Global feature-gradient calibration: ratio of medians, never layer-ratio median."""
import math
import statistics
import torch

TARGET_RATIO=.1
BOUNDS=(1e-4,10.)
MIN_GRADIENT=1e-8


def vectors(loss,named):
    gradients=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True,retain_graph=True)
    values={name:(torch.zeros_like(p) if g is None else g.detach()) for (name,p),g in zip(named,gradients)}
    if any(not torch.isfinite(g).all() for g in values.values()):
        raise FloatingPointError('Nonfinite actual feature gradient')
    return values


def norm(values):
    return math.sqrt(sum(float(v.double().square().sum()) for v in values.values()))


def dot(a,b):
    return sum(float((a[k].double()*b[k].double()).sum()) for k in a)


def compare(a,b):
    an,bn=norm(a),norm(b)
    return dict(satellite_CE_weighted_norm=an,auxiliary_weighted_norm=bn,
        ratio=bn/(an+1e-12),cosine=dot(a,b)/(an*bn) if an*bn else None)


def coefficient(ce_norms,aux_norms):
    if len(ce_norms)!=len(aux_norms) or not ce_norms:
        raise ValueError('Same source batches required for global calibration')
    if any(not math.isfinite(v) or v<0 for v in (*ce_norms,*aux_norms)):
        raise ValueError('Invalid global calibration norm')
    ce,aux=statistics.median(ce_norms),statistics.median(aux_norms)
    if ce<MIN_GRADIENT or aux<MIN_GRADIENT:
        raise ValueError('Near-zero global feature gradient: check wiring; do not amplify')
    raw=TARGET_RATIO*ce/(aux+1e-12)
    if not BOUNDS[0]<=raw<=BOUNDS[1]:
        raise ValueError(f'Calibration outside registered bounded strength: {raw}; no silent amplification')
    return dict(coefficient=raw,median_satellite_CE_gradient=ce,median_auxiliary_gradient=aux,
        target_gradient_ratio=TARGET_RATIO,weighted_ratio_of_medians=raw*aux/ce,
        bounds=list(BOUNDS),clipped=False,denominator='weighted satellite CE = .5 * CE',
        estimator='median global feature norm ratio; unused gradients zero; default W excluded')
