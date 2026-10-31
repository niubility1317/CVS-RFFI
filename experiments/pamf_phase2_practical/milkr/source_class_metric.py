"""Source-stage class-noise metric fused into frozen CCA encoder weights.

No covariance, class means, labels, or source examples are stored in the output.
Phase2 loads an ordinary PairedCCA parameter checkpoint and does not fit this.
"""
import numpy as np
from .source_bridge import blocks
from .paired_cca import PairedCCA

STRENGTHS=(0.,.25,.5,.75,.9)


def within_class_covariances(cca,raw,restored,labels,*,allow_subset=False):
    rank=cca.rank
    y=np.asarray(labels,np.int64)
    if y.ndim!=1 or len(y)<2 or (y<0).any() or any(len(v)!=len(y) for v in raw.values()) or any(len(v)!=len(y) for v in restored.values()):
        raise ValueError('Source paired labels/feature rows differ')
    classes,inverse=np.unique(y,return_inverse=True)
    if allow_subset:
        if len(classes)<2 or not np.isin(classes,np.arange(6)).all():raise ValueError('Legal source class subset required')
    elif not np.array_equal(classes,np.arange(6)):raise ValueError('Six legal source classes required')
    a=blocks(cca.transform(raw,restored=False));b=blocks(cca.transform(restored,restored=True));labels=np.tile(inverse,2);covariances=[]
    for left,right in zip(a,b):
        x=np.concatenate((left[:,:rank],right[:,:rank]));means=np.stack([x[labels==c].mean(0) for c in range(len(classes))]);residual=x-means[labels]
        covariance=residual.T@residual/len(residual)
        if not np.isfinite(covariance).all() or float(np.trace(covariance))<=0:raise FloatingPointError('No finite positive source class-noise covariance')
        covariances.append(covariance)
    return tuple(covariances)


def fuse_metric(cca,covariances,strength):
    rank=cca.rank
    if strength not in STRENGTHS or len(covariances)!=8:raise ValueError('Wrong source metric configuration')
    if strength==0.:return cca
    parameters=[]
    for (mx,my,px,py),covariance in zip(cca.parameters,covariances):
        covariance=np.asarray(covariance,np.float64)
        if covariance.shape!=(rank,rank) or not np.isfinite(covariance).all():raise ValueError('Wrong source metric covariance')
        scale=float(np.trace(covariance)/rank)
        mixed=(1-strength)*scale*np.eye(rank)+strength*covariance
        eigen,vectors=np.linalg.eigh(mixed)
        if (eigen<=0).any() or not np.isfinite(eigen).all():raise FloatingPointError('Source class metric not positive definite')
        transform=(vectors/np.sqrt(eigen))@vectors.T
        parameters.append((mx,my,px@transform,py@transform))
    return PairedCCA(parameters,rank)
