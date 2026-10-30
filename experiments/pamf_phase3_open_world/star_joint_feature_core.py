"""Joint frozen F0/signal final features with the existing source mixture fit."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as conditional

SCHEMA=density.JOINT_SCHEMA
METHODS=('pooled_ratio','predicted_ratio','joint_pooled_ratio','joint_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
DIMENSION=320


def join(u,z):
    """Each source representation contributes half the squared unit norm."""
    left=density.unit(u,dimension=160);right=density.unit(z,dimension=160)
    if left.shape!=right.shape:raise ValueError('Aligned nonempty F0/signal feature rows required')
    result=(np.concatenate((left,right),axis=1)/np.sqrt(2.)).astype(np.float32)
    if result.shape!=(len(left),DIMENSION) or not np.isfinite(result).all():raise ValueError('Finite FP32 joint320 features required')
    return result


def validate_joint(state):
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=SCHEMA or state['known_means'].shape!=(90,DIMENSION):raise ValueError('Only the explicit frozen joint320 ninety-mode schema is allowed')
    return state


def fit(Lu,Lz,Ly,Lrx,Lday,OEu,OEz,OE_rx,OE_day):
    known,kdiag=density.fit_known(join(Lu,Lz),Ly,Lrx,Lday,dimension=DIMENSION)
    state,bdiag=density.fit_background(known,join(OEu,OEz),OE_rx,OE_day)
    validate_joint(state)
    if any(not np.array_equal(v,state[k]) for k,v in known.items()):raise ValueError('Background fitting changed the joint known aggregates')
    return state,dict(known=kdiag,background=bdiag,dimension=DIMENSION,
        representation='FP64 unit F0 and signal; concat/sqrt(2); FP32; canonical FP64 density unit',
        source_only=True,neural_updates=0,identity_head_updates=0,query_inputs_used=False,OE_TX_labels_used=False)


def raw_scores(original,joint,u,z,closed):
    validate_joint(joint)
    original_scores=conditional.raw_scores(original,z,closed)
    features=join(u,z);white=(density.unit(features,dimension=DIMENSION)-joint['origin'])@joint['whitener']
    pooled=density._log_density(joint,white,'known');background=density._log_density(joint,white,'background')
    labels=np.asarray(closed);predicted=np.empty(len(white),np.float64)
    for c in density.CLASSES:
        rows=np.flatnonzero(labels==c)
        if not len(rows):continue
        selection=slice(c*15,(c+1)*15);variance=joint['known_variances'][selection]
        terms=(-np.log(15.)-DIMENSION/2*np.log(2*np.pi*variance)[None]
            -density._distances(white[rows],joint['known_means'][selection])/(2*variance[None]))
        high=terms.max(axis=1)
        predicted[rows]=high+np.log(np.exp(terms-high[:,None]).sum(axis=1))+float(joint['log_abs_det_whitener'])
    result=np.column_stack((original_scores[:,1],original_scores[:,3],pooled-background,predicted-background))
    if result.shape!=(len(features),4) or result.dtype!=np.float64 or not np.isfinite(result).all():raise ValueError('Four finite frozen original/joint density-ratio scores required')
    return result
