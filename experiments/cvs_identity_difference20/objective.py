"""Current-batch raw identity-difference directions; no W, head or stored bank."""
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_evidence_recomposition20.capture import HEAD_PATH, SEQUENCE_PATH
from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids as validate_ids

SOURCE_RX = (1, 3, 4, 6, 8)
CLASS_PAIRS = tuple((y,k) for y in range(6) for k in range(y+1,6))
MODES = ('ZERO', 'SELF_DIFF', 'CROSS_DIFF')
COEFFICIENT = .1
ALIGNMENT = .9
NORMALIZE_EPS = 1e-4


def _raw_joint(model,components):
    required=('identity','physical','reference_raw','sequence_raw','joint')
    if not isinstance(model,nn.Module) or not isinstance(components,Mapping) or any(key not in components for key in required):
        raise ValueError('Existing model and actual same-call four-evidence components required')
    a=components['identity']
    if (not isinstance(a,torch.Tensor) or a.layout!=torch.strided or a.shape!=(180,160)
            or not a.is_floating_point()):
        raise ValueError('Exactly two complete source L90 sides with floating components[180,160] required')
    for key in required:
        value=components[key]
        if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.shape!=a.shape
                or value.dtype!=a.dtype or value.device!=a.device or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{key} component shape/dtype/device/finite contract differs')
    try:
        head=model.get_submodule(HEAD_PATH);encoder=model.get_submodule('id_backbone.encoder')
        pool=model.get_submodule(SEQUENCE_PATH)
        gains=(head.gain,encoder.response_gain,pool.gain_raw)
    except AttributeError as error:
        raise ValueError('Actual PA/reference/SIG_TIME fusion gain interface required') from error
    for gain in gains:
        if (not isinstance(gain,torch.Tensor) or gain.shape!=(160,) or gain.dtype!=a.dtype
                or gain.device!=a.device or not bool(torch.isfinite(gain).all())):
            raise ValueError('Actual three fusion gains must be finite same-device/dtype160 tensors')
    b=gains[0].detach().tanh()*components['physical'].detach()
    c=gains[1].detach().tanh()*components['reference_raw'].detach()
    s=(.2*gains[2].detach().tanh())*components['sequence_raw']
    z=((a+b)+c)+s
    if not torch.equal(z.detach(),components['joint'].detach()):
        raise ValueError('Auxiliary SELF four-term raw joint differs bitwise from actual deployed joint')
    return z


def _cells(y,rx,side,physical_ids,device):
    for name,value in (('visible y',y),('rx',rx),('side',side)):
        if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.shape!=(180,)
                or value.dtype!=torch.long or value.device!=device):
            raise ValueError(f'{name} must be torch.long[180] on the component device; side0=A/1=B')
    if not bool(((y>=0)&(y<6)).all()) or not bool(((side==0)|(side==1)).all()):
        raise ValueError('Only visible y0..5 and fixed side0=A/1=B are permitted')
    if not isinstance(physical_ids,(list,tuple)): raise ValueError('Complete native-integer physical ID list required')
    ids=validate_ids(physical_ids,180)
    rows=list(zip(side.tolist(),rx.tolist(),y.tolist(),ids));cells={}
    for i,(which,receiver,label,identity) in enumerate(rows):
        if receiver not in SOURCE_RX or identity[1]!=receiver or identity[2] not in (1,2,3):
            raise ValueError('Current source RX/day metadata must agree with complete physical ID')
        key=(which,receiver,label,identity[2])
        if key in cells: raise ValueError('Repeated RX/class/day/side cell; every cell requires exactly one distinct physical packet')
        cells[key]=i
    expected={(which,receiver,label,day) for which in (0,1) for receiver in SOURCE_RX for label in range(6) for day in (1,2,3)}
    if set(cells)!=expected:
        raise ValueError('Incomplete or unbalanced source sides: each RX/class/side must contain days1/2/3 exactly once')
    return cells


def _stats(value):
    value=value.detach().double()
    return dict(min=float(value.min()),mean=float(value.mean()),max=float(value.max()),std=float(value.std(unbiased=False)))


def _norm_stats(value):
    norm=value.detach().double().norm(dim=-1)
    return dict(_stats(norm),nearzero_count=int((norm<=NORMALIZE_EPS).sum()),exact_zero_count=int((norm==0).sum()),count=norm.numel())


def objective(model,components,y,rx,side,physical_ids,mode,role='L_s'):
    """Fixed .1/300 sum ReLU(.9-cos(delta_A_r,delta_B_t)), with both ends attached.

    side is strictly torch.long0=A/1=B. Each side consists of every5sourceRX x
    6class x3day cell once, and all180 full physical IDs/base IDs are distinct.
    Mean raw joint first, subtract class means second, then normalize differences
    with eps1e-4. All15unordered class pairs use the same orientation on A/B.
    CROSS uses20ordered crossRX pairs; SELF uses75 sameRX pairs repeated4 times.
    Every mode actually computes both300-dot-product/hinge paths. ZERO returns
    only identity/sequence_raw connected zero. W is never read or used.
    """
    if role!='L_s' or mode not in MODES: raise ValueError('Only visible L_s and fixed identity-difference modes permitted')
    z=_raw_joint(model,components);cells=_cells(y,rx,side,physical_ids,z.device)
    means=[]
    for which in (0,1):
        # Fixed day order keeps cell arithmetic stable under input row order.
        means.append(torch.stack([torch.stack([z[[cells[which,receiver,label,day] for day in (1,2,3)]].mean(0)
            for label in range(6)]) for receiver in SOURCE_RX]))
    mu_A,mu_B=means
    if not bool(torch.isfinite(mu_A).all() and torch.isfinite(mu_B).all()): raise ValueError('Raw source centroids are nonfinite')
    delta_A=torch.stack([mu_A[:,label]-mu_A[:,other] for label,other in CLASS_PAIRS],dim=1)
    delta_B=torch.stack([mu_B[:,label]-mu_B[:,other] for label,other in CLASS_PAIRS],dim=1)
    if not bool(torch.isfinite(delta_A).all() and torch.isfinite(delta_B).all()): raise ValueError('Raw class-difference vectors are nonfinite')
    unit_A=F.normalize(delta_A,dim=-1,eps=NORMALIZE_EPS)
    unit_B=F.normalize(delta_B,dim=-1,eps=NORMALIZE_EPS)
    self_A=unit_A.reshape(75,160).repeat(4,1);self_B=unit_B.reshape(75,160).repeat(4,1)
    cos_self=(self_A*self_B).sum(-1)
    rx_pairs=[(r,t) for r in range(5) for t in range(5) if r!=t]
    cross_A=torch.cat([unit_A[r] for r,t in rx_pairs],dim=0)
    cross_B=torch.cat([unit_B[t] for r,t in rx_pairs],dim=0)
    cos_cross=(cross_A*cross_B).sum(-1)
    if not bool(torch.isfinite(cos_self).all() and torch.isfinite(cos_cross).all()): raise ValueError('Difference cosine values are nonfinite')
    self_terms=F.relu(ALIGNMENT-cos_self);cross_terms=F.relu(ALIGNMENT-cos_cross)
    self_loss=self_terms.sum()/300;cross_loss=cross_terms.sum()/300
    zero=(components['identity']*0.).sum()+(components['sequence_raw']*0.).sum()
    loss=zero if mode=='ZERO' else COEFFICIENT*(self_loss if mode=='SELF_DIFF' else cross_loss)
    if not bool(torch.isfinite(loss)): raise ValueError('Identity-difference loss is nonfinite')
    norms=torch.cat([delta_A.detach(),delta_B.detach()],dim=0)
    summary=dict(mode=mode,role=role,input_count=180,side_A_count=90,side_B_count=90,
        side_encoding='torch.long0=A/1=B',RX_class_side_cells=60,packets_per_cell=3,days_per_cell=3,
        class_pairs=15,ordered_cross_RX_pairs=20,SELF_base_terms=75,SELF_computed_terms=300,CROSS_computed_terms=300,
        SELF_unweighted_loss=float(self_loss.detach()),CROSS_unweighted_loss=float(cross_loss.detach()),
        SELF_active75=int((self_terms[:75]>0).sum()),SELF_active300=int((self_terms>0).sum()),
        CROSS_active300=int((cross_terms>0).sum()),loss=float(loss.detach()),weighted_loss=float(loss.detach()),
        coefficient=COEFFICIENT,alignment=ALIGNMENT,normalize_eps=NORMALIZE_EPS,denominator=300,
        difference_norms=_norm_stats(norms),difference_A_norms=_norm_stats(delta_A),difference_B_norms=_norm_stats(delta_B),
        SELF_cosine=_stats(cos_self[:75]),CROSS_cosine=_stats(cos_cross),SELF_joint_bitwise_exact=True,
        raw_mean_then_subtract_then_normalize=True,both_A_B_endpoints_attached=True,
        PA_reference_endpoints_detached=True,three_fusion_gains_detached=True,default_W_used=False,
        label_permutation_invariant_by_construction=True,RX_weighting_uniform=True,
        all_original_physical_packets_distinct=True,temporary_centroids_only=True,persistent_bank=False,
        new_parameters=0,new_head=False,head_readout_calls=0,extra_IQ_forwards=0,optimizer_updates=0,head_fit_updates=0,
        main_default_head_joint_training_unchanged=True,U_truth_read=False,target_access=False,
        boundary='Caller proves visible L provenance; common-offset cancellation is a raw-coordinate algebraic example, not hardware/causal/DG or efficacy proof')
    return dict(loss=loss,weighted_loss=loss,summary=summary,tensors=dict(raw_joint=z,mu_A=mu_A,mu_B=mu_B,
        delta_A=delta_A,delta_B=delta_B,cos_SELF75=cos_self[:75],cos_SELF300=cos_self,cos_CROSS300=cos_cross,
        terms_SELF300=self_terms,terms_CROSS300=cross_terms,
        class_pairs=torch.tensor(CLASS_PAIRS,dtype=torch.long,device=z.device)))
