"""Fixed source-L PA evidence recomposition with the existing deployed head."""
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids as validate_ids

PRIMARY = 'id_backbone.encoder.core.id_backbone'
ENCODER = 'id_backbone.encoder'
SOURCE_RX = (1, 3, 4, 6, 8)
MODES = ('ZERO', 'SELF', 'SWAP_PA')
COEFFICIENT = .1
NORMALIZE_EPS = 1e-4


def _labels(labels, count, device):
    if (not isinstance(labels, torch.Tensor) or labels.shape != (count,)
            or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or labels.device != device or not bool(((labels >= 0) & (labels < 6)).all())):
        raise ValueError('One visible integer source TX label0..5 per packet is required')
    return labels.detach().long()


def _partners(labels, identifiers):
    """One packet at each source RX per TX/eq/day; reject incomplete whole cohort."""
    count = len(labels); ids = validate_ids(identifiers, count)
    if any(identity[1] not in SOURCE_RX or identity[2] not in (1, 2, 3) for identity in ids):
        raise ValueError('Only source RX1/3/4/6/8 and day1/2/3 physical packets are permitted')
    groups = {}
    for i, (label, identity) in enumerate(zip(labels.tolist(), ids)):
        key = (label, identity[0], identity[2])
        groups.setdefault(key, {}).setdefault(identity[1], []).append(i)
    incomplete = [key for key, cells in groups.items()
                  if set(cells) != set(SOURCE_RX) or any(len(indices) != 1 for indices in cells.values())]
    if not count or incomplete:
        return torch.full((count,), -1, dtype=torch.long, device=labels.device), dict(
            eligible=False, reason='empty_cohort' if not count else 'incomplete_TX_eq_day_RX_cells',
            TX_eq_day_groups=len(groups), incomplete_groups=len(incomplete), pair_count=0)
    partners = [-1] * count
    for cells in groups.values():
        indices = [cells[receiver][0] for receiver in SOURCE_RX]
        for i, donor in zip(indices, indices[1:] + indices[:1]): partners[i] = donor
    if sorted(partners) != list(range(count)):
        raise RuntimeError('Fixed RX-cycle partner plan is not a bijection')
    for i, j in enumerate(partners):
        if (i == j or ids[i][-1] == ids[j][-1] or labels[i] != labels[j]
                or ids[i][0] != ids[j][0] or ids[i][2] != ids[j][2] or ids[i][1] == ids[j][1]):
            raise RuntimeError('Fixed donor plan violates TX/eq/day/crossRX/physical separation')
    return torch.tensor(partners, dtype=torch.long, device=labels.device), dict(
        eligible=True, reason=None, TX_eq_day_groups=len(groups), incomplete_groups=0, pair_count=count)


def objective(model, components, labels, physical_ids, mode, role='L_s'):
    """Swap only normalized PA between a fixed sameTX/eq/day crossRX cycle.

    components are actual attached identity/physical/sequence_raw/reference_raw
    tensors plus deployed joint[N,160], all from the same forward. Optional
    logits are accepted but are never used by this loss. SELF exactly follows
    the deployed floating-point operation order:
      ((identity+tanh(PA_gain)*physical)+tanh(ref_gain)*reference_raw)
      + (.2*tanh(SIG_gain))*sequence_raw.
    All three fusion gains and native W are detached; feature tensors remain
    attached. SWAP_PA changes only physical[partner_index]. Both readouts are
    computed for every mode. A missing complete5RX single-packet group skips
    the entire auxiliary cohort. Main ordinary CE is external and unchanged.
    """
    if role != 'L_s' or mode not in MODES or not isinstance(model, nn.Module):
        raise ValueError('Only an existing model and fixed visible L_s modes are permitted')
    required = ('identity','physical','sequence_raw','reference_raw','joint')
    if not isinstance(components, Mapping) or any(key not in components for key in required):
        raise ValueError('Actual identity/physical/sequence/reference/joint components required')
    b = components['identity']
    if (not isinstance(b, torch.Tensor) or b.layout != torch.strided or b.ndim != 2
            or b.shape[1] != 160 or not b.is_floating_point()):
        raise ValueError('Actual finite floating components[N,160] required')
    count = len(b)
    for key in required:
        value = components[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != b.shape or value.dtype != b.dtype or value.device != b.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{key} component shape/dtype/device/finite contract differs')
    y = _labels(labels,count,b.device)
    partners, plan = _partners(y,physical_ids)
    try:
        head = model.get_submodule(PRIMARY+'.cls_head')
        encoder = model.get_submodule(ENCODER)
        pool = model.get_submodule(ENCODER+'.sequence_pool')
        W, PA_gain, reference_gain, SIG_gain = head.weight, head.gain, encoder.response_gain, pool.gain_raw
        scale = float(head.scale)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Actual native residual head/reference/SIG_TIME fusion interface required') from error
    if not math.isfinite(scale) or scale != 30.:
        raise ValueError('Unchanged native cosine scale30 is required; no scale override')
    for name,value,shape in (('default W',W,(6,160)),('PA gain',PA_gain,(160,)),
                             ('reference gain',reference_gain,(160,)),('SIG gain',SIG_gain,(160,))):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != b.dtype or value.device != b.device or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{name} must be the finite actual native tensor on the component device/dtype')
    p, s, r = (components[key] for key in ('physical','sequence_raw','reference_raw'))
    PA = PA_gain.detach().tanh()
    reference = reference_gain.detach().tanh()
    SIG = .2*SIG_gain.detach().tanh()
    self_joint = ((b+PA*p)+reference*r)+SIG*s
    # An omitted reference increment or a component from another call is a
    # direct interface defect, never repaired by subtracting detached joint.
    exact = torch.equal(self_joint.detach(),components['joint'].detach())
    if not exact:
        error = float((self_joint.detach()-components['joint'].detach()).abs().max()) if count else 0.
        raise ValueError(f'SELF does not exactly reconstruct actual four-evidence deployed joint; maxabs={error}')
    readout_partner = partners if plan['eligible'] else torch.arange(count,device=b.device)
    swap_joint = ((b+PA*p[readout_partner])+reference*r)+SIG*s
    normalized_W = F.normalize(W.detach(),dim=1,eps=NORMALIZE_EPS)
    self_logits = scale*F.linear(F.normalize(self_joint,dim=1,eps=NORMALIZE_EPS),normalized_W)
    swap_logits = scale*F.linear(F.normalize(swap_joint,dim=1,eps=NORMALIZE_EPS),normalized_W)
    if not bool(torch.isfinite(self_logits).all() and torch.isfinite(swap_logits).all()):
        raise ValueError('Actual native cosine evidence readout is nonfinite')
    self_ce = F.cross_entropy(self_logits,y,reduction='none')
    swap_ce = F.cross_entropy(swap_logits,y,reduction='none')
    # Zero uses only raw feature endpoints, so every fusion gain and W remains
    # absent from auxiliary autograd, including skipped/inactive controls.
    zero = (b*0.).sum()+(p*0.).sum()+(s*0.).sum()+(r*0.).sum()
    self_mean = self_ce.mean() if count else zero
    swap_mean = swap_ce.mean() if count else zero
    chosen = self_mean if mode=='SELF' else swap_mean
    loss = COEFFICIENT*chosen if plan['eligible'] and mode!='ZERO' else zero
    if not bool(torch.isfinite(loss)):
        raise ValueError('Evidence recomposition auxiliary loss is nonfinite')
    change = swap_joint.detach()-self_joint.detach()
    summary = dict(mode=mode,role=role,input_count=count,eligible_count=count if plan['eligible'] else 0,
        coverage=float(plan['eligible']) if count else 0.,**plan,
        SELF_CE=float(self_mean.detach()) if count else None,
        SWAP_PA_CE=float(swap_mean.detach()) if count else None,
        loss=float(loss.detach()),weighted_loss=float(loss.detach()),
        unweighted_selected_loss=float(chosen.detach()) if plan['eligible'] and mode!='ZERO' else 0.,
        coefficient=COEFFICIENT,scale=scale,normalize_eps=NORMALIZE_EPS,
        self_reconstruction_exact=True,self_reconstruction_max_abs_error=0.,
        swap_changed_packets=int((change!=0).any(1).sum()),
        PA_replacement_norm_mean=float(change.double().norm(dim=1).mean()) if count else None,
        PA_replacement_norm_max=float(change.double().norm(dim=1).max()) if count else None,
        source_RX_counts={str(rx):sum(identity[1]==rx for identity in physical_ids) for rx in SOURCE_RX},
        head_readout_calls=2,head_readout_packets=2*count,extra_IQ_forwards=0,
        sameTX_eq_day_crossRX=True,partner_bijection=plan['eligible'],
        reference_and_sequence_remain_anchor=True,all_three_fusion_gains_detached=True,
        default_W_detached=True,feature_endpoints_detached=False,main_CE_external_unchanged=True,
        classifier_type=type(head).__name__,classifier_weight_shape=[6,160],
        new_parameters=0,extra_head=False,head_fit_updates=0,optimizer_updates=0,
        U_truth_read=False,target_access=False,
        boundary='Caller supplies actual visible source graph; PA recomposition is not real waveform counterfactual or generalization evidence')
    return dict(loss=loss,weighted_loss=loss,summary=summary,partner_index=partners,
                tensors=dict(self_joint=self_joint,swap_joint=swap_joint,self_logits=self_logits,
                             swap_logits=swap_logits,SELF_CE_per_packet=self_ce,SWAP_PA_CE_per_packet=swap_ce))
