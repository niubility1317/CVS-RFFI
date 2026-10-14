"""Fixed evidence-weakening stop loss; no model/gate/teacher state is created."""
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_evidence_recomposition20.capture import HEAD_PATH, SEQUENCE_PATH
from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids as validate_ids

BETA = .5
PROBABILITY = .9
COEFFICIENT = .1
NORMALIZE_EPS = 1e-4
MODES = ('ZERO', 'SELF_STOP', 'EVIDENCE_STOP')
VIEWS = ('self', 'PAweak', 'REFweak')


def readouts(model,components):
    """Three original-head readouts from same-call actual deployed components.

    This label-free function also works inside caller-owned no_grad source-V
    proxy evaluation. It does not change eval flags. All modes retain the same
    three-readout cost. PA/reference raw endpoints and W/three fusion gains are
    detached; identity and sequence_raw remain attached. No new head/module or
    IQ forward is used. Physical/reference evidence stays numerically present.
    """
    required=('identity','physical','reference_raw','sequence_raw','joint')
    if not isinstance(model,nn.Module) or not isinstance(components,Mapping) or any(key not in components for key in required):
        raise ValueError('Existing model and actual same-call four-evidence components required')
    a=components['identity']
    if (not isinstance(a,torch.Tensor) or a.layout!=torch.strided or a.ndim!=2
            or a.shape[1]!=160 or len(a)<1 or not a.is_floating_point()):
        raise ValueError('Finite real components[B,160] with nonempty batch required')
    for key in required:
        value=components[key]
        if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.shape!=a.shape
                or value.dtype!=a.dtype or value.device!=a.device or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{key} component shape/dtype/device/finite contract differs')
    try:
        head=model.get_submodule(HEAD_PATH);pool=model.get_submodule(SEQUENCE_PATH)
        encoder=model.get_submodule('id_backbone.encoder')
        W,PA_gain,reference_gain,SIG_gain=head.weight,head.gain,encoder.response_gain,pool.gain_raw
        scale=float(head.scale)
    except (AttributeError,TypeError,ValueError) as error:
        raise ValueError('Actual default residual head/reference/SIG_TIME fusion interface required') from error
    if not math.isfinite(scale) or scale!=30.:
        raise ValueError('Actual unchanged default scale30 required; no scale adjustment')
    for name,value,shape in (('W',W,(6,160)),('PA gain',PA_gain,(160,)),('reference gain',reference_gain,(160,)),('SIG gain',SIG_gain,(160,))):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=a.dtype
                or value.device!=a.device or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Actual {name} shape/dtype/device/finite contract differs')
    b=PA_gain.detach().tanh()*components['physical'].detach()
    c=reference_gain.detach().tanh()*components['reference_raw'].detach()
    s=(.2*SIG_gain.detach().tanh())*components['sequence_raw']
    joint=((a+b)+c)+s
    if not torch.equal(joint.detach(),components['joint'].detach()):
        raise ValueError('SELF four-term reconstruction differs bitwise from actual deployed joint')
    joints={'self':joint,'PAweak':((a+BETA*b)+c)+s,'REFweak':((a+b)+BETA*c)+s}
    normalized_W=F.normalize(W.detach(),dim=1,eps=NORMALIZE_EPS)
    result={}
    for view in VIEWS:
        logits=scale*F.linear(F.normalize(joints[view],dim=1,eps=NORMALIZE_EPS),normalized_W)
        if not bool(torch.isfinite(logits).all()): raise ValueError('Evidence challenge cosine readout is nonfinite')
        result[view]=dict(joint=joints[view],logits=logits)
    result['summary']=dict(input_count=len(a),beta=BETA,scale=scale,normalize_eps=NORMALIZE_EPS,
        SELF_joint_bitwise_exact=True,head_readout_calls=3,head_readout_packets=3*len(a),extra_IQ_forwards=0,
        physical_reference_endpoints_detached=True,default_W_three_gains_detached=True,
        identity_sequence_endpoints_detached=False,new_parameters=0,extra_head=False,optimizer_updates=0,
        classifier_type=type(head).__name__,classifier_weight_shape=[6,160])
    return result


def objective(model,components,targets,accept_mask,physical_ids,mode,role='L_s'):
    """.1/(2B) sum original q*(stop(view1)+stop(view2)), stop=ReLU(log .9-log p).

    L_s requires full-True q and caller's visible true targets. U_s targets and
    q are the caller's unchanged original trusted teacher pseudo-label/gate;
    this pure function cannot prove that provenance. Challenges never regate
    q or demand their argmax agree with targets. SELF_STOP uses two SELF terms;
    EVIDENCE_STOP uses PAweak and REFweak. ZERO connects only a/sequence_raw.
    Main CE and original head/gain joint training are caller-owned, unchanged.
    """
    if role not in ('L_s','U_s') or mode not in MODES:
        raise ValueError('Only registered modes and source L_s/U_s training roles are permitted')
    views=readouts(model,components)
    a=components['identity'];batch=len(a)
    if (not isinstance(targets,torch.Tensor) or targets.layout!=torch.strided or targets.shape!=(batch,)
            or targets.device!=a.device or targets.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)
            or not bool(((targets>=0)&(targets<6)).all())):
        raise ValueError('One integer class0..5 target per packet required; U targets must come from original teacher')
    if (not isinstance(accept_mask,torch.Tensor) or accept_mask.layout!=torch.strided
            or accept_mask.shape!=(batch,) or accept_mask.dtype!=torch.bool or accept_mask.device!=a.device):
        raise ValueError('Original accept_mask must be one bool per packet on the component device')
    if role=='L_s' and not bool(accept_mask.all()): raise ValueError('Source L_s uses all visible true labels, q must be full True')
    if not isinstance(physical_ids,(list,tuple)): raise ValueError('Complete five-native-integer physical ID sequence required')
    ids=validate_ids(physical_ids,batch)
    if any(identity[1] not in (1,3,4,6,8) or identity[2] not in (1,2,3) for identity in ids):
        raise ValueError('Training physical IDs must belong to allowed source RX/day')
    y=targets.detach().long();q=accept_mask.detach()
    logp={view:F.log_softmax(views[view]['logits'],dim=1).gather(1,y[:,None]).squeeze(1) for view in VIEWS}
    probability={view:value.exp() for view,value in logp.items()}
    stop={view:F.relu(math.log(PROBABILITY)-value) for view,value in logp.items()}
    selected=('PAweak','REFweak') if mode=='EVIDENCE_STOP' else ('self','self')
    packet_stop=stop[selected[0]]+stop[selected[1]]
    unweighted=(q.to(a.dtype)*packet_stop).sum()/(2*batch)
    zero=(a*0.).sum()+(components['sequence_raw']*0.).sum()
    loss=zero if mode=='ZERO' else COEFFICIENT*unweighted
    if not bool(torch.isfinite(loss)): raise ValueError('Evidence compensation stop loss is nonfinite')
    accepted=int(q.sum());active=q & (packet_stop>0)
    summary=dict(views['summary'],mode=mode,role=role,probability=PROBABILITY,coefficient=COEFFICIENT,
        accepted_count=accepted,accepted_fraction=accepted/batch,active_packets=int(active.sum()),
        loss_active_packets=0 if mode=='ZERO' else int(active.sum()),
        active_selected_view_packets=sum(int((q & (stop[view]>0)).sum()) for view in selected),
        selected_views=list(selected),denominator_full_batch=batch,denominator_view_packets=2*batch,
        loss=float(loss.detach()),weighted_loss=float(loss.detach()),
        unweighted_loss=0. if mode=='ZERO' else float(unweighted.detach()),
        diagnostic_unweighted_stop=float(unweighted.detach()),
        original_accept_mask_preserved=True,accept_mask_recomputed=False,student_argmax_agreement_required=False,
        target_probability_mean={view:float(probability[view].detach().mean()) for view in VIEWS},
        accepted_target_probability_mean={view:float(probability[view][q].detach().mean()) if accepted else None for view in VIEWS},
        accepted_argmax_mismatch={view:int((q & (views[view]['logits'].detach().argmax(1)!=y)).sum()) for view in VIEWS},
        physical_reference_aux_gradient_None=True,default_W_three_gains_aux_gradient_None=True,
        main_default_head_joint_training_unchanged=True,head_fit_updates=0,U_truth_read=False,target_access=False,
        targets_provenance='caller visible true L labels' if role=='L_s' else 'caller original trusted teacher pseudo-labels',
        boundary='Pure loss cannot prove label/gate/data provenance; no U truth is requested, no new gate or efficacy claim')
    return dict(loss=loss,weighted_loss=loss,summary=summary,
                tensors=dict(readouts=views,target_log_probabilities=logp,target_probabilities=probability,
                             stop_terms=stop,selected_packet_stop=packet_stop,accept_mask=q))
