"""Only repair excess true-class risk beyond the actual destination packet."""
import copy

import torch
from torch.nn import functional as F

from experiments.cvs_identity_transport20.relationship import _bank, SOURCE_RX, SOURCE_DAYS
from experiments.cvs_identity_decision_transfer20.objective import _stats, SCALE, EPS, COEFFICIENT
from experiments.cvs_identity_contrast_transfer20.objective import _exposure, _coverage_row


MODES = ('ZERO', 'SAME_RX', 'CROSS_RX')
TOLERANCE = 1e-6
STOP_RULE = 'no additional trueclass risk beyond actualdestination'


def _excess(virtual_ce, factual_ce):
    # ReLU has derivative zero at the exact boundary; clamp_min does not.
    return torch.relu(virtual_ce - factual_ce.detach() - TOLERANCE)


def _decisions(virtual, factual, labels, weight):
    """Default-head formulas; the factual threshold and class W are stopped.

    Factual is an actual destination packet with the same true TX as virtual's
    origin. Its readout is repeated for every reference, so budgets count both
    3600 virtual and 3600 factual readouts per control, without shared180 reuse.
    """
    if (not isinstance(virtual, torch.Tensor) or virtual.layout != torch.strided
            or virtual.ndim != 2 or virtual.shape[1] != 160 or not len(virtual)
            or virtual.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(virtual).all())
            or not isinstance(factual, torch.Tensor) or factual.layout != torch.strided
            or factual.shape != virtual.shape or factual.dtype != virtual.dtype
            or factual.device != virtual.device or not bool(torch.isfinite(factual).all())
            or not isinstance(labels, torch.Tensor) or labels.layout != torch.strided
            or labels.shape != (len(virtual),) or labels.dtype != torch.long
            or labels.device != virtual.device or bool(((labels < 0) | (labels > 5)).any())
            or not isinstance(weight, torch.Tensor) or weight.layout != torch.strided
            or weight.shape != (6,160) or weight.dtype not in (torch.float32, torch.float64)
            or weight.device != virtual.device or not bool(torch.isfinite(weight).all())):
        raise ValueError('Finite paired views[N,160], true labels[N], and existing W[6,160] required')
    fixed_W = F.normalize(weight.detach().double(), dim=1, eps=EPS)
    logits = SCALE * F.linear(F.normalize(virtual.double(), dim=1, eps=EPS), fixed_W)
    log_true = F.log_softmax(logits, dim=1).gather(1, labels[:,None]).squeeze(1)
    ce = -log_true
    with torch.no_grad():
        factual_logits = SCALE * F.linear(F.normalize(factual.detach().double(), dim=1, eps=EPS), fixed_W)
        factual_log_true = F.log_softmax(factual_logits, dim=1).gather(1, labels[:,None]).squeeze(1)
        factual_ce = -factual_log_true
        W32 = F.normalize(weight.detach().float(), dim=1, eps=EPS)
        logits32 = SCALE * F.linear(F.normalize(virtual.detach().float(), dim=1, eps=EPS), W32)
        factual_logits32 = SCALE * F.linear(F.normalize(factual.detach().float(), dim=1, eps=EPS), W32)
        probability32 = F.softmax(logits32, dim=1).gather(1,labels[:,None]).squeeze(1)
        factual_probability32 = F.softmax(factual_logits32, dim=1).gather(1,labels[:,None]).squeeze(1)
        ce32 = -F.log_softmax(logits32, dim=1).gather(1,labels[:,None]).squeeze(1)
        factual_ce32 = -F.log_softmax(factual_logits32, dim=1).gather(1,labels[:,None]).squeeze(1)
    signed_gap = ce - factual_ce
    excess = _excess(ce, factual_ce)
    return dict(logits=logits, probability=log_true.exp(), CE=ce, stop_terms=excess,
        active=excess.detach()>0., errors=logits.detach().argmax(dim=1)!=labels,
        norm=virtual.double().norm(dim=1), factual_CE=factual_ce,
        factual_probability=factual_log_true.exp(), factual_norm=factual.detach().double().norm(dim=1),
        excess_risk=signed_gap, positive_excess_risk=excess,
        FP32_logit_difference=(logits.detach()-logits32.double()).abs(),
        FP32_true_probability_difference=(log_true.detach().exp()-probability32.double()).abs(),
        FP32_CE_difference=(ce.detach()-ce32.double()).abs(),
        factual_FP32_logit_difference=(factual_logits-factual_logits32.double()).abs(),
        factual_FP32_true_probability_difference=(factual_log_true.exp()-factual_probability32.double()).abs(),
        factual_FP32_CE_difference=(factual_ce-factual_ce32.double()).abs())


def _risk_row(decision, selected, origin_indices):
    row = _coverage_row(decision, selected, origin_indices)
    row.update(mean_factual_CE=float(decision['factual_CE'][selected].mean()),
        mean_excess_risk=float(decision['excess_risk'][selected].detach().mean()),
        mean_positive_excess_risk=float(decision['positive_excess_risk'][selected].detach().mean()))
    return row


def losses(joint, bank, weight, source_role='L_s'):
    """Same fixed individual-reference views; no confidence stop or fitted state.

    Forward is a_y-a_c+b_c against actual b_y, reverse b_y-b_c+a_c against
    actual a_y, with all five c!=y. Virtual endpoints retain their real joint
    graph and three fusion gains. Factual CE is a detached comparison threshold.
    """
    labels, _, _, groups = _bank(joint, bank, source_role)
    if (not isinstance(weight, torch.Tensor) or weight.layout != torch.strided or weight.shape != (6,160)
            or weight.dtype not in (torch.float32, torch.float64) or weight.device != joint.device
            or not bool(torch.isfinite(weight).all())):
        raise ValueError('Finite existing default W[6,160] on joint device required')
    features = joint.double()
    target_rows = torch.tensor([y for y in range(6) for c in range(6) if y != c], device=joint.device)
    reference_rows = torch.tensor([c for y in range(6) for c in range(6) if y != c], device=joint.device)
    values, summaries = {}, {}
    for mode in ('SAME_RX', 'CROSS_RX'):
        pairs = [(receiver,receiver,day,repeat) for repeat in range(4)
                 for receiver in SOURCE_RX for day in SOURCE_DAYS] if mode == 'SAME_RX' else [
                     (anchor,donor,day,0) for anchor in SOURCE_RX for donor in SOURCE_RX
                     if anchor != donor for day in SOURCE_DAYS]
        views, origin_features, destinations, targets, factual_labels, reference_labels = [], [], [], [], [], []
        origin_indices, destination_RX, records = [], [], []
        endpoint = dict(anchor=[0]*180, donor=[0]*180)
        roles = {name:[0]*180 for name in ('origin','reference_source','reference_destination')}
        for anchor, donor, day, repeat in pairs:
            ia,ib = groups[(anchor,day,0)],groups[(donor,day,1)]
            a,b = features.index_select(0,ia),features.index_select(0,ib)
            ay,ac = a.index_select(0,target_rows),a.index_select(0,reference_rows)
            by,bc = b.index_select(0,target_rows),b.index_select(0,reference_rows)
            views.extend((ay-ac+bc,by-bc+ac)); origin_features.extend((ay,by)); destinations.extend((by,ay))
            ya,yb = labels.index_select(0,ia),labels.index_select(0,ib)
            targets.extend((ya.index_select(0,target_rows),yb.index_select(0,target_rows)))
            factual_labels.extend((yb.index_select(0,target_rows),ya.index_select(0,target_rows)))
            reference_labels.extend((ya.index_select(0,reference_rows),yb.index_select(0,reference_rows)))
            origin_indices.extend((ia.index_select(0,target_rows),ib.index_select(0,target_rows)))
            destination_RX.extend(([donor]*30,[anchor]*30))
            records.append(dict(anchor_RX=anchor,donor_RX=donor,day=day,repeat=repeat,
                directions=['a_y-a_c+b_c','b_y-b_c+a_c'],label_source_sides=[0,1],
                destination_RX=[donor,anchor],factual=['b_y','a_y'],
                references_per_origin=5,decisions_per_direction=30))
            for index in ia.tolist(): endpoint['anchor'][index]+=1
            for index in ib.tolist(): endpoint['donor'][index]+=1
            for origin,source_ref,destination_ref in (
                    (ia.index_select(0,target_rows),ia.index_select(0,reference_rows),ib.index_select(0,reference_rows)),
                    (ib.index_select(0,target_rows),ib.index_select(0,reference_rows),ia.index_select(0,reference_rows))):
                for role,indices in zip(roles,(origin,source_ref,destination_ref)):
                    for index in indices.tolist(): roles[role][index]+=1
        if (len(pairs)!=60 or len(views)!=120 or any(a+b!=4 for a,b in zip(endpoint['anchor'],endpoint['donor']))
                or any(count!=20 for role in roles.values() for count in role)):
            raise RuntimeError('Fixed60 pairs/all5 references/20 each triple-role exposure differs')
        transported,factual = torch.cat(views),torch.cat(destinations)
        target,reference,actual_labels = torch.cat(targets),torch.cat(reference_labels),torch.cat(factual_labels)
        origins = torch.cat(origin_indices)
        receiving_RX = torch.tensor([receiver for group in destination_RX for receiver in group],device=joint.device)
        if (transported.shape!=(3600,160) or factual.shape!=(3600,160) or target.shape!=(3600,)
                or bool((target==reference).any()) or not torch.equal(target,actual_labels)):
            raise RuntimeError('Exactly3600 origin-labeled virtual and same-TX actual destination pairs required')
        decision = _decisions(transported,factual,target,weight)
        values[mode] = (COEFFICIENT*decision['stop_terms'].sum()/3600).to(dtype=joint.dtype)
        row = _risk_row(decision,torch.ones(3600,dtype=torch.bool,device=joint.device),origins)
        difference = transported-torch.cat(origin_features)
        destination_error = transported-factual
        unique = {(record['anchor_RX'],record['donor_RX'],record['day']) for record in records}
        summaries[mode] = dict(row,mode=mode,loss=float(values[mode].detach()),source_role='L_s',
            active_origin_fraction=row['active_origin_packet_count']/180,
            matrix_count=60,unique_matrices=len(unique),unique_matrix_count=len(unique),
            synthetic_views=120,synthetic_decisions=3600,fixed_decision_denominator=3600,
            virtual_default_class_decisions=3600,factual_default_class_decisions=3600,
            total_default_class_decisions=7200,unique_factual_packets=180,
            factual_readout_reuse=False,references_per_origin=5,target_reference_pairs_per_direction=30,
            by_anchor_RX={str(receiver):_risk_row(decision,receiving_RX==receiver,origins) for receiver in SOURCE_RX},
            by_reference_TX={str(label):_risk_row(decision,reference==label,origins) for label in range(6)},
            by_target_TX={str(label):_risk_row(decision,target==label,origins) for label in range(6)},
            exposures=_exposure(endpoint,roles,bank),actual_pairs=records,
            transfer_difference_RMS=float(difference.detach().square().mean().sqrt()),
            transfer_difference_norm=_stats(difference.norm(dim=1)),
            to_actual_destination_identity_RMS=float(destination_error.detach().square().mean().sqrt()),
            to_actual_destination_identity_norm=_stats(destination_error.norm(dim=1)),
            destination_identity_comparison='diagnostic only; no destination feature matching loss',
            default_cos_FP64_FP32_max_abs_logit_difference=float(decision['FP32_logit_difference'].max()),
            default_cos_FP64_FP32_max_abs_true_probability_difference=float(decision['FP32_true_probability_difference'].max()),
            virtual_FP64_FP32_max_abs_CE_difference=float(decision['FP32_CE_difference'].max()),
            factual_default_cos_FP64_FP32_max_abs_logit_difference=float(decision['factual_FP32_logit_difference'].max()),
            factual_default_cos_FP64_FP32_max_abs_true_probability_difference=float(decision['factual_FP32_true_probability_difference'].max()),
            factual_FP64_FP32_max_abs_CE_difference=float(decision['factual_FP32_CE_difference'].max()),
            FP32_comparison_scope='same virtual/factual views convertedFP32/default formula; no native model call',
            view_and_readout_dtype='float64',returned_loss_dtype=str(joint.dtype),
            scale=SCALE,eps=EPS,stop_probability=None,stop_rule=STOP_RULE,tolerance=TOLERANCE,
            coefficient=COEFFICIENT,factual_threshold_gradient_stopped=True,
            factual_source='actual current visible-L destination packet of same true TX',
            label_application_consistent=True,by_anchor_RX_semantics='destination RX receiving the individual reference_delta',
            class_weight_gradient_stopped=True,joint_all_three_endpoints_attached=True,
            actual_three_gains_follow_joint_graph=True,new_head_fit_updates=0,persistent_statistics=0,
            common_RX_additive_offset_loss_invariance_claimed=False,
            unique_matrix_disclosure='SAME_RX15 unique repeated4' if mode=='SAME_RX' else 'CROSS_RX60 unique',
            IQ_forward_count=0)
    values['ZERO'] = joint.sum()*0.
    summaries['ZERO'] = dict(copy.deepcopy(summaries['CROSS_RX']),mode='ZERO',loss=0.,
        diagnostic_reference='CROSS_RX',connected_zero_graph=True)
    for summary in summaries.values():
        summary.update(all_controls_computed=True,default_class_decisions_this_call=14400,
            virtual_default_class_decisions_this_call=7200,factual_default_class_decisions_this_call=7200,
            FP64_formula_decisions_this_call=14400,FP32_comparison_formula_decisions_this_call=14400,
            computed_controls=list(MODES),active_origin_coverage_unit='distinct physical origin among180, not repeated views')
    return {mode:values[mode] for mode in MODES},{mode:summaries[mode] for mode in MODES}
