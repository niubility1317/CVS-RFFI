"""Default-head true-L decisions using each individual different-TX RX delta."""
import copy

import torch

from experiments.cvs_identity_transport20.relationship import _bank, SOURCE_RX, SOURCE_DAYS
from experiments.cvs_identity_decision_transfer20.objective import _decisions, _row, _stats, SCALE, EPS, STOP, COEFFICIENT


MODES = ('ZERO', 'SAME_RX', 'CROSS_RX')


def _exposure(endpoint, roles, bank):
    totals = [a + b for a, b in zip(endpoint['anchor'], endpoint['donor'])]
    triples = [sum(roles[role][index] for role in roles) for index in range(180)]
    def group(indices):
        return dict(anchor=sum(endpoint['anchor'][index] for index in indices),
                    donor=sum(endpoint['donor'][index] for index in indices),
                    **{role:sum(values[index] for index in indices) for role, values in roles.items()})
    return dict(unique_packets=180, anchor_endpoints=sum(endpoint['anchor']), donor_endpoints=sum(endpoint['donor']),
        total_endpoint_appearances=sum(totals), per_packet_min=min(totals), per_packet_max=max(totals),
        total_triple_role_appearances=sum(triples), per_packet_triple_role_min=min(triples), per_packet_triple_role_max=max(triples),
        **{role+'_appearances':sum(values) for role, values in roles.items()},
        **{'per_packet_'+role+'_min':min(values) for role, values in roles.items()},
        **{'per_packet_'+role+'_max':max(values) for role, values in roles.items()},
        by_RX={str(receiver):group([index for index in range(180) if int(bank['rx'][index]) == receiver]) for receiver in SOURCE_RX},
        by_day={str(day):group([index for index in range(180) if int(bank['day'][index]) == day]) for day in SOURCE_DAYS},
        by_TX={str(label):group([index for index in range(180) if int(bank['y'][index]) == label]) for label in range(6)},
        by_side={str(side):group([index for index in range(180) if int(bank['side'][index]) == side]) for side in (0,1)})


def _coverage_row(decision, selected, origin_indices):
    row = _row(decision, selected)
    row.update(origin_packet_count=int(torch.unique(origin_indices[selected]).numel()),
               active_origin_packet_count=int(torch.unique(origin_indices[selected & decision['active']]).numel()))
    return row


def losses(joint, bank, weight, source_role='L_s'):
    """Fixed3600 decisions per control, with all five c!=y references.

    a_y-a_c+b_c is labeled by a_y and uses b's RX as destination; its reverse
    b_y-b_c+a_c is labeled by b_y and uses a's RX. Every view retains all three
    joint endpoints. Only original class W is detached by the shared readout.
    This is a temporary decision task, with no mean/prototype or fitted state.
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
        pairs = [(receiver, receiver, day, repeat) for repeat in range(4)
                 for receiver in SOURCE_RX for day in SOURCE_DAYS] if mode == 'SAME_RX' else [
                     (anchor, donor, day, 0) for anchor in SOURCE_RX for donor in SOURCE_RX
                     if anchor != donor for day in SOURCE_DAYS]
        views, origin_features, destinations, targets, reference_labels = [], [], [], [], []
        origin_indices, destination_RX, records = [], [], []
        endpoint = dict(anchor=[0]*180, donor=[0]*180)
        roles = {name:[0]*180 for name in ('origin', 'reference_source', 'reference_destination')}
        for anchor, donor, day, repeat in pairs:
            ia, ib = groups[(anchor,day,0)], groups[(donor,day,1)]
            a, b = features.index_select(0,ia), features.index_select(0,ib)
            ay, ac = a.index_select(0,target_rows), a.index_select(0,reference_rows)
            by, bc = b.index_select(0,target_rows), b.index_select(0,reference_rows)
            forward, reverse = ay-ac+bc, by-bc+ac
            views.extend((forward,reverse)); origin_features.extend((ay,by)); destinations.extend((by,ay))
            ya, yb = labels.index_select(0,ia), labels.index_select(0,ib)
            targets.extend((ya.index_select(0,target_rows),yb.index_select(0,target_rows)))
            reference_labels.extend((ya.index_select(0,reference_rows),yb.index_select(0,reference_rows)))
            origin_indices.extend((ia.index_select(0,target_rows),ib.index_select(0,target_rows)))
            destination_RX.extend(([donor]*30,[anchor]*30))
            records.append(dict(anchor_RX=anchor,donor_RX=donor,day=day,repeat=repeat,
                directions=['a_y-a_c+b_c','b_y-b_c+a_c'],label_source_sides=[0,1],
                destination_RX=[donor,anchor],references_per_origin=5,decisions_per_direction=30))
            for index in ia.tolist(): endpoint['anchor'][index]+=1
            for index in ib.tolist(): endpoint['donor'][index]+=1
            for origin, source_ref, destination_ref in (
                    (ia.index_select(0,target_rows),ia.index_select(0,reference_rows),ib.index_select(0,reference_rows)),
                    (ib.index_select(0,target_rows),ib.index_select(0,reference_rows),ia.index_select(0,reference_rows))):
                for role, indices in zip(roles,(origin,source_ref,destination_ref)):
                    for index in indices.tolist(): roles[role][index]+=1
        if (len(pairs)!=60 or len(views)!=120 or any(a+b!=4 for a,b in zip(endpoint['anchor'],endpoint['donor']))
                or any(count!=20 for role in roles.values() for count in role)):
            raise RuntimeError('Fixed60 pairs/all5 references/20 each triple-role exposure differs')
        transported,target,reference = torch.cat(views),torch.cat(targets),torch.cat(reference_labels)
        origins = torch.cat(origin_indices)
        receiving_RX = torch.tensor([receiver for group in destination_RX for receiver in group],device=joint.device)
        if transported.shape!=(3600,160) or target.shape!=(3600,) or bool((target==reference).any()):
            raise RuntimeError('Exactly3600 origin-labeled decisions with y!=c required')
        decision = _decisions(transported,target,weight)
        values[mode] = (COEFFICIENT*decision['stop_terms'].sum()/3600).to(dtype=joint.dtype)
        row = _coverage_row(decision,torch.ones(3600,dtype=torch.bool,device=joint.device),origins)
        difference = transported-torch.cat(origin_features)
        destination_error = transported-torch.cat(destinations)
        unique={(record['anchor_RX'],record['donor_RX'],record['day']) for record in records}
        summaries[mode]=dict(row,mode=mode,loss=float(values[mode].detach()),source_role='L_s',
            active_origin_fraction=row['active_origin_packet_count']/180,
            matrix_count=60,unique_matrices=len(unique),unique_matrix_count=len(unique),
            synthetic_views=120,synthetic_decisions=3600,fixed_decision_denominator=3600,
            references_per_origin=5,target_reference_pairs_per_direction=30,
            by_anchor_RX={str(receiver):_coverage_row(decision,receiving_RX==receiver,origins) for receiver in SOURCE_RX},
            by_reference_TX={str(label):_coverage_row(decision,reference==label,origins) for label in range(6)},
            by_target_TX={str(label):_coverage_row(decision,target==label,origins) for label in range(6)},
            exposures=_exposure(endpoint,roles,bank),actual_pairs=records,
            transfer_difference_RMS=float(difference.detach().square().mean().sqrt()),
            transfer_difference_norm=_stats(difference.norm(dim=1)),
            to_actual_destination_identity_RMS=float(destination_error.detach().square().mean().sqrt()),
            to_actual_destination_identity_norm=_stats(destination_error.norm(dim=1)),
            destination_identity_comparison='diagnostic only; no destination feature matching loss',
            default_cos_FP64_FP32_max_abs_logit_difference=float(decision['FP32_logit_difference'].max()),
            default_cos_FP64_FP32_max_abs_true_probability_difference=float(decision['FP32_true_probability_difference'].max()),
            FP32_comparison_scope='same transported view convertedFP32/default formula; no native model call',
            view_and_readout_dtype='float64',returned_loss_dtype=str(joint.dtype),
            scale=SCALE,eps=EPS,stop=STOP,coefficient=COEFFICIENT,
            label_application_consistent=True,by_anchor_RX_semantics='destination RX receiving the individual reference_delta',
            class_weight_gradient_stopped=True,joint_all_three_endpoints_attached=True,
            actual_three_gains_follow_joint_graph=True,new_head_fit_updates=0,persistent_statistics=0,
            common_RX_additive_offset_loss_invariance_claimed=False,
            unique_matrix_disclosure='SAME_RX15 unique repeated4' if mode=='SAME_RX' else 'CROSS_RX60 unique',
            IQ_forward_count=0)
    values['ZERO']=joint.sum()*0.
    summaries['ZERO']=dict(copy.deepcopy(summaries['CROSS_RX']),mode='ZERO',loss=0.,diagnostic_reference='CROSS_RX',connected_zero_graph=True)
    for summary in summaries.values():
        summary.update(all_controls_computed=True,default_class_decisions_this_call=7200,
                       computed_controls=list(MODES),active_origin_coverage_unit='distinct physical origin among180, not repeated views')
    return {mode:values[mode] for mode in MODES},{mode:summaries[mode] for mode in MODES}
