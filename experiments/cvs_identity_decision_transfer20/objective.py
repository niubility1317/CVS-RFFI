"""True-L default-head decisions after temporary balanced common-mean transfer."""
from collections import Counter
import copy
import math

import torch
from torch.nn import functional as F

from experiments.cvs_identity_transport20.relationship import _bank, SOURCE_RX, SOURCE_DAYS


MODES = ('ZERO', 'SAME_RX', 'CROSS_RX')
SCALE = 30.
EPS = 1e-4
STOP = .9
COEFFICIENT = .1


def _stats(values):
    values = values.detach().double()
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                RMS=float(values.square().mean().sqrt()))


def _decisions(views, labels, weight):
    logits = SCALE * F.linear(F.normalize(views, dim=1, eps=EPS),
                              F.normalize(weight.detach().double(), dim=1, eps=EPS))
    log_probabilities = F.log_softmax(logits, dim=1)
    log_true = log_probabilities.gather(1, labels[:, None]).squeeze(1)
    stop_terms = (math.log(STOP) - log_true).clamp_min(0.)
    with torch.no_grad():
        logits32 = SCALE * F.linear(F.normalize(views.detach().float(), dim=1, eps=EPS),
                                    F.normalize(weight.detach().float(), dim=1, eps=EPS))
        probability32 = logits32.softmax(1).gather(1, labels[:, None]).squeeze(1)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(logits32).all()):
        raise ValueError('Finite transported default-head FP64/FP32 formula required')
    return dict(logits=logits, probability=log_true.exp(), CE=-log_true, stop_terms=stop_terms,
                active=stop_terms.detach() > 0., errors=logits.detach().argmax(1) != labels,
                norm=views.norm(dim=1),
                FP32_logit_difference=(logits.detach() - logits32.double()).abs(),
                FP32_true_probability_difference=(log_true.detach().exp() - probability32.double()).abs())


def _row(decisions, selected):
    return dict(decision_count=int(selected.sum()),
        active_decisions=int(decisions['active'][selected].sum()),
        errors_total=int(decisions['errors'][selected].sum()),
        mean_true_probability=float(decisions['probability'][selected].detach().mean()),
        mean_CE=float(decisions['CE'][selected].detach().mean()),
        loss=COEFFICIENT * float(decisions['stop_terms'][selected].detach().mean()),
        transported_view_norm=_stats(decisions['norm'][selected]),
        near_eps_count=int((decisions['norm'][selected].detach() <= EPS).sum()))


def _exposures(endpoint, classification, bank):
    totals = [a + b for a, b in zip(endpoint['anchor'], endpoint['donor'])]
    return dict(unique_packets=180, anchor_endpoints=sum(endpoint['anchor']), donor_endpoints=sum(endpoint['donor']),
        total_endpoint_appearances=sum(totals), per_packet_min=min(totals), per_packet_max=max(totals),
        total_synthetic_classification_appearances=sum(classification),
        per_packet_classification_min=min(classification), per_packet_classification_max=max(classification),
        by_RX={str(receiver): dict(anchor=sum(endpoint['anchor'][i] for i in range(180) if int(bank['rx'][i]) == receiver),
                                  donor=sum(endpoint['donor'][i] for i in range(180) if int(bank['rx'][i]) == receiver),
                                  synthetic=sum(classification[i] for i in range(180) if int(bank['rx'][i]) == receiver))
               for receiver in SOURCE_RX},
        by_day={str(day): dict(anchor=sum(endpoint['anchor'][i] for i in range(180) if int(bank['day'][i]) == day),
                              donor=sum(endpoint['donor'][i] for i in range(180) if int(bank['day'][i]) == day),
                              synthetic=sum(classification[i] for i in range(180) if int(bank['day'][i]) == day))
                for day in SOURCE_DAYS},
        by_TX={str(label): dict(anchor=sum(endpoint['anchor'][i] for i in range(180) if int(bank['y'][i]) == label),
                               donor=sum(endpoint['donor'][i] for i in range(180) if int(bank['y'][i]) == label),
                               synthetic=sum(classification[i] for i in range(180) if int(bank['y'][i]) == label))
               for label in range(6)},
        by_side={str(side): dict(endpoint=sum(totals[i] for i in range(180) if int(bank['side'][i]) == side),
                                synthetic=sum(classification[i] for i in range(180) if int(bank['side'][i]) == side))
                 for side in (0, 1)})


def losses(joint, bank, weight, source_role='L_s'):
    """Compute both fixed720-decision controls and graph-connected ZERO.

    For each pair a(side0),b(side1), classify b-mean(b)+mean(a) with labels_b
    and a-mean(a)+mean(b) with labels_a. Both endpoints and their real deployed
    gains retain the joint graph; only original class W is stopped here.
    Means are call-local balanced six-TX averages, not fitted RX/prototype state.
    """
    labels, _, _, groups = _bank(joint, bank, source_role)
    if (not isinstance(weight, torch.Tensor) or weight.layout != torch.strided or weight.shape != (6, 160)
            or weight.dtype not in (torch.float32, torch.float64) or weight.device != joint.device
            or not bool(torch.isfinite(weight).all())):
        raise ValueError('Finite existing default W[6,160] on the joint device required')
    features = joint.double()
    values, summaries = {}, {}
    for mode in ('SAME_RX', 'CROSS_RX'):
        pairs = [(receiver, receiver, day, repeat) for repeat in range(4)
                 for receiver in SOURCE_RX for day in SOURCE_DAYS] if mode == 'SAME_RX' else [
                     (anchor, donor, day, 0) for anchor in SOURCE_RX for donor in SOURCE_RX
                     if anchor != donor for day in SOURCE_DAYS]
        views, origin, view_labels, common_RX, means, differences, records = [], [], [], [], [], [], []
        endpoint = dict(anchor=[0] * 180, donor=[0] * 180); classification = [0] * 180
        for anchor, donor, day, repeat in pairs:
            indices_a, indices_b = groups[(anchor, day, 0)], groups[(donor, day, 1)]
            a, b = features.index_select(0, indices_a), features.index_select(0, indices_b)
            mu_a, mu_b = a.mean(0), b.mean(0)
            b_to_a, a_to_b = b - mu_b + mu_a, a - mu_a + mu_b
            views.extend((b_to_a, a_to_b)); origin.extend((b, a))
            view_labels.extend((labels.index_select(0, indices_b), labels.index_select(0, indices_a)))
            common_RX.extend(([anchor] * 6, [donor] * 6)); means.extend((mu_a, mu_b))
            differences.extend((b_to_a - b, a_to_b - a))
            records.append(dict(anchor_RX=anchor, donor_RX=donor, day=day, repeat=repeat,
                directions=['side1_residual_plus_side0_mean', 'side0_residual_plus_side1_mean'],
                label_source_sides=[1, 0], receiving_common_mean_RX=[anchor, donor]))
            for index in indices_a.tolist(): endpoint['anchor'][index] += 1; classification[index] += 1
            for index in indices_b.tolist(): endpoint['donor'][index] += 1; classification[index] += 1
        if (len(pairs) != 60 or len(views) != 120 or any(a + b != 4 for a, b in zip(endpoint['anchor'], endpoint['donor']))
                or any(count != 4 for count in classification)):
            raise RuntimeError('Fixed60 pairs/120 views/720 decisions/4 exposures differs')
        transported = torch.cat(views); target = torch.cat(view_labels)
        common_receivers = torch.tensor([receiver for group in common_RX for receiver in group], device=joint.device)
        if transported.shape != (720, 160) or target.shape != (720,) or Counter(target.tolist()) != {label:120 for label in range(6)}:
            raise RuntimeError('Transported views lost origin-label alignment or fixed720 population')
        decision = _decisions(transported, target, weight)
        values[mode] = (COEFFICIENT * decision['stop_terms'].sum() / 720).to(dtype=joint.dtype)
        row = _row(decision, torch.ones(720, dtype=torch.bool, device=joint.device))
        unique = {(record['anchor_RX'], record['donor_RX'], record['day']) for record in records}
        summaries[mode] = dict(row, mode=mode, loss=float(values[mode].detach()), source_role='L_s',
            matrix_count=60, unique_matrices=len(unique), unique_matrix_count=len(unique),
            synthetic_views=120, synthetic_decisions=720, fixed_decision_denominator=720,
            by_anchor_RX={str(receiver): _row(decision, common_receivers == receiver) for receiver in SOURCE_RX},
            exposures=_exposures(endpoint, classification, bank), actual_pairs=records,
            common_means_RMS=float(torch.stack(means).detach().square().mean().sqrt()),
            transport_difference_RMS=float(torch.cat(differences).detach().square().mean().sqrt()),
            origin_joint_RMS=float(torch.cat(origin).detach().square().mean().sqrt()),
            default_cos_FP64_FP32_max_abs_logit_difference=float(decision['FP32_logit_difference'].max()),
            default_cos_FP64_FP32_max_abs_true_probability_difference=float(decision['FP32_true_probability_difference'].max()),
            FP32_comparison_scope='same transported view convertedFP32/default formula; no native model call',
            view_and_readout_dtype='float64', returned_loss_dtype=str(joint.dtype),
            scale=SCALE, eps=EPS, stop=STOP, coefficient=COEFFICIENT,
            label_application_consistent=True, by_anchor_RX_semantics='RX supplying the receiving common mean, both directions',
            class_weight_gradient_stopped=True, joint_both_endpoints_attached=True,
            actual_three_gains_follow_joint_graph=True, new_head_fit_updates=0, persistent_statistics=0,
            common_mean_semantics='temporary balanced sixTX group mean; no pureRX-factor claim',
            common_RX_additive_offset_loss_invariance_claimed=False,
            positive_global_scale_invariance_boundary='cosine only when feature/weight epsilon floors are not activated; finite precision remains',
            weight_rows_near_eps_count=int((weight.detach().double().norm(dim=1) <= EPS).sum()),
            unique_matrix_disclosure='SAME_RX15 unique repeated4' if mode == 'SAME_RX' else 'CROSS_RX60 unique',
            IQ_forward_count=0)
    values['ZERO'] = joint.sum() * 0.
    summaries['ZERO'] = dict(copy.deepcopy(summaries['CROSS_RX']), mode='ZERO', loss=0.,
        diagnostic_reference='CROSS_RX', connected_zero_graph=True)
    for summary in summaries.values():
        summary.update(all_controls_computed=True, transported_views_this_call=240,
                       default_class_decisions_this_call=1440, computed_controls=list(MODES))
    return {mode: values[mode] for mode in MODES}, {mode: summaries[mode] for mode in MODES}
