"""Independent ROW-only identity correspondence on balanced visible source L."""
import copy

import torch

from experiments.cvs_identity_transport20.matching import match
from experiments.cvs_identity_transport20.relationship import _bank, _summarize, SOURCE_RX, SOURCE_DAYS


MODES = ('ZERO', 'SAME_ROW', 'CROSS_ROW')


def losses(joint, bank, source_role='L_s'):
    """Compute exactly120 ROW matches, then return three control objectives.

    SAME_ROW has15 unique same-RX/day side0-to-side1 matrices computed4 times.
    CROSS_ROW has60 unique ordered different-RX/day side0-to-side1 matrices.
    Each original packet has4 endpoint appearances in either control. ZERO
    retains the same diagnostics and connects exact0 to the actual joint graph.
    No balanced plan, Sinkhorn iteration, classifier or new parameter is used.
    """
    labels, _, _, groups = _bank(joint, bank, source_role)
    values, summaries = {}, {}
    for mode in ('SAME_ROW', 'CROSS_ROW'):
        work = [(receiver, receiver, day, repeat) for repeat in range(4)
                for receiver in SOURCE_RX for day in SOURCE_DAYS] if mode == 'SAME_ROW' else [
                    (anchor, donor, day, 0) for anchor in SOURCE_RX for donor in SOURCE_RX
                    if anchor != donor for day in SOURCE_DAYS]
        matrix_losses, records = [], []
        exposure = dict(anchor=[0] * 180, donor=[0] * 180)
        for anchor, donor, day, repeat in work:
            indices_a, indices_b = groups[(anchor, day, 0)], groups[(donor, day, 1)]
            value, summary = match(joint.index_select(0, indices_a), joint.index_select(0, indices_b),
                'ROW', labels.index_select(0, indices_a), labels.index_select(0, indices_b))
            matrix_losses.append(value)
            records.append(dict(anchor_RX=anchor, donor_RX=donor, day=day, repeat=repeat, matching=summary))
            for index in indices_a.tolist(): exposure['anchor'][index] += 1
            for index in indices_b.tolist(): exposure['donor'][index] += 1
        if len(matrix_losses) != 60 or any(a + b != 4 for a, b in zip(exposure['anchor'], exposure['donor'])):
            raise RuntimeError('ROW-only fixed60 population/endpoint exposure differs')
        values[mode] = torch.stack(matrix_losses).sum() / 60
        summaries[mode] = _summarize(mode, records, exposure, bank, values[mode])
        summaries[mode]['unique_matrix_disclosure'] = ('SAME_ROW has15 unique matrices repeated4'
            if mode == 'SAME_ROW' else 'CROSS_ROW has60 unique matrices')
    values['ZERO'] = joint.sum() * 0.
    summaries['ZERO'] = dict(copy.deepcopy(summaries['CROSS_ROW']), mode='ZERO', loss=0.,
        diagnostic_reference='CROSS_ROW', connected_zero_graph=True,
        unique_matrix_disclosure='ZERO reuses completed CROSS_ROW diagnostics')
    for summary in summaries.values():
        summary.update(matching_evaluations_this_call=120, ROW_matching_evaluations=120,
            BI_matching_evaluations=0, sinkhorn_iterations_executed=0,
            all_two_ROW_controls_computed=True, all_three_controls_computed=True,
            computed_controls=list(MODES), matching_algorithm='independent ROW and column softmax')
    return {mode: values[mode] for mode in MODES}, {mode: summaries[mode] for mode in MODES}
