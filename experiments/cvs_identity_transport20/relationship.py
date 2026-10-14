"""True-L-only balanced TX matching, with fixed and equal endpoint exposure."""
from collections import Counter
from collections.abc import Mapping
import copy
import math

import torch

from .matching import match


SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
MODES = ('ZERO', 'SAME_BI', 'CROSS_ROW', 'CROSS_BI')
_BANK_KEYS = {'x', 'y', 'rx', 'day', 'side', 'ids'}
_ROLE_HINTS = {'hide', 'role', 'source_role', 'tx_label_visible'}


def _bank(joint, bank, source_role):
    # Authority checks precede any read of true TX labels or their values.
    if source_role != 'L_s':
        raise ValueError('Only visible source L_s can define TX matching')
    if not isinstance(bank, Mapping):
        raise ValueError('Original visible source L180 bank required')
    if bank.get('hide', False) is not False or bank.get('tx_label_visible', True) is not True:
        raise ValueError('Hidden U labels cannot enter relationship grouping')
    if any(bank.get(key, 'L_s') != 'L_s' for key in ('role', 'source_role')):
        raise ValueError('U/V/target roles cannot define this source relationship')
    if set(bank) - _ROLE_HINTS != _BANK_KEYS:
        raise ValueError('The original x/y/rx/day/side/ids bank is required')
    if (not isinstance(joint, torch.Tensor) or joint.layout != torch.strided
            or joint.shape != (180, 160) or joint.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(joint).all())):
        raise ValueError('Finite real attached joint[180,160] in FP32/FP64 required')
    x = bank['x']
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.shape != (180, 2, 256)
            or x.dtype not in (torch.float32, torch.float64) or x.device != joint.device
            or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite source L180 IQ bank shape/device differs')
    y, rx, day, side = (bank[key] for key in ('y', 'rx', 'day', 'side'))
    for name, value in (('y', y), ('rx', rx), ('day', day), ('side', side)):
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != (180,) or value.dtype != torch.long or value.device != joint.device):
            raise ValueError('Legal source L integer vector differs: ' + name)
    if (not bool(((y >= 0) & (y < 6)).all()) or set(rx.tolist()) != set(SOURCE_RX)
            or set(day.tolist()) != set(SOURCE_DAYS) or set(side.tolist()) != {0, 1}):
        raise ValueError('Six true source TX, five source RX, three days and two sides required')
    identities = bank['ids']
    if not isinstance(identities, (list, tuple)) or len(identities) != 180:
        raise ValueError('One complete native physical identity per source packet required')
    physical, base_indices = set(), set()
    for index, identity in enumerate(identities):
        if (not isinstance(identity, (tuple, list)) or len(identity) != 5
                or any(type(value) is not int or value < 0 for value in identity)):
            raise ValueError('Physical IDs require five nonnegative native integers')
        identity = tuple(identity)
        if identity in physical or identity[4] in base_indices:
            raise ValueError('Duplicate L physical identity/base_index')
        if identity[1] != int(rx[index]) or identity[2] != int(day[index]):
            raise ValueError('Physical RX/day differs from bank grouping metadata')
        physical.add(identity); base_indices.add(identity[4])
    cells = Counter(zip(rx.tolist(), day.tolist(), side.tolist(), y.tolist()))
    expected = {(receiver, source_day, arm, label): 1 for receiver in SOURCE_RX
                for source_day in SOURCE_DAYS for arm in (0, 1) for label in range(6)}
    if cells != expected:
        raise ValueError('Exactly30 source groups of6 true TX, 36packets/RX, required')
    groups = {(receiver, source_day, arm): torch.tensor(
        sorted([index for index in range(180) if int(rx[index]) == receiver
                and int(day[index]) == source_day and int(side[index]) == arm],
               key=lambda index: int(y[index])), device=joint.device, dtype=torch.long)
        for receiver in SOURCE_RX for source_day in SOURCE_DAYS for arm in (0, 1)}
    return y, rx, day, groups


def _moments(values):
    return dict(count=len(values), mean=math.fsum(values) / len(values), min=min(values), max=max(values))


def _summarize(mode, records, exposures, bank, value):
    summaries = [record['matching'] for record in records]
    row_mass = [float(mass) for summary in summaries for mass in summary['diagmass']['row']]
    column_mass = [float(mass) for summary in summaries for mass in summary['diagmass']['column']]
    actual_pairs = [dict(anchor_RX=record['anchor_RX'], donor_RX=record['donor_RX'],
                         day=record['day'], repeat=record['repeat']) for record in records]
    unique = {(record['anchor_RX'], record['donor_RX'], record['day']) for record in records}
    pairs = {(record['anchor_RX'], record['donor_RX']) for record in records}
    by_rx = {str(receiver): dict(anchor=sum(exposures['anchor'][index] for index in range(180) if int(bank['rx'][index]) == receiver),
                                donor=sum(exposures['donor'][index] for index in range(180) if int(bank['rx'][index]) == receiver))
             for receiver in SOURCE_RX}
    by_day = {str(day): dict(anchor=sum(exposures['anchor'][index] for index in range(180) if int(bank['day'][index]) == day),
                            donor=sum(exposures['donor'][index] for index in range(180) if int(bank['day'][index]) == day))
              for day in SOURCE_DAYS}
    by_tx = {str(label): dict(anchor=sum(exposures['anchor'][index] for index in range(180) if int(bank['y'][index]) == label),
                             donor=sum(exposures['donor'][index] for index in range(180) if int(bank['y'][index]) == label))
             for label in range(6)}
    all_exposures = [anchor + donor for anchor, donor in zip(exposures['anchor'], exposures['donor'])]
    marginal = [summary['marginal_residual_max'] for summary in summaries if summary['marginal_residual_max'] is not None]
    return dict(mode=mode, loss=float(value.detach()), source_role='L_s',
        matrix_count=len(records), fixed_matrix_denominator=60, unique_matrix_count=len(unique),
        ordered_RX_pair_count=len(pairs), pairwise_cost_entries=36 * len(records),
        source_group_count=30, packets_per_RX=36, diagonal_row_count=6 * len(records),
        diagonal_column_count=6 * len(records),
        active_row_count=sum(summary['lowmass']['row'] for summary in summaries),
        active_column_count=sum(summary['lowmass']['column'] for summary in summaries),
        active_rows_and_columns=sum(summary['lowmass']['row'] + summary['lowmass']['column'] for summary in summaries),
        row_argmax_error_count=sum(summary['argmaxerrors']['row'] for summary in summaries),
        column_argmax_error_count=sum(summary['argmaxerrors']['column'] for summary in summaries),
        matching_errors_total=sum(summary['argmaxerrors']['row'] + summary['argmaxerrors']['column'] for summary in summaries),
        correct_row_mass=_moments(row_mass), correct_column_mass=_moments(column_mass),
        mean_correct_matching_mass=(math.fsum(row_mass) + math.fsum(column_mass)) / 720,
        marginal_residual_max=max(marginal) if marginal else None,
        max_marginal_residual=max(marginal) if marginal else None,
        approximate_plan_count=sum(summary['approximate_plan'] is True for summary in summaries),
        approximate_plan_matrix_count=sum(summary['approximate_plan'] is True for summary in summaries),
        row_control_column_marginal_residual_max=max(
            (summary.get('row_control_column_marginal_residual', 0.) or 0.) for summary in summaries),
        column_control_row_marginal_residual_max=max(
            (summary.get('column_control_row_marginal_residual', 0.) or 0.) for summary in summaries),
        cost_RMS=_moments([float(summary['costRMS']) for summary in summaries]),
        raw_D_norm=_moments([float(summary['rawDnorm']) for summary in summaries]),
        RMS_floor_active_matrix_count=sum(bool(summary['floor_active']) for summary in summaries),
        actual_pairs=actual_pairs, matrix_diagnostics=records,
        exposures=dict(unique_packets=180, anchor_endpoints=sum(exposures['anchor']),
            donor_endpoints=sum(exposures['donor']), total_endpoint_appearances=sum(all_exposures),
            per_packet_min=min(all_exposures), per_packet_max=max(all_exposures),
            by_RX=by_rx, by_day=by_day, by_TX=by_tx, by_side={'0':360, '1':360}),
        feature='actual fourterm raw joint160', class_sorting='canonical true label matching only',
        W_or_new_head_used=False, IQ_forward_count=0,
        loss_normalization='fixed60 matrices; matching weight.1 and row/column mean averaged',
        unique_matrix_disclosure='SAME has15 unique matrices repeated4' if mode == 'SAME_BI' else 'CROSS has60 unique matrices')


def losses(joint, bank, source_role='L_s'):
    """Compute all controls from the same L features, without a classifier.

    Each sample appears exactly4 times on its own side in every active control.
    SAME really executes its15 matrices4 times, matching CROSS's60 operations.
    ZERO performs the full call's matching diagnostics and returns connected0.
    Returned summaries contain scalar L diagnostics, never raw physical IDs.
    """
    y, rx, day, groups = _bank(joint, bank, source_role)
    values, summaries = {}, {}
    for mode in ('SAME_BI', 'CROSS_ROW', 'CROSS_BI'):
        work = [(receiver, receiver, source_day, repeat) for repeat in range(4)
                for receiver in SOURCE_RX for source_day in SOURCE_DAYS] if mode == 'SAME_BI' else [
                    (anchor, donor, source_day, 0) for anchor in SOURCE_RX for donor in SOURCE_RX
                    if anchor != donor for source_day in SOURCE_DAYS]
        matrix_losses, records = [], []
        exposure = dict(anchor=[0] * 180, donor=[0] * 180)
        for anchor, donor, source_day, repeat in work:
            indices_a, indices_b = groups[(anchor, source_day, 0)], groups[(donor, source_day, 1)]
            a, b = joint.index_select(0, indices_a), joint.index_select(0, indices_b)
            value, summary = match(a, b, 'ROW' if mode == 'CROSS_ROW' else 'BI',
                                   y.index_select(0, indices_a), y.index_select(0, indices_b))
            matrix_losses.append(value)
            records.append(dict(anchor_RX=anchor, donor_RX=donor, day=source_day, repeat=repeat, matching=summary))
            for index in indices_a.tolist(): exposure['anchor'][index] += 1
            for index in indices_b.tolist(): exposure['donor'][index] += 1
        if len(matrix_losses) != 60 or any(a + b != 4 for a, b in zip(exposure['anchor'], exposure['donor'])):
            raise RuntimeError('Fixed60 matching population/endpoint exposure was not respected')
        values[mode] = torch.stack(matrix_losses).sum() / 60
        summaries[mode] = _summarize(mode, records, exposure, bank, values[mode])
    values['ZERO'] = joint.sum() * 0.
    summaries['ZERO'] = dict(copy.deepcopy(summaries['CROSS_BI']), mode='ZERO', loss=0., diagnostic_reference='CROSS_BI',
        connected_zero_graph=True, unique_matrix_disclosure='ZERO reuses completed CROSS_BI diagnostics')
    for summary in summaries.values():
        summary['matching_evaluations_this_call'] = 180
        summary['all_three_controls_computed'] = True
    return {mode: values[mode] for mode in MODES}, {mode: summaries[mode] for mode in MODES}
