"""Frozen, no-fit source census from already decided pseudo-targets and masks."""
from collections.abc import Mapping, Sequence
import copy
import math

import torch


SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
VIEWS = ('SELF', 'PA_WEAK', 'REF_WEAK')
PI = .9
COEFFICIENT = .1
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_COUNTS = ('sample_count', 'accepted_count', 'SELF_active_count',
           'PA_WEAK_active_count', 'REF_WEAK_active_count',
           'EVIDENCE_active_count', 'new_active_vs_SELF_count')


def _vector(value, name, count=None, *, boolean=False, integer=False):
    if not isinstance(value, torch.Tensor):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f'{name} must be a detached one-dimensional vector')
        expected = bool if boolean else int if integer else None
        if expected is not None and any(type(item) is not expected for item in value):
            raise ValueError(f'{name} requires native {expected.__name__} values')
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f'{name} cannot be converted to a vector') from error
    if value.requires_grad or value.layout != torch.strided or value.ndim != 1:
        raise ValueError(f'{name} must be a detached dense vector')
    if count is not None and len(value) != count:
        raise ValueError(f'{name} batch shape differs')
    if boolean and value.dtype != torch.bool:
        raise ValueError(f'{name} must have bool dtype')
    if integer and value.dtype not in _INTEGER_DTYPES:
        raise ValueError(f'{name} must have integer dtype')
    if not boolean and not integer and not value.is_floating_point():
        raise ValueError(f'{name} must have floating dtype')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')
    return value.detach().to(device='cpu')


def _ids(value, count, rx, day):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != count:
        raise ValueError('physical_ids must contain one five-integer identity per packet')
    result = []
    for index, row in enumerate(value):
        if (not isinstance(row, (list, tuple)) or len(row) != 5
                or any(type(item) is not int or item < 0 for item in row)):
            raise ValueError('Physical identities require five nonnegative native integers')
        identity = tuple(row)
        # eq/sig/base are opaque identities. No TX field is accepted or read.
        if identity[1] != int(rx[index]) or identity[2] != int(day[index]):
            raise ValueError('Physical identity RX/day differs from census metadata')
        result.append(identity)
    if len(set(result)) != count or len({row[4] for row in result}) != count:
        raise ValueError('Duplicate full physical identity or base_index inside batch')
    return result


def _empty_stats():
    return dict(count=0, total=0., square_total=0., minimum=None, maximum=None)


def _stats_add(destination, values):
    if not len(values):
        return
    destination['count'] += len(values)
    destination['total'] += float(values.sum())
    destination['square_total'] += float(values.square().sum())
    minimum, maximum = float(values.min()), float(values.max())
    destination['minimum'] = minimum if destination['minimum'] is None else min(destination['minimum'], minimum)
    destination['maximum'] = maximum if destination['maximum'] is None else max(destination['maximum'], maximum)


def _stats_finish(value):
    count = value['count']
    mean = value['total'] / count if count else None
    variance = max(0., value['square_total'] / count - mean * mean) if count else None
    return dict(count=count, mean=mean, std=math.sqrt(variance) if count else None,
                min=value['minimum'], max=value['maximum'])


def _empty_row():
    result = {name: 0 for name in _COUNTS}
    result.update(accepted_correct_count=0, accepted_error_count=0,
                  EVIDENCE_active_correct_count=0, EVIDENCE_active_error_count=0,
                  stop_sum_by_view={view: 0. for view in VIEWS},
                  target_probability_stats={view: {subset: _empty_stats()
                      for subset in ('all', 'accepted', 'EVIDENCE_active')} for view in VIEWS},
                  predicted_class_counts={str(label): 0 for label in range(6)},
                  accepted_predicted_class_counts={str(label): 0 for label in range(6)},
                  EVIDENCE_active_predicted_class_counts={str(label): 0 for label in range(6)})
    return result


def _add_row(row, selected, targets, accepted, active, evidence, new, probabilities, stops, correct):
    row['sample_count'] += int(selected.sum())
    row['accepted_count'] += int((selected & accepted).sum())
    for view in VIEWS:
        row[view + '_active_count'] += int((selected & active[view]).sum())
        row['stop_sum_by_view'][view] += float(stops[view][selected & accepted].sum())
        for subset, mask in (('all', selected), ('accepted', selected & accepted),
                             ('EVIDENCE_active', selected & evidence)):
            _stats_add(row['target_probability_stats'][view][subset], probabilities[view][mask])
    row['EVIDENCE_active_count'] += int((selected & evidence).sum())
    row['new_active_vs_SELF_count'] += int((selected & new).sum())
    for label in range(6):
        predicted = selected & (targets == label)
        row['predicted_class_counts'][str(label)] += int(predicted.sum())
        row['accepted_predicted_class_counts'][str(label)] += int((predicted & accepted).sum())
        row['EVIDENCE_active_predicted_class_counts'][str(label)] += int((predicted & evidence).sum())
    if correct is not None:
        row['accepted_correct_count'] += int((selected & accepted & correct).sum())
        row['accepted_error_count'] += int((selected & accepted & ~correct).sum())
        row['EVIDENCE_active_correct_count'] += int((selected & evidence & correct).sum())
        row['EVIDENCE_active_error_count'] += int((selected & evidence & ~correct).sum())


def _finish_row(row, has_truth):
    result = copy.deepcopy(row)
    count, accepted, evidence = row['sample_count'], row['accepted_count'], row['EVIDENCE_active_count']
    result.update(full_denominator_count=count, view_denominator_count=2 * count,
                  SELF_STOP_loss=COEFFICIENT * row['stop_sum_by_view']['SELF'] / count if count else 0.,
                  EVIDENCE_STOP_loss=COEFFICIENT * (row['stop_sum_by_view']['PA_WEAK']
                      + row['stop_sum_by_view']['REF_WEAK']) / (2 * count) if count else 0.,
                  teacher_precision_among_q=row['accepted_correct_count'] / accepted if has_truth and accepted else None,
                  EVIDENCE_active_precision=row['EVIDENCE_active_correct_count'] / evidence if has_truth and evidence else None)
    if not has_truth:
        for key in ('accepted_correct_count', 'accepted_error_count',
                    'EVIDENCE_active_correct_count', 'EVIDENCE_active_error_count'):
            result[key] = None
    result['target_probability_stats'] = {view: {subset: _stats_finish(value)
        for subset, value in row['target_probability_stats'][view].items()} for view in VIEWS}
    return result


class Census:
    """Aggregate frozen decisions; source truth has no route to active decisions.

    Input probabilities are for the already fixed teacher pseudo-target, not
    challenge-view argmax labels. The caller's q is used without a second gate.
    IDs are retained only in private in-memory sets and never returned.
    """
    def __init__(self, role):
        if role not in ('V_proxy', 'U_s'):
            raise ValueError('Census role must be V_proxy or hidden U_s')
        self.role = role
        self._overall = _empty_row()
        self._by_rx = {receiver: _empty_row() for receiver in SOURCE_RX}
        self._strata = {(receiver, day): _empty_row() for receiver in SOURCE_RX for day in SOURCE_DAYS}
        self._physical_ids, self._base_indices = set(), set()
        self._truth_present = None
        self._batches = []
        self._finished = None

    def add(self, physical_ids, rx, day, targets, accept_mask, target_probabilities, source_truth=None):
        if self._finished is not None:
            raise ValueError('Finished census cannot accept more packets')
        if self.role == 'U_s' and source_truth is not None:
            raise ValueError('U_s truth is hidden and cannot enter census')
        targets = _vector(targets, 'targets', integer=True)
        count = len(targets)
        if count < 1 or bool(((targets < 0) | (targets > 5)).any()):
            raise ValueError('Nonempty teacher pseudo-targets must be classes 0..5')
        rx, day = _vector(rx, 'rx', count, integer=True), _vector(day, 'day', count, integer=True)
        if any(int(value) not in SOURCE_RX for value in rx) or any(int(value) not in SOURCE_DAYS for value in day):
            raise ValueError('Only source RX 1/3/4/6/8 and day 1/2/3 are permitted')
        accepted = _vector(accept_mask, 'accept_mask', count, boolean=True)
        ids = _ids(physical_ids, count, rx, day)
        if set(ids) & self._physical_ids or {row[4] for row in ids} & self._base_indices:
            raise ValueError('Physical packet or base_index was already counted')
        if not isinstance(target_probabilities, Mapping) or set(target_probabilities) != set(VIEWS):
            raise ValueError('Exactly SELF/PA_WEAK/REF_WEAK target probabilities required')
        probabilities = {view: _vector(target_probabilities[view], view, count).double() for view in VIEWS}
        if any(bool(((value <= 0) | (value > 1)).any()) for value in probabilities.values()):
            raise ValueError('Finite target probabilities must be in (0,1]; stopped log-loss is not clipped')
        # Complete decisions are fixed before optional source truth is consumed.
        active = {view: accepted & (probabilities[view] < PI) for view in VIEWS}
        evidence = active['PA_WEAK'] | active['REF_WEAK']
        new = evidence & ~active['SELF']
        stops = {view: (math.log(PI) - value.log()).clamp_min(0.) for view, value in probabilities.items()}
        has_truth = source_truth is not None
        if self._truth_present is not None and self._truth_present != has_truth:
            raise ValueError('Source truth coverage must be consistent across complete census')
        correct = None
        if has_truth:
            truth = _vector(source_truth, 'source_truth', count, integer=True)
            if bool(((truth < 0) | (truth > 5)).any()):
                raise ValueError('V_proxy truth must be source classes 0..5')
            correct = targets == truth
        # Validation above is atomic: rejected additions do not reserve IDs or counts.
        self._truth_present = has_truth
        self._physical_ids.update(ids)
        self._base_indices.update(row[4] for row in ids)
        arguments = (targets, accepted, active, evidence, new, probabilities, stops, correct)
        all_samples = torch.ones(count, dtype=torch.bool)
        _add_row(self._overall, all_samples, *arguments)
        for receiver in SOURCE_RX:
            _add_row(self._by_rx[receiver], rx == receiver, *arguments)
            for source_day in SOURCE_DAYS:
                _add_row(self._strata[(receiver, source_day)], (rx == receiver) & (day == source_day), *arguments)
        batch = _empty_row()
        _add_row(batch, all_samples, *arguments)
        # Only scalar batch summaries are retained, never per-packet probabilities.
        self._batches.append(dict(batch_index=len(self._batches), **{key: batch[key] for key in _COUNTS},
            full_denominator_count=count, view_denominator_count=2 * count,
            stop_sum_by_view=batch['stop_sum_by_view'],
            SELF_STOP_loss=COEFFICIENT * batch['stop_sum_by_view']['SELF'] / count,
            EVIDENCE_STOP_loss=COEFFICIENT * (batch['stop_sum_by_view']['PA_WEAK']
                + batch['stop_sum_by_view']['REF_WEAK']) / (2 * count)))

    def finish(self, expected_count):
        if type(expected_count) is not int or expected_count < 1:
            raise ValueError('Positive native integer expected_count required')
        if self._overall['sample_count'] != expected_count:
            raise ValueError('Complete census count differs from expected_count')
        if self._finished is None:
            has_truth = self.role == 'V_proxy' and self._truth_present is True
            summary = _finish_row(self._overall, has_truth)
            summary.update(batch_count=len(self._batches),
                SELF_STOP_batch_loss_sum=sum(row['SELF_STOP_loss'] for row in self._batches),
                EVIDENCE_STOP_batch_loss_sum=sum(row['EVIDENCE_STOP_loss'] for row in self._batches))
            self._finished = dict(status='COMPLETE', role=self.role, expected_count=expected_count,
                truth_status='HIDDEN' if self.role == 'U_s' else 'PROVIDED' if has_truth else 'NOT_PROVIDED',
                pi=PI, coefficient=COEFFICIENT, summary=summary,
                by_RX={str(receiver): dict(RX=receiver, **_finish_row(self._by_rx[receiver], has_truth))
                       for receiver in SOURCE_RX},
                strata=[dict(RX=receiver, day=day, **_finish_row(self._strata[(receiver, day)], has_truth))
                        for receiver in SOURCE_RX for day in SOURCE_DAYS],
                batches=copy.deepcopy(self._batches))
        return copy.deepcopy(self._finished)
