"""Coverage summaries accept only legally visible labels, never infer U_s TX."""
from collections import Counter
import math


def _association(left, right):
    pairs = Counter(zip(map(str, left), map(str, right)))
    a, b = Counter(map(str, left)), Counter(map(str, right))
    n = len(left)
    mi = sum(count / n * math.log(count * n / (a[x] * b[y])) for (x, y), count in pairs.items()) if n else 0.
    return dict(counts={f'{x}|{y}': count for (x, y), count in pairs.items()}, mutual_information_nats=mi)


def coverage_audit(rx, *, tx=None, role='L_s', day=None, snr=None, view=None, grouping_kind=None):
    """TX may be supplied only for L_s or read-only V; reject U_s TX input."""
    if role not in ('L_s', 'U_s', 'V'):
        raise ValueError('only legal source roles allowed')
    if tx is not None and role == 'U_s':
        raise ValueError('U_s hidden TX labels forbidden')
    rx = list(rx)
    for name, values in (('tx', tx), ('day', day), ('snr', snr), ('view', view)):
        if values is not None and len(values) != len(rx):
            raise ValueError(name + ' length differs from RX')
    result = dict(role=role, samples=len(rx), rx_counts=dict(Counter(map(str, rx))), grouping_kind=grouping_kind,
                  tx_rx='N/A', day_rx='N/A', snr_rx='N/A', view_rx='N/A')
    for name, values in (('tx', tx), ('day', day), ('snr', snr), ('view', view)):
        if values is not None:
            values = list(values)
            if name == 'tx' and any(v is None or int(v) < 0 for v in values):
                raise ValueError('unknown or hidden TX in labeled coverage')
            result[name + '_rx'] = _association(values, rx)
    if tx is not None:
        cells = len(set(zip(map(str, tx), map(str, rx))))
        possible = len(set(map(str, tx))) * len(set(map(str, rx)))
        result['tx_rx'].update(observed_cells=cells, possible_cells=possible, coverage=cells / possible if possible else 0.)
    return result
