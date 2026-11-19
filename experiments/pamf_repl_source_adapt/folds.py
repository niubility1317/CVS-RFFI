"""Preregister held-out source groups and their immutable role positions."""
import numpy as np


ROLES = ('L_s', 'U_s', 'V')


def fold_groups(receivers, days):
    receivers = sorted(set(receivers))
    days = sorted(set(days))
    if len(receivers) < 2 or len(days) < 2:
        return {}
    if len(receivers) >= 3:
        # Use the two ends of the existing RX order, without looking at results.
        return {f'rx_{rx}': tuple((rx, day) for day in days) for rx in (receivers[0], receivers[-1])}
    if len(days) < 3:
        return {}
    # Keep both RXs and every date represented during training. A one-RX
    # leave-out fit could not run the original cross-RX loss on a two-RX setting.
    return {f'cells_{i}': tuple((rx, days[(i + j) % len(days)]) for j, rx in enumerate(receivers))
            for i in range(2)}


def fold_positions(metadata, holdout):
    """Return retained L/U/V and held V indices; held L/U are never consumed."""
    if set(metadata) != set(ROLES) or not holdout:
        raise ValueError('Original source roles and a nonempty held-out cell set required')
    roles = {}
    group_positions = {}
    for role in ROLES:
        values = metadata[role]
        rx, day = values['rx_i'], values['day_i']
        if rx.ndim != 1 or day.shape != rx.shape or not len(rx):
            raise ValueError(f'Physical source axes missing for {role}')
        if role == 'U_s' and 'y' in values:
            raise ValueError('U truth is forbidden in fold construction')
        held = np.zeros(len(rx), dtype=bool)
        for receiver, date in holdout:
            mask = (rx == receiver) & (day == date)
            held |= mask
            if role == 'V':
                group_positions[(receiver, date)] = np.flatnonzero(mask)
        roles[role] = np.flatnonzero(~held)
    labels = metadata['L_s']['y']
    if labels.shape != metadata['L_s']['rx_i'].shape:
        raise ValueError('Source L labels must align with source IDs')
    if (len(roles['L_s']) < 128 or len(roles['U_s']) == 0 or len(roles['V']) == 0
            or set(labels[roles['L_s']]) != set(labels) or any(not len(idx) for idx in group_positions.values())):
        return None
    return dict(train=roles, held_v=group_positions)
