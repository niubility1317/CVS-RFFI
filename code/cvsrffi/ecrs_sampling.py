"""Resumable source samplers. The U interface deliberately has no TX field."""
from __future__ import annotations

import copy
import numpy as np


def _integer_array(values, name):
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    result = np.asarray(values).reshape(-1)
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f'{name} must contain integer identities')
    return result.astype(np.int64, copy=True)


class RXDayBalancedQueue:
    """Continuous no-replacement U cycles, round-robin over RX/day cells.

    Exhausted cells are skipped until every sample has been drawn once. This
    preserves complete unique coverage for unequal cells; exact cell balance
    is possible only while each cell still has samples. Epochs do not reset it.
    """
    def __init__(self, rx, day, seed=0):
        self.rx, self.day = _integer_array(rx, 'rx'), _integer_array(day, 'day')
        if len(self.rx) != len(self.day) or not len(self.rx):
            raise ValueError('U RX/day must have equal nonzero lengths')
        if np.any(self.rx < 0) or np.any(self.day < 0):
            raise ValueError('U RX/day identities must be nonnegative')
        self.rng = np.random.default_rng(seed)
        self.cells = sorted(set(zip(self.rx.tolist(), self.day.tolist())))
        self.members = [np.flatnonzero((self.rx == rx_id) & (self.day == day_id)) for rx_id, day_id in self.cells]
        self.cycle = 0
        self.draws = 0
        self.seen = set()
        self.cursor = 0
        self._reset_cycle()

    def _reset_cycle(self):
        self.queues = [self.rng.permutation(indices).tolist() for indices in self.members]
        self.positions = [0] * len(self.cells)

    def take(self, n):
        if int(n) != n or n < 0:
            raise ValueError('U sample count must be a nonnegative integer')
        result = []
        while len(result) < n:
            if all(p == len(q) for p, q in zip(self.positions, self.queues)):
                self.cycle += 1
                self._reset_cycle()
            i = self.cursor
            self.cursor = (i + 1) % len(self.cells)
            if self.positions[i] == len(self.queues[i]):
                continue
            index = self.queues[i][self.positions[i]]
            self.positions[i] += 1
            self.draws += 1
            self.seen.add(index)
            result.append(index)
        return result

    @property
    def coverage(self):
        return {'unique_count': len(self.seen), 'population': len(self.rx),
                'unique_fraction': len(self.seen) / len(self.rx),
                'draw_count': self.draws, 'completed_cycles': self.draws // len(self.rx),
                'cycle_unique_count': sum(self.positions)}

    def state_dict(self):
        return copy.deepcopy({'rx': self.rx, 'day': self.day,
                              'rng': self.rng.bit_generator.state,
                              'queues': self.queues, 'positions': self.positions,
                              'cursor': self.cursor, 'cycle': self.cycle,
                              'draws': self.draws, 'seen': sorted(self.seen)})

    def load_state_dict(self, state):
        if not np.array_equal(state['rx'], self.rx) or not np.array_equal(state['day'], self.day):
            raise ValueError('U queue dataset identity/order mismatch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.queues = copy.deepcopy(state['queues'])
        self.positions = list(state['positions'])
        self.cursor, self.cycle, self.draws = int(state['cursor']), int(state['cycle']), int(state['draws'])
        self.seen = set(state['seen'])


class MatchedLabeledSampler:
    """Candidate 6 TX x all source RX x 4 packets, day matched within RX.

    Missing cells and small cells shorten the batch and are counted, never
    filled with another RX/day/TX or duplicate physical samples. Only L labels
    are supplied; optional label_mask excludes any other rows before grouping.
    """
    def __init__(self, labels, rx, day, tx_per_batch=6, packets_per_cell=4,
                 seed=0, label_mask=None):
        self.labels = _integer_array(labels, 'labels')
        self.rx, self.day = _integer_array(rx, 'rx'), _integer_array(day, 'day')
        if not len(self.labels) == len(self.rx) == len(self.day):
            raise ValueError('L metadata lengths differ')
        if int(tx_per_batch) != tx_per_batch or int(packets_per_cell) != packets_per_cell or min(tx_per_batch, packets_per_cell) < 1:
            raise ValueError('L batch dimensions must be positive integers')
        valid = self.labels >= 0
        if label_mask is not None:
            if hasattr(label_mask, 'detach'):
                label_mask = label_mask.detach().cpu().numpy()
            mask = np.asarray(label_mask, dtype=bool).reshape(-1)
            if len(mask) != len(valid):
                raise ValueError('L label mask length differs')
            valid &= mask
        if not valid.any() or np.any(self.rx[valid] < 0) or np.any(self.day[valid] < 0):
            raise ValueError('L sampler needs valid nonnegative labeled identities')
        self.valid = valid
        self.tx_per_batch, self.packets_per_cell = int(tx_per_batch), int(packets_per_cell)
        self.rng = np.random.default_rng(seed)
        self.txs = sorted(set(self.labels[valid].tolist()))
        self.receivers = sorted(set(self.rx[valid].tolist()))
        self.days = {r: sorted(set(self.day[valid & (self.rx == r)].tolist())) for r in self.receivers}
        self.members = {}
        for index in np.flatnonzero(valid):
            key = (int(self.labels[index]), int(self.rx[index]), int(self.day[index]))
            self.members.setdefault(key, []).append(int(index))
        self.queues = {key: self.rng.permutation(value).tolist() for key, value in self.members.items()}
        self.positions = {key: 0 for key in self.members}
        self.batch_index = 0
        self.tx_queue = self.rng.permutation(self.txs).tolist()
        self.tx_position = 0

    def _take_txs(self):
        selected = []
        while len(selected) < min(self.tx_per_batch, len(self.txs)):
            if self.tx_position == len(self.tx_queue):
                self.tx_queue = self.rng.permutation(self.txs).tolist()
                self.tx_position = 0
            tx = self.tx_queue[self.tx_position]
            self.tx_position += 1
            if tx not in selected:
                selected.append(tx)
        return selected

    def _cell_take(self, key, count):
        selected = []
        while len(selected) < min(count, len(self.members[key])):
            if self.positions[key] == len(self.queues[key]):
                self.queues[key] = self.rng.permutation(self.members[key]).tolist()
                self.positions[key] = 0
            index = self.queues[key][self.positions[key]]
            self.positions[key] += 1
            if index not in selected:
                selected.append(index)
        return selected

    def take_batch(self):
        txs = self._take_txs()
        result, missing, short = [], 0, 0
        cell_counts = []
        for rx in self.receivers:
            day = self.days[rx][self.batch_index % len(self.days[rx])]
            for tx in txs:
                key = (tx, rx, day)
                indices = self._cell_take(key, self.packets_per_cell) if key in self.members else []
                missing += int(not indices)
                short += int(0 < len(indices) < self.packets_per_cell)
                result.extend(indices)
                cell_counts.append({'tx': tx, 'rx': rx, 'day': day, 'actual': len(indices), 'requested': self.packets_per_cell})
        requested = self.tx_per_batch * len(self.receivers) * self.packets_per_cell
        counts = {'batch_index': self.batch_index, 'requested': requested,
                  'actual': len(result), 'missing_samples': requested - len(result),
                  'missing_cells': missing, 'short_cells': short,
                  'missing_tx_slots': max(0, self.tx_per_batch - len(txs)),
                  'selected_tx': txs, 'cells': cell_counts}
        self.batch_index += 1
        return result, counts

    def state_dict(self):
        return copy.deepcopy({'labels': self.labels, 'rx': self.rx, 'day': self.day, 'valid': self.valid,
                              'tx_per_batch': self.tx_per_batch, 'packets_per_cell': self.packets_per_cell,
                              'rng': self.rng.bit_generator.state, 'queues': self.queues,
                              'positions': self.positions, 'batch_index': self.batch_index,
                              'tx_queue': self.tx_queue, 'tx_position': self.tx_position})

    def load_state_dict(self, state):
        if any(not np.array_equal(state[name], getattr(self, name)) for name in ('labels', 'rx', 'day', 'valid')):
            raise ValueError('L sampler dataset identity/order mismatch')
        if any(state[name] != getattr(self, name) for name in ('tx_per_batch', 'packets_per_cell')):
            raise ValueError('L sampler batch configuration mismatch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.queues, self.positions = copy.deepcopy(state['queues']), dict(state['positions'])
        self.batch_index = int(state['batch_index'])
        self.tx_queue, self.tx_position = list(state['tx_queue']), int(state['tx_position'])
