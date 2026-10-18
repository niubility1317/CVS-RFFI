"""Replay original P43 As occurrences only after exact current-L pairing checks."""
from copy import deepcopy

import numpy as np


def _values(value, dtype=np.int64):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError('Integer visible labels, domains and physical-ID fields required')
    return np.asarray(array, dtype=dtype)


class OrderedReader:
    method = 'ORIGINAL_GENERATION_ORDER'

    def __init__(self, reader):
        self.reader = reader
        self._pending = None
        self.last_origin = None

    @property
    def calls(self):
        return self.reader.calls

    @property
    def packets(self):
        return self.reader.packets

    def __deepcopy__(self, memo):
        value = type(self)(deepcopy(self.reader, memo))
        memo[id(self)] = value
        return value

    def validate_origin(self, step, epoch, *, labels, domains, physical_ids):
        """Labels and compact domains come from this actual visible L batch."""
        spec = self.reader._shared['summary']
        if (type(step) is not int or type(epoch) is not int or step != self.calls + 1
                or not 1 <= step <= spec['updates']
                or epoch != (step-1)//spec['steps']+1 or self._pending is not None):
            raise ValueError('Validate exactly the next local tail As64 occurrence')
        arrays = self.reader._shared['arrays']
        begin, end = 64*(step-1), 64*step
        selected = np.flatnonzero((np.arange(128)+step)%2 != 0)
        y, d, ids = _values(labels), _values(domains), _values(physical_ids)
        if (y.shape != (64,) or d.shape != (64,) or ids.shape != (64,5)
                or not np.array_equal(y, arrays['labels'][begin:end])
                or not np.array_equal(d, arrays['domains'][begin:end])
                or not np.array_equal(ids, arrays['physical_ids'][begin:end])
                or not np.all(arrays['origin_steps'][begin:end] == step)
                or not np.array_equal(arrays['origin_slots'][begin:end], selected)):
            raise ValueError('Cached As64 does not match the current raw physical IDs/labels/domains/slots')
        self._pending = dict(step=step, epoch=epoch, count=64,
            origin_verified=True, selected_raw_indices=selected.tolist(),
            physical_ids=ids.tolist(), labels=y.tolist(), domains=d.tolist(),
            used_pool_indices=list(range(begin,end)))
        return dict(self._pending)

    def consume(self, step, epoch, device):
        if self._pending is None or self._pending['step'] != step or self._pending['epoch'] != epoch:
            raise ValueError('Validate the actual current raw-As origin before consuming its cache')
        expected = self._pending
        iq, labels, domains, metadata, record = self.reader.consume(step, epoch, device)
        if (record['used_pool_indices'] != expected['used_pool_indices']
                or _values(labels).tolist() != expected['labels']
                or _values(domains).tolist() != expected['domains']):
            raise ValueError('Ordered cache selection changed after origin validation')
        for j, row in enumerate(metadata):
            if (list(row['sample_id']) != expected['physical_ids'][j]
                    or list(row['physical_id']) != expected['physical_ids'][j]
                    or int(row['label']) != expected['labels'][j]
                    or int(row['domain']) != expected['domains'][j]
                    or int(row['origin_step']) != step
                    or int(row['origin_slot']) != expected['selected_raw_indices'][j]):
                raise ValueError('Cached channel metadata is not aligned to the current raw observation')
        record = dict(record, origin_verified=True, order=self.method,
                      selected_raw_indices=expected['selected_raw_indices'])
        self.last_origin = dict(expected)
        self._pending = None
        return iq, labels, domains, metadata, record

    def _report(self, original):
        result = dict(original)
        result.update(method=self.method, original_generation_order=True,
            pool_read_only=True, no_quality_weighting=True,
            new_channel_calls=0, pool_generated_this_run=0,
            origin_validated_steps=self.calls, origin_validated_observations=self.packets,
            one_use=True, consumed_pool_range=[0,self.packets],
            complete=self.calls == self.reader._shared['summary']['updates'])
        return result

    def audit(self):
        return self._report(self.reader.audit())

    def summary(self):
        return self._report(self.reader.summary())


def load(directory, *, seed):
    """Use P70 storage; substitute an identity order on a new Reader only."""
    from experiments.cvs_quality_risk70 import pool
    template = pool.load(directory, 'POOL_RANDOM', seed=seed)
    spec = template._shared['summary']
    if spec['N'] != 142080 or spec['updates'] != 2220 or spec['steps'] != 222:
        raise ValueError('Only the exact completed original P43 tail nominal pool')
    order = np.arange(spec['N'], dtype=np.int64).reshape(spec['updates'],64)
    order.flags.writeable = False
    reader = pool.Reader(template._shared, order, OrderedReader.method, seed)
    return OrderedReader(reader)
