"""S-BATCH-only synchronous loader; no workers or speculative sampling."""
from __future__ import annotations

import copy
import math

from torch.utils.data import default_collate

from .ecrs_sampling import MatchedLabeledSampler


class MatchedLabeledLoader:
    """Iterable loader with exact sampler continuation and explicit short cells.

    Length is ceil(L population / requested structured batch size). This is a
    batch budget, not a claim that all L samples occur exactly once per epoch.
    Short/missing cells are retained and reported in ``last_counts``. Iteration
    never resets the sampler, so repeated iterators and epochs continue it.
    """
    num_workers = 0
    persistent_workers = False
    drop_last = False

    def __init__(self, dataset, *, tx_per_batch=6, packets_per_cell=4, seed=0):
        # Check the role before touching index TX fields: U indices retain
        # hidden TX metadata internally but are never legal input here.
        if not bool(getattr(dataset, 'tx_label_visible', True)):
            raise ValueError('Matched L loader cannot inspect a TX-hidden U dataset')
        if not hasattr(dataset, 'index') or len(dataset.index) != len(dataset):
            raise ValueError('Matched L loader requires a direct per-sample .index')
        self.dataset = dataset
        self.sampler = MatchedLabeledSampler(
            [it.tx_i for it in dataset.index], [it.rx_i for it in dataset.index],
            [it.day_i for it in dataset.index], tx_per_batch=tx_per_batch,
            packets_per_cell=packets_per_cell, seed=seed)
        self.batch_size = self.sampler.tx_per_batch * len(self.sampler.receivers) * self.sampler.packets_per_cell
        self._length = math.ceil(len(dataset) / self.batch_size)
        self.batches_yielded = 0
        self.last_counts = None

    def __len__(self):
        return self._length

    def __iter__(self):
        for _ in range(len(self)):
            indices, counts = self.sampler.take_batch()
            self.last_counts = counts
            if not indices:
                # With day rotation, a fully missing selected TX grid can
                # occur. Do not fabricate data or invoke collate on no samples.
                raise RuntimeError(f'Matched L batch has no legal samples: {counts}')
            batch = default_collate([self.dataset[i] for i in indices])
            self.batches_yielded += 1
            yield batch

    def state_dict(self):
        return copy.deepcopy({'sampler': self.sampler.state_dict(),
                              'batches_yielded': self.batches_yielded,
                              'last_counts': self.last_counts})

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state['sampler'])
        self.batches_yielded = int(state['batches_yielded'])
        self.last_counts = copy.deepcopy(state['last_counts'])


def make_revision_l_loader(train_ds, args):
    """Return the isolated S-BATCH loader, or None to retain the original one."""
    if not bool(getattr(args, 'use_ecrs', False)) or getattr(args, 'ecrs_sampler_mode', 'legacy') == 'legacy':
        return None
    if getattr(args, 'ecrs_sampler_mode', 'legacy') != 'balanced_tx_rx':
        raise ValueError('Unsupported revision sampler mode')
    if str(getattr(args, 'ecrs_version', 'v1')) != 'v2':
        raise ValueError('Balanced TX/RX sampling is an explicit V2 S-BATCH control')
    if bool(getattr(args, 'ecrs_compute_only', False)):
        raise ValueError('Compute-only parity control cannot change L sampling')
    return MatchedLabeledLoader(train_ds, seed=int(getattr(args, 'seed', 0)))
